from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone
from typing import List, Optional
from app.models import (
    ExtractedClaim, ClaimType,
    ActiveState, BeliefHistory, RecentEvent, MemorySnapshot
)
import math


# ── Salience scoring ────────────────────────────────────────────────────────

# Half-life in days per stability tier
_HALF_LIVES = {
    "permanent": 3650,   # 10 years — name, birthplace, deceased relatives
    "stable":    365,    # 1 year  — job, home, long-term relationships
    "transient": 14,     # 2 weeks — mood, plans, recent events
}

# Facts below this score are excluded from normal retrieval snapshots.
# They still exist in the DB and can be surfaced via explicit search.
SALIENCE_THRESHOLD = 0.10


def _recency_weight(age_days: float, stability: str) -> float:
    half_life = _HALF_LIVES.get(stability, 180)
    return 0.5 ** (age_days / half_life)


def compute_salience(importance: float, age_days: float,
                     stability: str, access_count: int) -> float:
    """
    Salience in [0, 1].

    importance    × 0.40  — how fundamental is this fact
    recency       × 0.40  — half-life decay based on stability tier
    frequency     × 0.20  — saturates at 10 accesses
    """
    recency = _recency_weight(age_days, stability)
    freq    = min(access_count / 10.0, 1.0)
    return round(importance * 0.40 + recency * 0.40 + freq * 0.20, 4)


# ── WRITE PATH ──────────────────────────────────────────────────────────────

def log_raw(db: Session, speaker: str, raw_text: str,
            emotion: str = None, scene: str = None, metadata: dict = {}):
    import json
    db.execute(text("""
        INSERT INTO memory_logs (speaker, raw_text, emotion, scene, metadata)
        VALUES (:speaker, :raw_text, :emotion, :scene, CAST(:metadata AS jsonb))
    """), {
        "speaker": speaker, "raw_text": raw_text,
        "emotion": emotion, "scene": scene,
        "metadata": json.dumps(metadata or {}),
    })


def write_state(db: Session, claim: ExtractedClaim,
                emotion: str = None, speaker_id: str = "user"):
    db.execute(text("""
        UPDATE state_memory
        SET valid_to = NOW()
        WHERE entity = :entity AND attribute = :attribute
          AND speaker_id = :speaker_id AND valid_to IS NULL
    """), {"entity": claim.entity, "attribute": claim.attribute, "speaker_id": speaker_id})

    db.execute(text("""
        INSERT INTO state_memory
               (entity, attribute, value, confidence, emotion,
                importance, stability, speaker_id)
        VALUES (:entity, :attribute, :value, :confidence, :emotion,
                :importance, :stability, :speaker_id)
    """), {
        "entity":     claim.entity,
        "attribute":  claim.attribute,
        "value":      claim.value,
        "confidence": claim.confidence,
        "emotion":    emotion,
        "importance": claim.importance,
        "stability":  claim.stability,
        "speaker_id": speaker_id,
    })


def write_belief(db: Session, claim: ExtractedClaim, speaker_id: str = "user"):
    db.execute(text("""
        UPDATE belief_memory
        SET valid_to = NOW()
        WHERE observer = :observer AND entity_or_event = :entity_or_event
          AND attribute = :attribute AND speaker_id = :speaker_id AND valid_to IS NULL
    """), {
        "observer": claim.observer, "entity_or_event": claim.entity_or_event,
        "attribute": claim.attribute, "speaker_id": speaker_id,
    })

    db.execute(text("""
        INSERT INTO belief_memory
               (observer, entity_or_event, attribute, value, confidence,
                importance, stability, speaker_id)
        VALUES (:observer, :entity_or_event, :attribute, :value, :confidence,
                :importance, :stability, :speaker_id)
    """), {
        "observer":      claim.observer,
        "entity_or_event": claim.entity_or_event,
        "attribute":     claim.attribute,
        "value":         claim.value,
        "confidence":    claim.confidence,
        "importance":    claim.importance,
        "stability":     claim.stability,
        "speaker_id":    speaker_id,
    })


def write_event(db: Session, claim: ExtractedClaim,
                emotion: str = None, scene: str = None,
                embedding: list = None, speaker_id: str = "user"):
    db.execute(text("""
        INSERT INTO event_memory
               (event_type, actor, description, emotion, scene, embedding,
                importance, stability, speaker_id)
        VALUES (:event_type, :actor, :description, :emotion, :scene, :embedding,
                :importance, :stability, :speaker_id)
    """), {
        "event_type":  claim.attribute or "general",
        "actor":       claim.entity or "user",
        "description": claim.value,
        "emotion":     emotion,
        "scene":       scene,
        "embedding":   embedding,
        "importance":  claim.importance,
        "stability":   claim.stability,
        "speaker_id":  speaker_id,
    })


def update_frequency(db: Session, topic: str):
    if not topic:
        return
    db.execute(text("""
        INSERT INTO topic_frequency (topic, count, last_seen)
        VALUES (:topic, 1, NOW())
        ON CONFLICT (topic)
        DO UPDATE SET count = topic_frequency.count + 1, last_seen = NOW()
    """), {"topic": topic.lower()})


def bump_access_count(db: Session, table: str, row_ids: List[int]):
    """Increment access_count and last_accessed for retrieved rows."""
    if not row_ids:
        return
    placeholders = ", ".join(f":id{i}" for i in range(len(row_ids)))
    params = {f"id{i}": rid for i, rid in enumerate(row_ids)}
    db.execute(text(f"""
        UPDATE {table}
        SET access_count  = access_count + 1,
            last_accessed = NOW()
        WHERE id IN ({placeholders})
    """), params)


# ── READ PATH ───────────────────────────────────────────────────────────────

_JUNK_ENTITIES = {"assistant", "him", "he", "she", "they", "it"}


def get_active_states(
    db: Session,
    top_n: int = 20,
    entities: List[str] = None,
    speaker_id: str = "user",
) -> List[ActiveState]:
    """
    Retrieve current active states, scored by salience.
    Facts below SALIENCE_THRESHOLD are excluded (soft forgetting).
    """
    if entities:
        safe = [e for e in entities if e not in _JUNK_ENTITIES]
        if safe:
            placeholders = ", ".join(f":e{i}" for i in range(len(safe)))
            params = {f"e{i}": e for i, e in enumerate(safe)}
            params.update({"speaker_id": speaker_id})
            rows = db.execute(text(f"""
                SELECT id, entity, attribute, value, confidence, emotion,
                       valid_from, importance, stability, access_count
                FROM state_memory
                WHERE valid_to IS NULL AND speaker_id = :speaker_id
                  AND LOWER(entity) IN ({placeholders})
                ORDER BY valid_from DESC
                LIMIT 100
            """), params).fetchall()
        else:
            rows = []
    else:
        rows = db.execute(text("""
            SELECT id, entity, attribute, value, confidence, emotion,
                   valid_from, importance, stability, access_count
            FROM state_memory
            WHERE valid_to IS NULL AND speaker_id = :speaker_id
              AND LOWER(entity) NOT IN ('assistant', 'him', 'he', 'she', 'they', 'it')
            ORDER BY valid_from DESC
            LIMIT 100
        """), {"speaker_id": speaker_id}).fetchall()

    now = datetime.now(timezone.utc)
    scored = []
    for r in rows:
        age_days = (now - r[6].replace(tzinfo=timezone.utc)).total_seconds() / 86400
        importance   = float(r[7] or 0.5)
        stability    = r[8] or "stable"
        access_count = int(r[9] or 0)
        salience     = compute_salience(importance, age_days, stability, access_count)
        if salience >= SALIENCE_THRESHOLD:
            scored.append((salience, r))

    # Sort by salience descending, return top_n
    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    retrieved_ids = []
    for salience, r in scored[:top_n]:
        retrieved_ids.append(r[0])
        result.append(ActiveState(
            entity=r[1], attribute=r[2], value=r[3],
            confidence=float(r[4]), emotion=r[5], valid_from=r[6],
            importance=float(r[7] or 0.5),
            stability=r[8] or "stable",
            access_count=int(r[9] or 0),
            salience_score=salience,
        ))

    if retrieved_ids:
        bump_access_count(db, "state_memory", retrieved_ids)

    return result


def get_belief_history(
    db: Session,
    subject: str = None,
    speaker_id: str = "user",
) -> List[BeliefHistory]:
    query  = """
        SELECT id, entity_or_event, attribute, value, confidence,
               valid_from, valid_to, importance, stability, access_count
        FROM belief_memory
        WHERE observer = 'user' AND speaker_id = :speaker_id
    """
    params = {"speaker_id": speaker_id}
    if subject:
        query  += " AND (entity_or_event ILIKE :subject OR attribute ILIKE :subject)"
        params["subject"] = f"%{subject}%"
    query += " ORDER BY entity_or_event, valid_from ASC"

    rows = db.execute(text(query), params).fetchall()

    now     = datetime.now(timezone.utc)
    grouped: dict = {}
    for r in rows:
        key = f"{r[1]}::{r[2]}"
        if key not in grouped:
            grouped[key] = {
                "about": r[1], "attribute": r[2],
                "history": [], "current_value": None,
                "importance": float(r[7] or 0.5),
                "stability":  r[8] or "stable",
                "_ids": [], "_access_counts": [], "_ages": [],
            }
        entry = {
            "value": r[3], "confidence": float(r[4]),
            "from": r[5].isoformat(),
            "to": r[6].isoformat() if r[6] else "current",
        }
        grouped[key]["history"].append(entry)
        if r[6] is None:
            grouped[key]["current_value"] = r[3]
        grouped[key]["_ids"].append(r[0])
        grouped[key]["_access_counts"].append(int(r[9] or 0))
        age = (now - r[5].replace(tzinfo=timezone.utc)).total_seconds() / 86400
        grouped[key]["_ages"].append(age)

    result = []
    retrieved_ids = []
    for g in grouped.values():
        # Use the most-recent entry's age for salience
        min_age  = min(g["_ages"]) if g["_ages"] else 0
        max_acc  = max(g["_access_counts"]) if g["_access_counts"] else 0
        salience = compute_salience(g["importance"], min_age, g["stability"], max_acc)
        if salience >= SALIENCE_THRESHOLD:
            retrieved_ids.extend(g["_ids"])
            result.append(BeliefHistory(
                about=g["about"], attribute=g["attribute"],
                history=g["history"], current_value=g["current_value"],
                importance=g["importance"], stability=g["stability"],
                salience_score=salience,
            ))

    if retrieved_ids:
        bump_access_count(db, "belief_memory", retrieved_ids)

    result.sort(key=lambda b: b.salience_score, reverse=True)
    return result


def get_recent_events(
    db: Session,
    limit: int = 5,
    query_embedding: list = None,
    speaker_id: str = "user",
) -> List[RecentEvent]:
    if query_embedding:
        rows = db.execute(text("""
            SELECT id, event_type, actor, description, emotion, timestamp,
                   importance, stability
            FROM event_memory
            WHERE speaker_id = :speaker_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """), {"embedding": str(query_embedding), "limit": limit * 3,
               "speaker_id": speaker_id}).fetchall()
    else:
        rows = db.execute(text("""
            SELECT id, event_type, actor, description, emotion, timestamp,
                   importance, stability
            FROM event_memory
            WHERE speaker_id = :speaker_id
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"limit": limit * 3, "speaker_id": speaker_id}).fetchall()

    now    = datetime.now(timezone.utc)
    scored = []
    for r in rows:
        age_days   = (now - r[5].replace(tzinfo=timezone.utc)).total_seconds() / 86400
        importance = float(r[6] or 0.5)
        stability  = r[7] or "stable"
        salience   = compute_salience(importance, age_days, stability, 0)
        if salience >= SALIENCE_THRESHOLD:
            scored.append((salience, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        RecentEvent(
            event_type=r[1], actor=r[2], description=r[3],
            emotion=r[4], timestamp=r[5],
            importance=float(r[6] or 0.5),
            salience_score=salience,
        )
        for salience, r in scored[:limit]
    ]


# ── LLM relevance reranking ─────────────────────────────────────────────────

def rerank_by_relevance(
    question: str,
    states: List[ActiveState],
) -> List[ActiveState]:
    """
    Post-filter: LLM scores each state's relevance to the question.
    Final score = salience × 0.5 + llm_relevance × 0.5.
    Keeps top-N by combined score; very low combined scores are dropped.
    """
    from app.extractor import score_relevance

    if not states:
        return states

    now = datetime.now(timezone.utc)
    candidates = [
        {
            "entity":     s.entity,
            "attribute":  s.attribute,
            "value":      s.value,
            "importance": s.importance,
            "stability":  s.stability,
            "age_days":   (now - s.valid_from.replace(tzinfo=timezone.utc)).total_seconds() / 86400,
        }
        for s in states
    ]

    llm_scores = score_relevance(question, candidates)   # list[float] in [0,1]

    combined = []
    for state, llm in zip(states, llm_scores):
        combined_score = state.salience_score * 0.5 + llm * 0.5
        combined.append((combined_score, state))

    combined.sort(key=lambda x: x[0], reverse=True)

    # Drop facts that score below threshold even after reranking
    result = [s for score, s in combined if score >= SALIENCE_THRESHOLD]
    print(f"[Rerank] {len(states)} → {len(result)} after LLM relevance filter")
    return result


# ── Episodic memory ─────────────────────────────────────────────────────────

def write_episode(db: Session, speaker_id: str, user_turn: str,
                  assistant_turn: str = None, embedding: list = None) -> None:
    db.execute(text("""
        INSERT INTO episodes (speaker_id, user_turn, assistant_turn, embedding)
        VALUES (:speaker_id, :user_turn, :assistant_turn, :embedding)
    """), {
        "speaker_id":    speaker_id,
        "user_turn":     user_turn,
        "assistant_turn": assistant_turn,
        "embedding":     str(embedding) if embedding else None,
    })


def get_similar_episodes(db: Session, query_embedding: list,
                         speaker_id: str = None, top_k: int = 3) -> list:
    params: dict = {"embedding": str(query_embedding), "top_k": top_k}
    speaker_filter = "AND speaker_id = :speaker_id" if speaker_id else ""
    if speaker_id:
        params["speaker_id"] = speaker_id

    rows = db.execute(text(f"""
        SELECT id, speaker_id, user_turn, assistant_turn, timestamp,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM episodes
        WHERE embedding IS NOT NULL {speaker_filter}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """), params).fetchall()

    return [
        {
            "id": r[0], "speaker_id": r[1], "user_turn": r[2],
            "assistant_turn": r[3], "timestamp": r[4], "similarity": float(r[5]),
        }
        for r in rows
    ]


def get_last_n_turns(db: Session, n: int = 5) -> List[dict]:
    rows = db.execute(text("""
        SELECT speaker, raw_text, emotion, scene, timestamp
        FROM memory_logs
        ORDER BY timestamp DESC
        LIMIT :n
    """), {"n": n}).fetchall()
    return [
        {"speaker": r[0], "text": r[1], "emotion": r[2],
         "scene": r[3], "timestamp": r[4].isoformat()}
        for r in reversed(rows)
    ]


# ── Assembly ─────────────────────────────────────────────────────────────────

_KNOWN_ENTITIES = {
    "user", "son", "daughter", "neighbour", "neighbor",
    "wife", "husband", "friend", "brother", "sister",
    "mother", "father", "doctor", "caregiver",
}


def _entities_from_question(question: str) -> List[str]:
    if not question:
        return []
    q = question.lower()
    return [e for e in _KNOWN_ENTITIES if e in q]


def assemble_snapshot(
    db: Session,
    intent: str,
    question: str = None,
    query_embedding: list = None,
    speaker_id: str = "user",
    llm_rerank: bool = False,
) -> MemorySnapshot:
    """
    Assemble the clean resolved snapshot.
    Salience filtering is always applied.
    LLM reranking is applied when llm_rerank=True (retrieval queries).
    """
    focused_entities = _entities_from_question(question)
    active_states    = get_active_states(
        db,
        entities=focused_entities if focused_entities else None,
        speaker_id=speaker_id,
    )

    # Optional: LLM relevance rerank for retrieval queries
    if llm_rerank and question and active_states:
        active_states = rerank_by_relevance(question, active_states)

    last_5 = get_last_n_turns(db)

    if intent == "CURRENT_STATE":
        beliefs = []
        events  = get_recent_events(db, limit=3, speaker_id=speaker_id)

    elif intent == "PAST_BELIEF":
        subject = question.split()[-1] if question else None
        beliefs = get_belief_history(db, subject=subject, speaker_id=speaker_id)
        events  = []

    elif intent == "EVENT":
        beliefs = []
        events  = get_recent_events(db, limit=10, query_embedding=query_embedding,
                                    speaker_id=speaker_id)

    elif intent == "HISTORY":
        beliefs = get_belief_history(db, speaker_id=speaker_id)
        events  = get_recent_events(db, limit=5, speaker_id=speaker_id)

    else:  # GENERAL
        beliefs = get_belief_history(db, speaker_id=speaker_id)
        events  = get_recent_events(db, limit=3, speaker_id=speaker_id)

    return MemorySnapshot(
        active_states=active_states,
        relevant_beliefs=beliefs,
        recent_events=events,
        last_5_turns=last_5,
        intent=intent,
    )
