from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone
from typing import List, Optional
from app.models import (
    ExtractedClaim, ClaimType,
    ActiveState, BeliefHistory, RecentEvent, MemorySnapshot
)


# ── WRITE PATH ─────────────────────────────────────────────────────────────

def log_raw(db: Session, speaker: str, raw_text: str,
            emotion: str = None, scene: str = None, metadata: dict = {}):
    """Always log first. Immutable. Never queried during chat."""
    import json
    db.execute(text("""
        INSERT INTO memory_logs (speaker, raw_text, emotion, scene, metadata)
        VALUES (:speaker, :raw_text, :emotion, :scene, CAST(:metadata AS jsonb))
    """), {
        "speaker": speaker,
        "raw_text": raw_text,
        "emotion": emotion,
        "scene": scene,
        "metadata": json.dumps(metadata or {})
    })


def write_state(db: Session, claim: ExtractedClaim, emotion: str = None):
    """Close existing active state, insert new one."""
    # Close old version
    db.execute(text("""
        UPDATE state_memory
        SET valid_to = NOW()
        WHERE entity = :entity
          AND attribute = :attribute
          AND valid_to IS NULL
    """), {"entity": claim.entity, "attribute": claim.attribute})

    # Insert new version
    db.execute(text("""
        INSERT INTO state_memory (entity, attribute, value, confidence, emotion)
        VALUES (:entity, :attribute, :value, :confidence, :emotion)
    """), {
        "entity": claim.entity,
        "attribute": claim.attribute,
        "value": claim.value,
        "confidence": claim.confidence,
        "emotion": emotion
    })


def write_belief(db: Session, claim: ExtractedClaim):
    """Close existing active belief, insert new one."""
    db.execute(text("""
        UPDATE belief_memory
        SET valid_to = NOW()
        WHERE observer = :observer
          AND entity_or_event = :entity_or_event
          AND attribute = :attribute
          AND valid_to IS NULL
    """), {
        "observer": claim.observer,
        "entity_or_event": claim.entity_or_event,
        "attribute": claim.attribute
    })

    db.execute(text("""
        INSERT INTO belief_memory (observer, entity_or_event, attribute, value, confidence)
        VALUES (:observer, :entity_or_event, :attribute, :value, :confidence)
    """), {
        "observer": claim.observer,
        "entity_or_event": claim.entity_or_event,
        "attribute": claim.attribute,
        "value": claim.value,
        "confidence": claim.confidence
    })


def write_event(db: Session, claim: ExtractedClaim,
                emotion: str = None, scene: str = None, embedding: list = None):
    """Insert event. Immutable — never updated."""
    db.execute(text("""
        INSERT INTO event_memory (event_type, actor, description, emotion, scene, embedding)
        VALUES (:event_type, :actor, :description, :emotion, :scene, :embedding)
    """), {
        "event_type": claim.attribute or "general",
        "actor": claim.entity or "user",
        "description": claim.value,
        "emotion": emotion,
        "scene": scene,
        "embedding": embedding
    })


def update_frequency(db: Session, topic: str):
    """Deterministic frequency tracking. No LLM guessing."""
    if not topic:
        return
    db.execute(text("""
        INSERT INTO topic_frequency (topic, count, last_seen)
        VALUES (:topic, 1, NOW())
        ON CONFLICT (topic)
        DO UPDATE SET count = topic_frequency.count + 1, last_seen = NOW()
    """), {"topic": topic.lower()})


# ── READ PATH ──────────────────────────────────────────────────────────────

def get_active_states(
    db: Session,
    top_n: int = 20,
    entities: List[str] = None,
) -> List[ActiveState]:
    """O(1) indexed lookup of current truth. ~2ms.

    Pass `entities` to restrict results to specific entity names (e.g. ["son", "user"]).
    This keeps Elara's context focused rather than dumping every stored fact.
    """
    # Filter out known garbage entities that the extractor sometimes produces
    _JUNK_ENTITIES = {"assistant", "him", "he", "she", "they", "it"}

    if entities:
        safe = [e for e in entities if e not in _JUNK_ENTITIES]
        if safe:
            placeholders = ", ".join(f":e{i}" for i in range(len(safe)))
            params = {f"e{i}": e for i, e in enumerate(safe)}
            params["top_n"] = top_n
            rows = db.execute(text(f"""
                SELECT entity, attribute, value, confidence, emotion, valid_from
                FROM state_memory
                WHERE valid_to IS NULL
                  AND LOWER(entity) IN ({placeholders})
                ORDER BY valid_from DESC
                LIMIT :top_n
            """), params).fetchall()
        else:
            rows = []
    else:
        rows = db.execute(text("""
            SELECT entity, attribute, value, confidence, emotion, valid_from
            FROM state_memory
            WHERE valid_to IS NULL
              AND LOWER(entity) NOT IN ('assistant', 'him', 'he', 'she', 'they', 'it')
            ORDER BY valid_from DESC
            LIMIT :top_n
        """), {"top_n": top_n}).fetchall()

    return [ActiveState(
        entity=r[0], attribute=r[1], value=r[2],
        confidence=r[3], emotion=r[4], valid_from=r[5]
    ) for r in rows]


def get_belief_history(db: Session, subject: str = None) -> List[BeliefHistory]:
    """Return full belief history so LLM can narrate evolution."""
    query = """
        SELECT entity_or_event, attribute, value, confidence, valid_from, valid_to
        FROM belief_memory
        WHERE observer = 'user'
    """
    params = {}
    if subject:
        query += " AND (entity_or_event ILIKE :subject OR attribute ILIKE :subject)"
        params["subject"] = f"%{subject}%"

    query += " ORDER BY entity_or_event, valid_from ASC"
    rows = db.execute(text(query), params).fetchall()

    # Group by entity_or_event + attribute
    grouped: dict = {}
    for r in rows:
        key = f"{r[0]}::{r[1]}"
        if key not in grouped:
            grouped[key] = {
                "about": r[0], "attribute": r[1],
                "history": [], "current_value": None
            }
        entry = {
            "value": r[2],
            "confidence": r[3],
            "from": r[4].isoformat(),
            "to": r[5].isoformat() if r[5] else "current"
        }
        grouped[key]["history"].append(entry)
        if r[5] is None:
            grouped[key]["current_value"] = r[2]

    return [BeliefHistory(**v) for v in grouped.values()]


def get_recent_events(db: Session, limit: int = 5,
                      query_embedding: list = None) -> List[RecentEvent]:
    """Recent events, with optional semantic search."""
    if query_embedding:
        rows = db.execute(text("""
            SELECT event_type, actor, description, emotion, timestamp
            FROM event_memory
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """), {"embedding": str(query_embedding), "limit": limit}).fetchall()
    else:
        rows = db.execute(text("""
            SELECT event_type, actor, description, emotion, timestamp
            FROM event_memory
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"limit": limit}).fetchall()

    return [RecentEvent(
        event_type=r[0], actor=r[1], description=r[2],
        emotion=r[3], timestamp=r[4]
    ) for r in rows]


def get_last_n_turns(db: Session, n: int = 5) -> List[dict]:
    """Last N conversation turns for context window."""
    rows = db.execute(text("""
        SELECT speaker, raw_text, emotion, scene, timestamp
        FROM memory_logs
        ORDER BY timestamp DESC
        LIMIT :n
    """), {"n": n}).fetchall()

    return [{"speaker": r[0], "text": r[1], "emotion": r[2],
             "scene": r[3], "timestamp": r[4].isoformat()}
            for r in reversed(rows)]


# Entities the system tracks; used to filter active_states to what's relevant.
_KNOWN_ENTITIES = {"user", "son", "daughter", "neighbour", "neighbor", "wife", "husband",
                   "friend", "brother", "sister", "mother", "father", "doctor", "caregiver"}


def _entities_from_question(question: str) -> List[str]:
    """Return entity names mentioned in the question so we only fetch relevant states."""
    if not question:
        return []
    q = question.lower()
    return [e for e in _KNOWN_ENTITIES if e in q]


def assemble_snapshot(db: Session, intent: str,
                      question: str = None,
                      query_embedding: list = None) -> MemorySnapshot:
    """
    Assemble the clean resolved snapshot.
    The LLM never queries memory — it only sees this.
    """
    # For focused queries, restrict to entities mentioned in the question so we
    # don't flood Elara's context with unrelated facts.
    focused_entities = _entities_from_question(question)
    active_states = get_active_states(
        db,
        entities=focused_entities if focused_entities else None,
    )
    last_5 = get_last_n_turns(db)

    # Branch based on intent
    if intent == "CURRENT_STATE":
        beliefs = []
        events = get_recent_events(db, limit=3)

    elif intent == "PAST_BELIEF":
        subject = question.split()[-1] if question else None
        beliefs = get_belief_history(db, subject=subject)
        events = []

    elif intent == "EVENT":
        beliefs = []
        events = get_recent_events(db, limit=10, query_embedding=query_embedding)

    elif intent == "HISTORY":
        beliefs = get_belief_history(db)
        events = get_recent_events(db, limit=5)

    else:  # GENERAL
        beliefs = get_belief_history(db)
        events = get_recent_events(db, limit=3)

    return MemorySnapshot(
        active_states=active_states,
        relevant_beliefs=beliefs,
        recent_events=events,
        last_5_turns=last_5,
        intent=intent
    )