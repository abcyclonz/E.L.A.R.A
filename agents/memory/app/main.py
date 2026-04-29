from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import get_db, check_connection
from app.models import (
    ProcessRequest, ProcessResponse, RetrieveRequest, MemorySnapshot,
    EpisodeRequest, RecallRequest, RecallResponse, Episode,
)
from app.extractor import extract_claims, classify_intent, embed_text
from app.memory import (
    log_raw, write_state, write_belief, write_event,
    update_frequency, assemble_snapshot,
    write_episode, get_similar_episodes,
    get_grounding_facts,
)
from app.models import ClaimType


# ── DDL ────────────────────────────────────────────────────────────────────

_EPISODES_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    id              SERIAL PRIMARY KEY,
    speaker_id      TEXT NOT NULL DEFAULT 'user',
    user_turn       TEXT NOT NULL,
    assistant_turn  TEXT,
    embedding       VECTOR(768),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_episodes_speaker   ON episodes(speaker_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
"""

_SPEAKER_ID_MIGRATION = """
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS speaker_id TEXT NOT NULL DEFAULT 'user';
ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS speaker_id TEXT NOT NULL DEFAULT 'user';
ALTER TABLE event_memory  ADD COLUMN IF NOT EXISTS speaker_id TEXT NOT NULL DEFAULT 'user';
"""

# Adds importance, stability, access_count, last_accessed to all memory layers.
# Safe on existing data — new columns default to neutral values.
_PRIORITY_MIGRATION = """
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS importance    FLOAT       DEFAULT 0.5;
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS stability     TEXT        DEFAULT 'stable';
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS access_count  INT         DEFAULT 0;
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;

ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS importance    FLOAT       DEFAULT 0.5;
ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS stability     TEXT        DEFAULT 'stable';
ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS access_count  INT         DEFAULT 0;
ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ;

ALTER TABLE event_memory  ADD COLUMN IF NOT EXISTS importance    FLOAT       DEFAULT 0.5;
ALTER TABLE event_memory  ADD COLUMN IF NOT EXISTS stability     TEXT        DEFAULT 'stable';

CREATE INDEX IF NOT EXISTS idx_state_importance  ON state_memory(importance DESC);
CREATE INDEX IF NOT EXISTS idx_belief_importance ON belief_memory(importance DESC);
"""

_GROUNDING_MIGRATION = """
ALTER TABLE state_memory  ADD COLUMN IF NOT EXISTS is_grounding BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE belief_memory ADD COLUMN IF NOT EXISTS is_grounding BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_state_grounding  ON state_memory(is_grounding, speaker_id) WHERE is_grounding = TRUE;
CREATE INDEX IF NOT EXISTS idx_belief_grounding ON belief_memory(is_grounding, speaker_id) WHERE is_grounding = TRUE;
UPDATE state_memory  SET is_grounding = TRUE
  WHERE importance >= 0.85 AND stability = 'permanent' AND valid_to IS NULL AND is_grounding = FALSE;
UPDATE belief_memory SET is_grounding = TRUE
  WHERE importance >= 0.85 AND stability = 'permanent' AND valid_to IS NULL AND is_grounding = FALSE;
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        check_connection()
        print("✅ Database connected")
        with get_db() as db:
            from sqlalchemy import text as _t
            db.execute(_t(_EPISODES_DDL))
            db.execute(_t(_SPEAKER_ID_MIGRATION))
            db.execute(_t(_PRIORITY_MIGRATION))
            db.execute(_t(_GROUNDING_MIGRATION))
        print("✅ Schema ready (episodes + speaker_id + priority + grounding columns)")
    except Exception as e:
        print(f"❌ Startup error: {e}")
    yield


app = FastAPI(
    title="Memory Agent",
    description="Versioned multimodal memory with salience scoring",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── PROCESS ────────────────────────────────────────────────────────────────

@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    """
    Main endpoint — called on every user turn.
    1. Logs raw input
    2. Extracts typed claims (with importance + stability)
    3. Writes to correct memory layer
    4. Returns salience-filtered snapshot for LLM
    """
    with get_db() as db:
        log_raw(db, req.speaker, req.text, req.emotion, req.scene, req.metadata or {})

        extraction = extract_claims(req.text, req.emotion, req.scene)
        speaker_id = req.speaker or "user"

        for claim in extraction.claims:
            if claim.type == ClaimType.STATE:
                if claim.entity and claim.attribute and claim.value:
                    if claim.corrects_entity:
                        from sqlalchemy import text as _text
                        db.execute(_text("""
                            UPDATE state_memory SET valid_to = NOW()
                            WHERE entity = :entity AND attribute = :attribute
                              AND speaker_id = :speaker_id AND valid_to IS NULL
                        """), {"entity": claim.corrects_entity, "attribute": claim.attribute,
                               "speaker_id": speaker_id})
                        print(f"[Memory] Closed wrong state: {claim.corrects_entity}.{claim.attribute}")
                    write_state(db, claim, emotion=req.emotion, speaker_id=speaker_id)

            elif claim.type == ClaimType.BELIEF:
                if claim.entity_or_event and claim.attribute and claim.value:
                    write_belief(db, claim, speaker_id=speaker_id)

            elif claim.type == ClaimType.EVENT:
                if claim.value:
                    embedding = embed_text(claim.value)
                    write_event(db, claim, req.emotion, req.scene, embedding, speaker_id=speaker_id)

            if claim.topic:
                update_frequency(db, claim.topic)

        intent   = classify_intent(req.text)
        snapshot = assemble_snapshot(
            db, intent, question=req.text,
            speaker_id=speaker_id, llm_rerank=False,
        )

    return ProcessResponse(
        status="ok",
        claims_extracted=len(extraction.claims),
        snapshot=snapshot,
    )


# ── RETRIEVE ───────────────────────────────────────────────────────────────

@app.post("/retrieve", response_model=MemorySnapshot)
def retrieve(req: RetrieveRequest):
    """
    Called when orchestrator needs context before replying.
    Applies salience filtering + optional LLM relevance reranking.
    """
    with get_db() as db:
        intent = classify_intent(req.question)

        query_embedding = None
        if intent == "EVENT":
            query_embedding = embed_text(req.question)

        snapshot = assemble_snapshot(
            db, intent,
            question=req.question,
            query_embedding=query_embedding,
            speaker_id=req.speaker_id,
            llm_rerank=req.llm_rerank,
        )

    return snapshot


# ── EPISODE ────────────────────────────────────────────────────────────────

@app.post("/episode", status_code=201)
def store_episode(req: EpisodeRequest):
    combined  = req.user_turn
    if req.assistant_turn:
        combined += "\n" + req.assistant_turn
    embedding = embed_text(combined)
    with get_db() as db:
        write_episode(db, req.speaker_id, req.user_turn, req.assistant_turn, embedding)
    return {"status": "ok"}


# ── RECALL ─────────────────────────────────────────────────────────────────

@app.post("/recall", response_model=RecallResponse)
def recall_episodes(req: RecallRequest):
    query_embedding = embed_text(req.question)
    with get_db() as db:
        rows = get_similar_episodes(
            db, query_embedding=query_embedding,
            speaker_id=req.speaker_id, top_k=req.top_k,
        )
    episodes = [
        Episode(
            id=r["id"], speaker_id=r["speaker_id"],
            user_turn=r["user_turn"], assistant_turn=r["assistant_turn"],
            timestamp=r["timestamp"], similarity=r["similarity"],
        )
        for r in rows
        if r["similarity"] > 0.5
    ]
    return RecallResponse(episodes=episodes)


# ── GROUNDING ──────────────────────────────────────────────────────────────

@app.get("/grounding/{speaker_id}")
def grounding(speaker_id: str):
    """
    Return all grounding facts for a speaker — high-importance permanent facts
    that Elara should always carry in context regardless of the current topic.
    Never filtered by salience or query.
    """
    with get_db() as db:
        facts = get_grounding_facts(db, speaker_id)
    return {"grounding": facts}


# ── HEALTH ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        check_connection()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── DEBUG ───────────────────────────────────────────────────────────────────

@app.get("/debug/states")
def debug_states():
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT entity, attribute, value, confidence, emotion,
                   importance, stability, access_count, valid_from
            FROM state_memory WHERE valid_to IS NULL
            ORDER BY importance DESC, valid_from DESC
        """)).fetchall()
        return [
            {
                "entity": r[0], "attribute": r[1], "value": r[2],
                "confidence": r[3], "emotion": r[4],
                "importance": r[5], "stability": r[6],
                "access_count": r[7], "since": r[8].isoformat(),
            }
            for r in rows
        ]


@app.get("/debug/beliefs")
def debug_beliefs():
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT observer, entity_or_event, attribute, value, confidence,
                   importance, stability, access_count, valid_from, valid_to
            FROM belief_memory ORDER BY importance DESC, valid_from DESC
        """)).fetchall()
        return [
            {
                "observer": r[0], "about": r[1], "attribute": r[2],
                "value": r[3], "confidence": r[4],
                "importance": r[5], "stability": r[6],
                "access_count": r[7],
                "from": r[8].isoformat(),
                "to": r[9].isoformat() if r[9] else "current",
            }
            for r in rows
        ]


@app.get("/debug/logs")
def debug_logs(limit: int = 20):
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT speaker, raw_text, emotion, scene, timestamp
            FROM memory_logs ORDER BY timestamp DESC LIMIT :limit
        """), {"limit": limit}).fetchall()
        return [{"speaker": r[0], "text": r[1], "emotion": r[2],
                 "scene": r[3], "timestamp": r[4].isoformat()}
                for r in rows]


@app.delete("/debug/states/purge")
def debug_purge_states():
    _GARBAGE_ENTITIES   = ("assistant", "him", "he", "she", "they", "it")
    _GARBAGE_VALUE_PATTERN = "%(corrects%"
    with get_db() as db:
        from sqlalchemy import text
        result = db.execute(text("""
            DELETE FROM state_memory
            WHERE valid_to IS NULL
              AND (
                LOWER(entity) = ANY(:junk_entities)
                OR value ILIKE :garbage_val
              )
        """), {
            "junk_entities": list(_GARBAGE_ENTITIES),
            "garbage_val":   _GARBAGE_VALUE_PATTERN,
        })
        db.commit()
        return {"deleted": result.rowcount}


@app.get("/debug/salience")
def debug_salience(speaker_id: str = "user", threshold: float = 0.0):
    """Show all active states with their salience scores. Useful for tuning."""
    from app.memory import compute_salience, SALIENCE_THRESHOLD
    from datetime import datetime, timezone
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT entity, attribute, value, importance, stability,
                   access_count, valid_from
            FROM state_memory
            WHERE valid_to IS NULL AND speaker_id = :speaker_id
            ORDER BY valid_from DESC
        """), {"speaker_id": speaker_id}).fetchall()

    now = datetime.now(timezone.utc)
    result = []
    for r in rows:
        age_days = (now - r[6].replace(tzinfo=timezone.utc)).total_seconds() / 86400
        salience = compute_salience(
            float(r[3] or 0.5), age_days, r[4] or "stable", int(r[5] or 0)
        )
        if salience >= threshold:
            result.append({
                "entity": r[0], "attribute": r[1], "value": r[2],
                "importance": r[3], "stability": r[4],
                "access_count": r[5], "age_days": round(age_days, 1),
                "salience": salience,
                "active": salience >= SALIENCE_THRESHOLD,
            })

    result.sort(key=lambda x: x["salience"], reverse=True)
    return result
