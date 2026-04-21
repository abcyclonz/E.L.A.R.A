from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import get_db, check_connection
from app.models import ProcessRequest, ProcessResponse, RetrieveRequest, MemorySnapshot
from app.extractor import extract_claims, classify_intent, embed_text
from app.memory import (
    log_raw, write_state, write_belief, write_event,
    update_frequency, assemble_snapshot
)
from app.models import ClaimType


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        check_connection()
        print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
    yield


app = FastAPI(
    title="Memory Agent",
    description="Versioned multimodal memory for your multiagent system",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── PROCESS  (write + snapshot in one call) ────────────────────────────────

@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    """
    Main endpoint. Orchestrator calls this on every user turn.
    1. Logs raw input
    2. Extracts typed claims
    3. Writes to correct memory layer
    4. Returns resolved snapshot for LLM
    """
    with get_db() as db:
        # Step 1 — Always log first
        log_raw(db, req.speaker, req.text, req.emotion, req.scene, req.metadata or {})

        # Step 2 — Extract claims via Gemini
        extraction = extract_claims(req.text, req.emotion, req.scene)

        # Step 3 — Write each claim to correct layer
        for claim in extraction.claims:
            if claim.type == ClaimType.STATE:
                if claim.entity and claim.attribute and claim.value:
                    # If this corrects a previously wrong entity, close its state first
                    if claim.corrects_entity:
                        from sqlalchemy import text as _text
                        db.execute(_text("""
                            UPDATE state_memory SET valid_to = NOW()
                            WHERE entity = :entity AND attribute = :attribute AND valid_to IS NULL
                        """), {"entity": claim.corrects_entity, "attribute": claim.attribute})
                        print(f"[Memory] Closed wrong state: {claim.corrects_entity}.{claim.attribute}")
                    write_state(db, claim, emotion=req.emotion)

            elif claim.type == ClaimType.BELIEF:
                if claim.entity_or_event and claim.attribute and claim.value:
                    write_belief(db, claim)

            elif claim.type == ClaimType.EVENT:
                if claim.value:
                    embedding = embed_text(claim.value)
                    write_event(db, claim, req.emotion, req.scene, embedding)

            if claim.topic:
                update_frequency(db, claim.topic)

        # Step 4 — Assemble clean snapshot
        intent = classify_intent(req.text)
        snapshot = assemble_snapshot(db, intent, question=req.text)

    return ProcessResponse(
        status="ok",
        claims_extracted=len(extraction.claims),
        snapshot=snapshot
    )


# ── RETRIEVE  (read-only, intent-aware) ────────────────────────────────────

@app.post("/retrieve", response_model=MemorySnapshot)
def retrieve(req: RetrieveRequest):
    """
    Called when orchestrator needs context before generating a reply.
    Intent is classified, correct memory layer is queried.
    Returns clean snapshot — no raw logs, no conflicting data.
    """
    with get_db() as db:
        intent = classify_intent(req.question)

        query_embedding = None
        if intent == "EVENT":
            query_embedding = embed_text(req.question)

        snapshot = assemble_snapshot(
            db, intent,
            question=req.question,
            query_embedding=query_embedding
        )

    return snapshot


# ── HEALTH ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        check_connection()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── DEBUG (remove in production) ───────────────────────────────────────────

@app.get("/debug/states")
def debug_states():
    """View all active states. Dev only."""
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text(
            "SELECT entity, attribute, value, confidence, emotion, valid_from "
            "FROM state_memory WHERE valid_to IS NULL ORDER BY valid_from DESC"
        )).fetchall()
        return [{"entity": r[0], "attribute": r[1], "value": r[2],
                 "confidence": r[3], "emotion": r[4], "since": r[5].isoformat()}
                for r in rows]


@app.get("/debug/beliefs")
def debug_beliefs():
    """View full belief history. Dev only."""
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text(
            "SELECT observer, entity_or_event, attribute, value, confidence, "
            "valid_from, valid_to FROM belief_memory ORDER BY valid_from DESC"
        )).fetchall()
        return [{"observer": r[0], "about": r[1], "attribute": r[2],
                 "value": r[3], "confidence": r[4],
                 "from": r[5].isoformat(),
                 "to": r[6].isoformat() if r[6] else "current"}
                for r in rows]


@app.get("/debug/logs")
def debug_logs(limit: int = 20):
    """View raw log. Dev only."""
    with get_db() as db:
        from sqlalchemy import text
        rows = db.execute(text(
            "SELECT speaker, raw_text, emotion, scene, timestamp "
            "FROM memory_logs ORDER BY timestamp DESC LIMIT :limit"
        ), {"limit": limit}).fetchall()
        return [{"speaker": r[0], "text": r[1], "emotion": r[2],
                 "scene": r[3], "timestamp": r[4].isoformat()}
                for r in rows]


@app.delete("/debug/states/purge")
def debug_purge_states():
    """
    Delete all active states that have garbage or meta values.
    Call this to clean up bad data written by the extractor before the garbage filter was added.
    """
    _GARBAGE_ENTITIES = ("assistant", "him", "he", "she", "they", "it")
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
            "garbage_val": _GARBAGE_VALUE_PATTERN,
        })
        db.commit()
        return {"deleted": result.rowcount}
