"""
main.py — Perceptive Learning Agent

Passively polls the perception timeline (SQLite written by monitor.py)
for new snapshots, cross-references each with chat memory, and stores
grounded inferences back into the memory agent as EVENT memories.

This agent is entirely passive — it never touches the live conversation
pipeline and never interferes with Elara's own learning system.
"""
import asyncio
import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import reasoner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("perception_learner")

PERCEPTION_DB_PATH = Path(os.environ.get("PERCEPTION_DB_PATH", "/data/perception/timeline.db"))
POLL_INTERVAL_S    = int(os.environ.get("POLL_INTERVAL_S", "600"))
STATE_DB_PATH      = Path("/data/learner_state.db")


# ── State tracking (which snapshots have been processed) ──────────────────────

def _init_state_db():
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(STATE_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed (
                snapshot_id  INTEGER PRIMARY KEY,
                processed_at TEXT NOT NULL DEFAULT (datetime('now')),
                inference    TEXT
            )
        """)


def _last_processed_id() -> int:
    with sqlite3.connect(str(STATE_DB_PATH)) as conn:
        row = conn.execute("SELECT MAX(snapshot_id) FROM processed").fetchone()
    return row[0] or 0


def _mark_processed(snapshot_id: int, inference: str | None):
    with sqlite3.connect(str(STATE_DB_PATH)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed (snapshot_id, inference) VALUES (?, ?)",
            (snapshot_id, inference),
        )


def _processed_count() -> int:
    with sqlite3.connect(str(STATE_DB_PATH)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM processed").fetchone()
    return row[0] or 0


def _inference_count() -> int:
    with sqlite3.connect(str(STATE_DB_PATH)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM processed WHERE inference IS NOT NULL"
        ).fetchone()
    return row[0] or 0


# ── Timeline reading ───────────────────────────────────────────────────────────

def _fetch_new_snapshots(since_id: int) -> list[dict]:
    if not PERCEPTION_DB_PATH.exists():
        return []
    try:
        with sqlite3.connect(str(PERCEPTION_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE id > ? ORDER BY id ASC",
                (since_id,),
            ).fetchall()
        result = []
        for r in rows:
            result.append({
                "id":               r["id"],
                "ts":               r["ts"],
                "user":             r["user"],
                "emotion":          r["emotion"],
                "confidence":       r["confidence"],
                "scene":            r["scene"],
                "subjects":         json.loads(r["subjects"]),
                "emotion_affected": bool(r["affected"]),
                "reason":           r["reason"],
                "summary":          r["summary"],
            })
        return result
    except Exception as e:
        log.error("failed to read timeline.db: %s", e)
        return []


def _total_snapshot_count() -> int:
    if not PERCEPTION_DB_PATH.exists():
        return 0
    try:
        with sqlite3.connect(str(PERCEPTION_DB_PATH)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()
        return row[0] or 0
    except Exception:
        return 0


# ── Poll logic (shared by background loop and /debug/trigger) ─────────────────

async def _run_one_poll() -> dict:
    last_id   = _last_processed_id()
    snapshots = _fetch_new_snapshots(since_id=last_id)

    results = []
    for snap in snapshots:
        log.info("--- snapshot #%d  user=%-12s  emotion=%s",
                 snap["id"], snap["user"], snap["emotion"])
        try:
            inference = await asyncio.to_thread(reasoner.process_snapshot, snap)
        except Exception as e:
            log.exception("reasoner crashed on snapshot #%d: %s", snap["id"], e)
            inference = None

        _mark_processed(snap["id"], inference)
        results.append({
            "snapshot_id": snap["id"],
            "user":        snap["user"],
            "emotion":     snap["emotion"],
            "inference":   inference,
            "stored":      inference is not None,
        })

    return {"processed": len(results), "results": results}


# ── Poll loop ──────────────────────────────────────────────────────────────────

async def _poll_loop():
    log.info("perceptive learner started  poll_interval=%ds", POLL_INTERVAL_S)
    log.info("perception db: %s", PERCEPTION_DB_PATH)

    while True:
        try:
            await _run_one_poll()
        except Exception as e:
            log.exception("poll loop error: %s", e)
        await asyncio.sleep(POLL_INTERVAL_S)


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_state_db()
    task = asyncio.create_task(_poll_loop())
    yield
    task.cancel()


app = FastAPI(
    title="Perception Learner",
    description="Cross-modal inference: perception timeline × chat memory → EVENT memories",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status":            "ok",
        "perception_db":     str(PERCEPTION_DB_PATH),
        "perception_db_ok":  PERCEPTION_DB_PATH.exists(),
        "last_processed_id": _last_processed_id(),
    }


@app.get("/status")
def status():
    last_id  = _last_processed_id()
    pending  = len(_fetch_new_snapshots(since_id=last_id))
    return {
        "total_perception_snapshots": _total_snapshot_count(),
        "processed_count":            _processed_count(),
        "inferences_generated":       _inference_count(),
        "pending_snapshots":          pending,
        "poll_interval_s":            POLL_INTERVAL_S,
    }


# ── Debug endpoints (used by test suite) ──────────────────────────────────────

@app.post("/debug/inject_snapshot")
def inject_snapshot(snap: dict):
    """
    Write a fake perception snapshot directly into timeline.db.
    Used by the test suite to simulate monitor.py output without needing a Pi.
    """
    PERCEPTION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(PERCEPTION_DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                user       TEXT    NOT NULL,
                emotion    TEXT    NOT NULL,
                confidence REAL    NOT NULL,
                scene      TEXT    NOT NULL,
                subjects   TEXT    NOT NULL,
                affected   INTEGER NOT NULL,
                reason     TEXT    NOT NULL,
                summary    TEXT    NOT NULL,
                thumbnail  BLOB
            )
        """)
        cur = conn.execute(
            "INSERT INTO snapshots (ts, user, emotion, confidence, scene, subjects, "
            "affected, reason, summary) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                snap.get("ts", "2024-01-01T00:00:00+00:00"),
                snap["user"],
                snap.get("emotion", "neutral"),
                snap.get("confidence", 0.8),
                snap.get("scene", ""),
                json.dumps(snap.get("subjects", [])),
                int(snap.get("emotion_affected", False)),
                snap.get("reason", ""),
                snap.get("summary", ""),
            ),
        )
    return {"snapshot_id": cur.lastrowid}


@app.post("/debug/trigger")
async def trigger_poll():
    """
    Force one immediate poll cycle and return per-snapshot results.
    Used by the test suite to avoid waiting 10 minutes.
    """
    result = await _run_one_poll()
    return result
