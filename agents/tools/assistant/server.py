"""
Assistant tool server — FastAPI REST interface for reminders and calendar (SQLite).

POST /call/set_reminder        {"text": "...", "when": "...", "speaker_id": "user"}
POST /call/list_reminders      {"speaker_id": "user"}
POST /call/complete_reminder   {"text_match": "...", "speaker_id": "user"}
POST /call/add_calendar_event  {"title": "...", "when": "...", "description": "", "speaker_id": "user"}
POST /call/list_calendar_events {"speaker_id": "user"}
GET  /health
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime

app = FastAPI(title="Assistant Tools")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = os.getenv("DB_PATH", "/data/assistant.db")


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id  TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                when_time   TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                completed   INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS calendar_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id  TEXT    NOT NULL,
                title       TEXT    NOT NULL,
                when_time   TEXT    NOT NULL,
                description TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL
            );
        """)


_init_db()


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Request models ─────────────────────────────────────────────────────────

class ReminderSetReq(BaseModel):
    text: str
    when: str
    speaker_id: str = "user"

class ReminderListReq(BaseModel):
    speaker_id: str = "user"

class ReminderCompleteReq(BaseModel):
    text_match: str
    speaker_id: str = "user"

class CalendarAddReq(BaseModel):
    title: str
    when: str
    description: str = ""
    speaker_id: str = "user"

class CalendarListReq(BaseModel):
    speaker_id: str = "user"


# ── Reminder endpoints ─────────────────────────────────────────────────────

@app.post("/call/set_reminder")
def set_reminder(req: ReminderSetReq):
    with _db() as conn:
        conn.execute(
            "INSERT INTO reminders (speaker_id, text, when_time, created_at) VALUES (?, ?, ?, ?)",
            (req.speaker_id, req.text, req.when, datetime.now().isoformat())
        )
    return {"result": f"Reminder set: '{req.text}' at {req.when}."}


@app.post("/call/list_reminders")
def list_reminders(req: ReminderListReq):
    with _db() as conn:
        rows = conn.execute(
            "SELECT text, when_time FROM reminders WHERE speaker_id=? AND completed=0 ORDER BY created_at DESC LIMIT 10",
            (req.speaker_id,)
        ).fetchall()
    if not rows:
        return {"result": "You have no pending reminders."}
    lines = [f"• {r['text']} — {r['when_time']}" for r in rows]
    return {"result": "Your pending reminders:\n" + "\n".join(lines)}


@app.post("/call/complete_reminder")
def complete_reminder(req: ReminderCompleteReq):
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE reminders SET completed=1 WHERE speaker_id=? AND text LIKE ? AND completed=0",
            (req.speaker_id, f"%{req.text_match}%")
        )
        count = cursor.rowcount
    if count:
        return {"result": f"Done — marked {count} reminder(s) as complete."}
    return {"result": "No matching pending reminder found."}


# ── Calendar endpoints ─────────────────────────────────────────────────────

@app.post("/call/add_calendar_event")
def add_calendar_event(req: CalendarAddReq):
    with _db() as conn:
        conn.execute(
            "INSERT INTO calendar_events (speaker_id, title, when_time, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (req.speaker_id, req.title, req.when, req.description, datetime.now().isoformat())
        )
    return {"result": f"Event added: '{req.title}' on {req.when}."}


@app.post("/call/list_calendar_events")
def list_calendar_events(req: CalendarListReq):
    with _db() as conn:
        rows = conn.execute(
            "SELECT title, when_time, description FROM calendar_events WHERE speaker_id=? ORDER BY created_at DESC LIMIT 15",
            (req.speaker_id,)
        ).fetchall()
    if not rows:
        return {"result": "No upcoming calendar events."}
    lines = [
        f"• {r['title']} — {r['when_time']}" + (f" ({r['description']})" if r["description"] else "")
        for r in rows
    ]
    return {"result": "Your upcoming events:\n" + "\n".join(lines)}


@app.get("/health")
def health():
    return {"status": "ok", "service": "assistant_tool"}
