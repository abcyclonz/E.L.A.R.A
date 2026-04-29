"""
Assistant tool server — FastAPI REST interface for reminders and calendar.

Reminders: SQLite (persistent across restarts via Docker volume).
Calendar:  Google Calendar API when token.json is present;
           falls back to SQLite if not authenticated.

POST /call/set_reminder        {"text": "...", "when": "...", "speaker_id": "user"}
POST /call/list_reminders      {"speaker_id": "user"}
POST /call/complete_reminder   {"text_match": "...", "speaker_id": "user"}
POST /call/add_calendar_event  {"title": "...", "when": "...", "description": "", "speaker_id": "user"}
POST /call/list_calendar_events {"speaker_id": "user"}
GET  /health
"""

from __future__ import annotations

import datetime
import logging
import os
from contextlib import contextmanager

import dateparser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3

log = logging.getLogger(__name__)

app = FastAPI(title="Assistant Tools")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB_PATH = os.getenv("DB_PATH", "/data/assistant.db")
GOOGLE_AUTH_DIR = os.getenv("GOOGLE_AUTH_DIR", "/app/google_auth")
TOKEN_FILE = os.path.join(GOOGLE_AUTH_DIR, "token.json")
CREDENTIALS_FILE = os.path.join(GOOGLE_AUTH_DIR, "credentials.json")
CALENDAR_TZ = os.getenv("CALENDAR_TIMEZONE", "Asia/Kolkata")
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


# ── SQLite setup ──────────────────────────────────────────────────────────────

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


# ── Google Calendar client ────────────────────────────────────────────────────

_gcal_service = None


def _get_gcal_service():
    """Return (service, error_string). Caches the service after first init."""
    global _gcal_service
    if _gcal_service:
        return _gcal_service, None

    if not os.path.exists(TOKEN_FILE):
        return None, (
            "Google Calendar not connected. "
            "Run auth_setup.py once to generate google_auth/token.json."
        )

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials.from_authorized_user_file(TOKEN_FILE, CALENDAR_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())

        _gcal_service = build("calendar", "v3", credentials=creds)
        log.info("Google Calendar service initialised.")
        return _gcal_service, None

    except Exception as exc:
        return None, f"Google Calendar auth error: {exc}"


_FUZZY_TIMES = {
    "morning": "9:00 AM",
    "afternoon": "2:00 PM",
    "evening": "6:00 PM",
    "night": "8:00 PM",
    "noon": "12:00 PM",
    "midnight": "11:59 PM",
}


def _parse_when(when_str: str) -> datetime.datetime | None:
    """Parse a natural-language date/time string into a datetime object."""
    # Replace fuzzy time words so dateparser gets a concrete time
    lower = when_str.lower().strip()
    for word, replacement in _FUZZY_TIMES.items():
        if word in lower:
            lower = lower.replace(word, replacement)
            break
    now = datetime.datetime.now()
    parsed = dateparser.parse(
        lower,
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": CALENDAR_TZ,
            "RETURN_AS_TIMEZONE_AWARE": False,
            "RELATIVE_BASE": now,          # anchors "today", "tomorrow", "next week"
        },
    )
    # If dateparser returns a past time on the same day, bump it to today at that time
    if parsed and parsed < now and (now - parsed).days == 0:
        parsed = parsed.replace(
            year=now.year, month=now.month, day=now.day
        )
    return parsed


# ── Request models ────────────────────────────────────────────────────────────

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


# ── Reminder endpoints (SQLite) ───────────────────────────────────────────────

@app.post("/call/set_reminder")
def set_reminder(req: ReminderSetReq):
    with _db() as conn:
        conn.execute(
            "INSERT INTO reminders (speaker_id, text, when_time, created_at) VALUES (?, ?, ?, ?)",
            (req.speaker_id, req.text, req.when, datetime.datetime.now().isoformat())
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


# ── Calendar endpoints (Google Calendar with SQLite fallback) ─────────────────

@app.post("/call/add_calendar_event")
def add_calendar_event(req: CalendarAddReq):
    service, gcal_err = _get_gcal_service()

    # Parse the natural-language "when" into a datetime
    start_dt = _parse_when(req.when)
    if not start_dt:
        return {"result": f"Sorry, I couldn't understand the date '{req.when}'. Please say something like 'tomorrow at 3pm' or 'next Monday at 10am'."}

    end_dt = start_dt + datetime.timedelta(hours=1)

    if service:
        try:
            from googleapiclient.errors import HttpError
            event_body = {
                "summary": req.title,
                "description": req.description,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": CALENDAR_TZ},
                "end":   {"dateTime": end_dt.isoformat(),   "timeZone": CALENDAR_TZ},
            }
            service.events().insert(calendarId="primary", body=event_body).execute()
            friendly = start_dt.strftime("%B %d at %I:%M %p")
            return {"result": f"Added '{req.title}' to your Google Calendar for {friendly}."}
        except Exception as exc:
            log.error("Google Calendar insert failed: %s", exc)
            gcal_err = str(exc)

    # Fallback: persist in SQLite so the event isn't lost
    with _db() as conn:
        conn.execute(
            "INSERT INTO calendar_events (speaker_id, title, when_time, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (req.speaker_id, req.title, req.when, req.description, datetime.datetime.now().isoformat())
        )
    return {"result": f"Event '{req.title}' saved locally for {req.when}. (Google Calendar unavailable: {gcal_err})"}


@app.post("/call/list_calendar_events")
def list_calendar_events(req: CalendarListReq):
    service, gcal_err = _get_gcal_service()

    if service:
        try:
            now_iso = datetime.datetime.utcnow().isoformat() + "Z"
            result = service.events().list(
                calendarId="primary",
                timeMin=now_iso,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            items = result.get("items", [])
            if not items:
                return {"result": "You have no upcoming events in Google Calendar."}
            lines = []
            for ev in items:
                start = ev["start"].get("dateTime", ev["start"].get("date", ""))
                try:
                    dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                    start = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    pass
                lines.append(f"• {ev.get('summary', '(no title)')} — {start}")
            return {"result": "Your upcoming events:\n" + "\n".join(lines)}
        except Exception as exc:
            log.error("Google Calendar list failed: %s", exc)
            gcal_err = str(exc)

    # Fallback: read from SQLite
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
    note = f" (Google Calendar unavailable: {gcal_err})" if gcal_err else ""
    return {"result": "Your upcoming events" + note + ":\n" + "\n".join(lines)}


@app.get("/health")
def health():
    _, gcal_err = _get_gcal_service()
    return {
        "status": "ok",
        "service": "assistant_tool",
        "google_calendar": "connected" if not gcal_err else f"unavailable: {gcal_err}",
    }
