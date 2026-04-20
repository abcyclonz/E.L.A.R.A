"""
Assistant Tools MCP Server — reminders and calendar backed by SQLite.

Tools:
  set_reminder(text, when, speaker_id)
  list_reminders(speaker_id)
  complete_reminder(text_match, speaker_id)
  add_calendar_event(title, when, description, speaker_id)
  list_calendar_events(speaker_id, days_ahead)

Runs as SSE server on port 8011.
"""
from mcp.server.fastmcp import FastMCP
import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

mcp = FastMCP("Assistant Tools")

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


# ── REMINDERS ─────────────────────────────────────────────────────────────────

@mcp.tool()
def set_reminder(text: str, when: str, speaker_id: str = "user") -> str:
    """Set a reminder for the user at a specific time or date."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO reminders (speaker_id, text, when_time, created_at) VALUES (?, ?, ?, ?)",
            (speaker_id, text, when, datetime.now().isoformat())
        )
    return f"Reminder set: '{text}' — I'll remind you at {when}."


@mcp.tool()
def list_reminders(speaker_id: str = "user") -> str:
    """List all pending reminders for the user."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT text, when_time FROM reminders WHERE speaker_id=? AND completed=0 ORDER BY created_at DESC LIMIT 10",
            (speaker_id,)
        ).fetchall()

    if not rows:
        return "You have no pending reminders."

    lines = [f"• {r['text']} — {r['when_time']}" for r in rows]
    return "Your pending reminders:\n" + "\n".join(lines)


@mcp.tool()
def complete_reminder(text_match: str, speaker_id: str = "user") -> str:
    """Mark a reminder as done (matches by partial text)."""
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE reminders SET completed=1 WHERE speaker_id=? AND text LIKE ? AND completed=0",
            (speaker_id, f"%{text_match}%")
        )
        count = cursor.rowcount

    if count:
        return f"Done — marked {count} reminder(s) as complete."
    return "No matching pending reminder found."


# ── CALENDAR ──────────────────────────────────────────────────────────────────

@mcp.tool()
def add_calendar_event(title: str, when: str, description: str = "", speaker_id: str = "user") -> str:
    """Add an event to the user's calendar."""
    with _db() as conn:
        conn.execute(
            "INSERT INTO calendar_events (speaker_id, title, when_time, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (speaker_id, title, when, description, datetime.now().isoformat())
        )
    return f"Event added: '{title}' on {when}."


@mcp.tool()
def list_calendar_events(speaker_id: str = "user", days_ahead: int = 7) -> str:
    """List upcoming calendar events for the user."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT title, when_time, description FROM calendar_events WHERE speaker_id=? ORDER BY created_at DESC LIMIT 15",
            (speaker_id,)
        ).fetchall()

    if not rows:
        return "No upcoming calendar events."

    lines = [f"• {r['title']} — {r['when_time']}" + (f" ({r['description']})" if r['description'] else "")
             for r in rows]
    return "Your upcoming events:\n" + "\n".join(lines)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8011"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)
