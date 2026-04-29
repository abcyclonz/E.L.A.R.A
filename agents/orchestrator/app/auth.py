"""
auth.py — SQLite-based user authentication for the Orchestrator.

Tables:
  users — id (UUID), email (unique), password_hash, profile fields, created_at

JWT tokens expire in 1 week. Secret is read from JWT_SECRET_KEY env var.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTH_DB_PATH = os.environ.get("AUTH_DB_PATH", "/app/data/users.db")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "elara-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------


def _get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory set so rows behave like dicts."""
    # Ensure the parent directory exists (useful when mounting volumes)
    os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Create the users table if it doesn't exist yet."""
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id                       TEXT PRIMARY KEY,
            email                    TEXT UNIQUE NOT NULL,
            password_hash            TEXT NOT NULL,
            full_name                TEXT,
            age                      TEXT,
            preferred_language       TEXT,
            background               TEXT,
            interests                TEXT,          -- JSON array
            conversation_preferences TEXT,          -- JSON array
            technology_usage         TEXT,
            conversation_goals       TEXT,          -- JSON array
            additional_info          TEXT,
            created_at               TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    print("[Auth] DB initialised at", AUTH_DB_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a DB row to a plain dict with JSON fields deserialized."""
    if row is None:
        return {}
    d = dict(row)
    for key in ("interests", "conversation_preferences", "conversation_goals"):
        raw = d.get(key)
        if raw:
            try:
                d[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d[key] = []
        else:
            d[key] = []
    return d


def _create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def signup(
    email: str,
    password: str,
    full_name: str = "",
    age: str = "",
    preferred_language: str = "",
    background: str = "",
    interests: list = None,
    conversation_preferences: list = None,
    technology_usage: str = "",
    conversation_goals: list = None,
    additional_info: str = "",
) -> dict:
    """
    Create a new user. Returns {user_id, token} on success.
    Raises ValueError if the email is already taken.
    """
    user_id = str(uuid.uuid4())
    pw_hash = pwd_context.hash(password)
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO users
                (id, email, password_hash, full_name, age, preferred_language,
                 background, interests, conversation_preferences, technology_usage,
                 conversation_goals, additional_info, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                email.lower().strip(),
                pw_hash,
                full_name,
                age,
                preferred_language,
                background,
                json.dumps(interests or []),
                json.dumps(conversation_preferences or []),
                technology_usage,
                json.dumps(conversation_goals or []),
                additional_info,
                created_at,
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError(f"Email already registered: {email}")
    finally:
        conn.close()

    token = _create_token(user_id)
    return {"user_id": user_id, "token": token}


def login(email: str, password: str) -> dict:
    """
    Verify credentials. Returns {user_id, row (dict), token} on success.
    Raises ValueError on bad credentials.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()

    if row is None or not pwd_context.verify(password, row["password_hash"]):
        raise ValueError("Invalid email or password")

    user_dict = _row_to_dict(row)
    token = _create_token(user_dict["id"])
    return {"user_id": user_dict["id"], "row": user_dict, "token": token}


def get_user(user_id: str) -> Optional[dict]:
    """Return the user dict for the given UUID, or None if not found."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def verify_token(token: str) -> Optional[str]:
    """Decode and validate a JWT. Returns the user_id (sub claim) or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
