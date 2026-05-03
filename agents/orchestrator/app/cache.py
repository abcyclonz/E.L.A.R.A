"""
Redis-backed session and turn-counter cache for the orchestrator.

Keys:
  session:{speaker_id}  — full Elara SessionState dict (24h TTL)
  turns:{speaker_id}    — summarization turn counter (no TTL)

Falls back to in-memory dicts if Redis is unreachable, so the system
keeps working without Redis at the cost of losing persistence.
"""
from __future__ import annotations
import json
import redis
from app.config import settings

_client: redis.Redis | None = None
_connected: bool | None = None          # None = not yet tried

# In-memory fallbacks (used when Redis is down)
_fb_sessions: dict[str, dict] = {}
_fb_turns: dict[str, int] = {}


def _get() -> redis.Redis | None:
    global _client, _connected
    if _connected is not None:
        return _client if _connected else None
    try:
        r = redis.from_url(settings.redis_url, decode_responses=True,
                           socket_connect_timeout=2)
        r.ping()
        _client = r
        _connected = True
        print("[Cache] Redis connected")
    except Exception as e:
        _connected = False
        print(f"[Cache] Redis unavailable — using in-memory fallback: {e}")
    return _client


# ── SESSION ────────────────────────────────────────────────────────────────

def get_session(speaker_id: str) -> dict | None:
    r = _get()
    if r:
        try:
            raw = r.get(f"session:{speaker_id}")
            return json.loads(raw) if raw else None
        except Exception:
            pass
    return _fb_sessions.get(speaker_id)


def set_session(speaker_id: str, state: dict) -> None:
    r = _get()
    if r:
        try:
            r.set(f"session:{speaker_id}", json.dumps(state), ex=86400)
            return
        except Exception:
            pass
    _fb_sessions[speaker_id] = state


# ── LAST ROUTER ACTION ─────────────────────────────────────────────────────

_fb_last_action: dict[str, str] = {}


def get_last_action(speaker_id: str) -> str | None:
    r = _get()
    if r:
        try:
            return r.get(f"last_action:{speaker_id}")
        except Exception:
            pass
    return _fb_last_action.get(speaker_id)


def set_last_action(speaker_id: str, action: str) -> None:
    r = _get()
    if r:
        try:
            r.set(f"last_action:{speaker_id}", action, ex=86400)
            return
        except Exception:
            pass
    _fb_last_action[speaker_id] = action


# ── TURN COUNTERS ──────────────────────────────────────────────────────────

def incr_turn_count(speaker_id: str) -> int:
    """Increment and return the new count."""
    r = _get()
    if r:
        try:
            return int(r.incr(f"turns:{speaker_id}"))
        except Exception:
            pass
    _fb_turns[speaker_id] = _fb_turns.get(speaker_id, 0) + 1
    return _fb_turns[speaker_id]


def reset_turn_count(speaker_id: str) -> None:
    r = _get()
    if r:
        try:
            r.set(f"turns:{speaker_id}", 0)
            return
        except Exception:
            pass
    _fb_turns[speaker_id] = 0
