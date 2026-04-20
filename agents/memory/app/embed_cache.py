"""
Redis-backed embedding cache for the memory agent.

Keys:
  embed:{sha256(text)}  — 768D float list as JSON (24h TTL)

Falls back to no-cache (always calls Ollama) if Redis is unreachable.
The same text always produces the same embedding, so no invalidation needed.
"""
from __future__ import annotations
import hashlib
import json
import redis
from app.config import settings

_client: redis.Redis | None = None
_connected: bool | None = None


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
        print("[EmbedCache] Redis connected")
    except Exception as e:
        _connected = False
        print(f"[EmbedCache] Redis unavailable — embeddings won't be cached: {e}")
    return _client


def _key(text: str) -> str:
    return "embed:" + hashlib.sha256(text.encode()).hexdigest()


def get(text: str) -> list[float] | None:
    r = _get()
    if not r:
        return None
    try:
        raw = r.get(_key(text))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def put(text: str, vector: list[float]) -> None:
    r = _get()
    if not r:
        return
    try:
        r.set(_key(text), json.dumps(vector), ex=86400)  # 24h TTL
    except Exception:
        pass
