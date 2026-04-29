"""
Curiosity injector — decides whether and which question to ask this turn.

Scoring:
  base score  = item.priority
  trigger hit = +0.45 if any topic_trigger word appears in current message
  blocked     = 0 if affect blocked, suppress topic matched, or sensitivity gate fails

Probability gate:
  base 28%, rises to 65% on strong trigger match, scaled by proactive_receptiveness.

LLM timing veto (optional, for emotional_sensitivity >= 0.6):
  A fast Ollama call rates the timing 0–10. If score < 6, the question is suppressed.
  This is what catches the "flowers at a funeral" case that rule-based logic might miss.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import requests

from .schemas import CuriosityItem

log = logging.getLogger(__name__)

_MIN_TURNS_GAP   = 4     # minimum turns between consecutive proactive questions
_BASE_PROB       = 0.28  # base probability of injecting on any eligible turn
_BLOCKED_AFFECTS = {"sad", "frustrated"}


def _sensitivity_gate(sensitivity: float, affect: str) -> bool:
    """Higher-sensitivity items require calmer emotional states."""
    if sensitivity >= 0.7:
        return affect == "calm"
    if sensitivity >= 0.4:
        return affect in {"calm", "disengaged"}
    return affect in {"calm", "confused", "disengaged"}


def select_proactive(
    queue: list[CuriosityItem],
    current_message: str,
    current_affect: str,
    interaction_count: int,
    last_proactive_at: int,
    receptiveness: float,
) -> Optional[CuriosityItem]:
    """Return the best curiosity item to inject this turn, or None."""

    if not queue:
        return None
    # User has signalled they don't want proactive questions
    if receptiveness < 0.15:
        return None
    # Cooldown: don't ask back-to-back
    if interaction_count - last_proactive_at < _MIN_TURNS_GAP:
        return None
    # Never interrupt grief or frustration proactively
    if current_affect in _BLOCKED_AFFECTS:
        return None

    msg_lower  = current_message.lower()
    best: Optional[CuriosityItem] = None
    best_score = -1.0

    for item in queue:
        if item.ask_count >= 3:
            continue
        if current_affect in item.suppress_if_affects:
            continue
        if not _sensitivity_gate(item.emotional_sensitivity, current_affect):
            continue
        # Topic-based suppression — catches wrong emotional context (funeral + flowers)
        if any(t.lower() in msg_lower for t in item.suppress_if_topics):
            continue

        score = item.priority
        # Trigger bonus: this topic is live in the current message (hotel → cupcakes)
        if any(t.lower() in msg_lower for t in item.topic_triggers):
            score += 0.45

        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return None

    # Probability gate — scales with match quality and learned receptiveness
    prob = _BASE_PROB
    if best_score > 0.85:
        prob = 0.65
    elif best_score > 0.55:
        prob = 0.40
    prob *= receptiveness

    return best if random.random() < prob else None


def llm_timing_check(
    question: str,
    current_message: str,
    current_affect: str,
) -> bool:
    """
    LLM veto for high-sensitivity questions.

    A fast single-turn Ollama call rates whether now is a good moment.
    Returns True (allow) on any failure so we don't silently drop questions.
    """
    url   = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

    prompt = (
        f"User just said: \"{current_message}\"\n"
        f"User's current emotional state: {current_affect}\n"
        f"ELARA is considering casually asking: \"{question}\"\n\n"
        f"Rate how appropriate this moment is on a scale of 0 (terrible timing) "
        f"to 10 (perfect timing). Consider emotional context carefully — e.g. if the "
        f"user is talking about grief or loss, a light question about flowers would be "
        f"very inappropriate even if the user likes flowers.\n"
        f"Reply with ONLY a single integer 0–10."
    )

    try:
        resp = requests.post(
            url,
            json={
                "model":   model,
                "messages": [{"role": "user", "content": prompt}],
                "stream":  False,
                "options": {"num_predict": 5},
            },
            timeout=8,
        )
        resp.raise_for_status()
        raw    = resp.json().get("message", {}).get("content", "").strip()
        digits = "".join(c for c in raw if c.isdigit())[:2]
        score  = int(digits) if digits else 5
        log.debug("llm_timing_check score=%d for: %s", score, question[:60])
        return score >= 6
    except Exception as exc:
        log.debug("llm_timing_check failed (%s) — allowing by default", exc)
        return True
