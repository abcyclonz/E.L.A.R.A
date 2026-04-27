"""
Layer 1 — NLP Signal Extractor (LLM-based)

Replaces VADER + regex pattern matching with a single structured Ollama call.
The LLM extracts: sentiment, confusion, sadness, humor_receptive, wants_shorter,
wants_simpler, explicit_positive, explicit_negative, personality_hints.

Jaccard repetition is kept as a fast local computation.

Fallback: if Ollama is unavailable or JSON parse fails, reverts to
VADER + minimal keyword scan so the pipeline never hard-fails.
"""

from __future__ import annotations
import json
import logging
import os
import re
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


# ── Signal container ──────────────────────────────────────────────────────────

@dataclass
class NLPSignals:
    sentiment:         float = 0.0    # -1 to 1, overall emotional tone
    repetition:        float = 0.0    # 0 to 1, Jaccard word overlap with prev turn
    confusion:         float = 0.0    # 0 to 1
    sadness:           float = 0.0    # 0 to 1
    humor_receptive:   float = 0.5    # 0 to 1, did user respond well to prior humor
    wants_shorter:     float = 0.0    # 0 to 1, user wants briefer replies
    wants_simpler:     float = 0.0    # 0 to 1, user wants simpler language
    explicit_positive: bool  = False  # "thanks / that helped / great"
    explicit_negative: bool  = False  # expressed frustration with assistant
    personality_hints: Dict[str, Optional[float]] = field(default_factory=dict)
    # ^ keys: "humor", "warmth", "playfulness", "formality"
    # value: desired level 0-1, or None if no signal


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a signal extractor for an elderly care AI. "
    "Analyse the conversation turn and return a JSON object only. "
    "No explanations, no markdown fences, no prose."
)

_USER_TEMPLATE = """\
Previous assistant reply (empty if first turn):
{prev_reply}

User message:
{message}

Return exactly this JSON (all keys required, floats 0.0-1.0 unless noted):
{{
  "sentiment": <-1.0 to 1.0 overall emotional tone of user message>,
  "confusion": <0-1 how confused or frustrated with the communication is the user>,
  "sadness": <0-1 how sad, lonely, or distressed is the user>,
  "humor_receptive": <0-1 did user react positively to humor in prior reply>,
  "wants_shorter": <0-1 user is asking for or implying briefer replies>,
  "wants_simpler": <0-1 user wants simpler language or finds things too complex>,
  "explicit_positive": <true if user said thanks/great/that helped/perfect/wonderful>,
  "explicit_negative": <true if user expressed frustration or dissatisfaction with assistant>,
  "personality_hints": {{
    "humor": <desired humor level 0-1 or null if no signal>,
    "warmth": <desired warmth level 0-1 or null if no signal>,
    "playfulness": <desired playfulness 0-1 or null if no signal>,
    "formality": <desired formality 0-1 or null if no signal>
  }}
}}"""


# ── Public API ────────────────────────────────────────────────────────────────

def extract_signals(turns: list) -> NLPSignals:
    """
    Primary entry point. Accepts a list of Turn objects (role, text).
    Returns NLPSignals populated via Ollama, falling back to VADER+keywords.
    """
    user_texts  = [t.text for t in turns if t.role == "user"]
    agent_texts = [t.text for t in turns if t.role == "agent"]

    message    = user_texts[-1]  if user_texts  else ""
    prev_reply = agent_texts[-1] if agent_texts else ""

    repetition = _jaccard_repetition(user_texts)

    # Skip LLM for very short inputs (single words, greetings) — saves ~150ms
    if len(message.split()) <= 2:
        return NLPSignals(repetition=repetition)

    return _llm_extract(message, prev_reply, repetition)


# ── LLM extraction ────────────────────────────────────────────────────────────

def _llm_extract(message: str, prev_reply: str, repetition: float) -> NLPSignals:
    prompt = _USER_TEMPLATE.format(
        message    = message[:500],
        prev_reply = prev_reply[:250] if prev_reply else "(none)",
    )
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "options": {"num_predict": 200, "temperature": 0.0},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=10)
        resp.raise_for_status()
        raw  = resp.json().get("message", {}).get("content", "")
        data = _parse_json(raw)
        if data:
            return NLPSignals(
                sentiment         = _f(data.get("sentiment", 0.0), -1.0, 1.0),
                repetition        = repetition,
                confusion         = _f(data.get("confusion", 0.0)),
                sadness           = _f(data.get("sadness", 0.0)),
                humor_receptive   = _f(data.get("humor_receptive", 0.5)),
                wants_shorter     = _f(data.get("wants_shorter", 0.0)),
                wants_simpler     = _f(data.get("wants_simpler", 0.0)),
                explicit_positive = bool(data.get("explicit_positive", False)),
                explicit_negative = bool(data.get("explicit_negative", False)),
                personality_hints = _parse_hints(data.get("personality_hints", {})),
            )
    except Exception as exc:
        log.warning("[nlp_layer] LLM extraction failed (%s) — using fallback", exc)

    return _fallback_extract(message, repetition)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _parse_hints(raw: dict) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in ("humor", "warmth", "playfulness", "formality"):
        val = raw.get(key)
        out[key] = _f(val) if val is not None else None
    return out


# ── Fallback: VADER + keyword scan ────────────────────────────────────────────

def _fallback_extract(message: str, repetition: float) -> NLPSignals:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sentiment = SentimentIntensityAnalyzer().polarity_scores(message)["compound"]
    except Exception:
        sentiment = 0.0

    confusion = _kw_score(message, _CONFUSION_COMPILED)
    sadness   = _kw_score(message, _SADNESS_COMPILED)
    explicit_pos = bool(re.search(
        r"\b(thank you|thanks|that helped|much better|got it|perfect|brilliant|wonderful)\b",
        message, re.I,
    ))
    explicit_neg = bool(re.search(
        r"\b(stop (being|doing)|you'?re? (useless|annoying|wrong)|that'?s? (wrong|bad|terrible))\b",
        message, re.I,
    ))
    return NLPSignals(
        sentiment=sentiment, repetition=repetition,
        confusion=confusion, sadness=sadness,
        explicit_positive=explicit_pos, explicit_negative=explicit_neg,
    )


_CONFUSION_KW = [
    r"\bdon'?t understand\b", r"\bconfus(ed|ing)\b", r"\bi'?m lost\b",
    r"\btoo complicated\b", r"\bmakes no sense\b", r"\bwhat do you mean\b",
    r"\bdoesn'?t make sense\b", r"\bnot following\b", r"\byou'?ve lost me\b",
    r"\bsay that again\b", r"\bshorter (please|reply|answer)?\b",
    r"\bdon'?t know what you'?re? saying\b",
]
_SADNESS_KW = [
    r"\blon(ely|eliness)\b", r"\ball alone\b", r"\bnobody (calls?|visits?)\b",
    r"\bfeel(ing)? (sad|down|low|blue|empty)\b", r"\bgriev(e|ing)\b",
    r"\bsince (he|she) (passed|died)\b", r"\bno(body|one) to talk to\b",
    r"\bhaven'?t heard from\b", r"\bfeel(ing)? so alone\b",
]
_CONFUSION_COMPILED = [(re.compile(p, re.I), 0.65) for p in _CONFUSION_KW]
_SADNESS_COMPILED   = [(re.compile(p, re.I), 0.65) for p in _SADNESS_KW]


def _kw_score(text: str, patterns: list) -> float:
    return min(1.0, sum(w for p, w in patterns if p.search(text)))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _jaccard_repetition(texts: List[str]) -> float:
    if len(texts) < 2:
        return 0.0
    a = set(texts[-2].lower().split())
    b = set(texts[-1].lower().split())
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _f(v, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.0
