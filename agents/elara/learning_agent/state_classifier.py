"""
Layer 2 — State Classifier

Affect states: frustrated | confused | sad | calm | disengaged

Input: NLPSignals (from nlp_layer) containing:
  sentiment  : LLM-extracted compound score    (-1 to +1)
  repetition : Jaccard word overlap             (0 to 1)
  confusion  : LLM-extracted confusion score   (0 to 1)
  sadness    : LLM-extracted sadness score     (0 to 1)

Decision priority (checked top to bottom, first match wins):
  frustrated   : strong negative + repetition
  confused     : confusion signal or repetition
  sad          : sadness signal (no confusion)
  disengaged   : very short input with neutral signal
  calm         : none of the above

── Escalation Smoother ───────────────────────────────────────────────────────
After raw affect is determined, apply_escalation_rules() inspects the rolling
affect_window and may downgrade the raw affect if evidence is insufficient.

Rules (in order, first match wins):
  R3  all_calm_history      — frustrated after all-calm window → confused
  R1  insufficient_streak   — frustrated needs ≥2 consecutive non-calm turns
  R4  disengaged_calm_history — short message after all-calm → calm (not disengaged)

── Feature encoding ──────────────────────────────────────────────────────────
encode_context_features(affect, personality) → 14D vector
  [ One-Hot Affect (5D) | PersonalityVector (9D) ]

N_CONTEXTS = 5 (one per affect state) — used only for diagnostics.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
SENTIMENT_THRESHOLD   = -0.2
REPETITION_THRESHOLD  = 0.55
CONFUSION_THRESHOLD   = 0.5
SADNESS_THRESHOLD     = 0.5
DISENGAGED_WORD_COUNT = 3

WINDOW_SIZE        = 5
MIN_NONCALM_STREAK = 2

# ── Encoding maps ─────────────────────────────────────────────────────────────
AFFECT_MAP  = {"frustrated": 0, "confused": 1, "sad": 2, "calm": 3, "disengaged": 4}

# Single source of truth imported by bandit.py and storage.py
N_CONTEXTS = 5    # one per affect state (context_id = affect index)
N_ACTIONS  = 19   # 0=DO_NOTHING, 1-18=INC/DEC per personality dim

_VALID_AFFECTS = frozenset(AFFECT_MAP.keys())


# ── Public API ────────────────────────────────────────────────────────────────

def classify_state(
    signals_or_sentiment,
    repetition:     float = 0.0,
    confusion:      float = 0.0,
    sadness:        float = 0.0,
    last_user_text: str   = "",
    affect_window:  Optional[List[str]] = None,
) -> Tuple[str, float, List[str], Optional[str]]:
    """
    Accepts either:
      classify_state(NLPSignals, last_user_text=..., affect_window=...)
    or the old positional form:
      classify_state(sentiment, repetition, confusion, sadness, last_user_text, affect_window)
    for backward compatibility.

    Returns (affect, confidence, signals_used, escalation_rule_applied).
    """
    from nlp_layer import NLPSignals
    if isinstance(signals_or_sentiment, NLPSignals):
        sig = signals_or_sentiment
        sentiment  = sig.sentiment
        repetition = sig.repetition
        confusion  = sig.confusion
        sadness    = sig.sadness
    else:
        sentiment = float(signals_or_sentiment)

    # Validate and sanitise affect_window
    clean_window: Optional[List[str]] = None
    if affect_window is not None:
        clean_window = []
        for entry in affect_window:
            if entry in _VALID_AFFECTS:
                clean_window.append(entry)
            else:
                log.warning(
                    "[state_classifier] Unknown affect '%s' in affect_window — ignored.",
                    entry,
                )

    negative    = sentiment  < SENTIMENT_THRESHOLD
    repetitive  = repetition > REPETITION_THRESHOLD
    confused_kw = confusion  > CONFUSION_THRESHOLD
    sad_kw      = sadness    > SADNESS_THRESHOLD

    signals_used: List[str] = []
    if negative:    signals_used.append("sentiment")
    if repetitive:  signals_used.append("repetition")
    if confused_kw: signals_used.append("confusion_keywords")
    if sad_kw:      signals_used.append("sadness_keywords")

    # ── 1. frustrated ────────────────────────────────────────────
    strong_neg   = negative or confusion > 0.65
    high_conf_kw = confusion > 0.8
    if (strong_neg and repetitive) or (high_conf_kw and (negative or repetitive or confusion > 0.9)):
        affect = "frustrated"
        conf = _blend(
            abs(sentiment - SENTIMENT_THRESHOLD) / 1.0 if negative else confusion,
            max(
                (repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD),
                (confusion  - 0.65) / 0.35,
            ),
        )

    # ── 2. confused ──────────────────────────────────────────────
    elif confused_kw or (repetitive and not sad_kw):
        affect = "confused"
        scores = []
        if confused_kw:
            scores.append((confusion - CONFUSION_THRESHOLD) / (1.0 - CONFUSION_THRESHOLD))
        if repetitive:
            scores.append((repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD))
        if negative and not sad_kw:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 3. sad ───────────────────────────────────────────────────
    elif sad_kw or (negative and not confused_kw and not repetitive):
        affect = "sad"
        scores = []
        if sad_kw:
            scores.append((sadness - SADNESS_THRESHOLD) / (1.0 - SADNESS_THRESHOLD))
        if negative:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 4. disengaged ────────────────────────────────────────────
    elif (
        last_user_text
        and len(last_user_text.split()) <= DISENGAGED_WORD_COUNT
        and not negative
        and not confused_kw
        and not sad_kw
        and not repetitive
    ):
        affect = "disengaged"
        signals_used.append("short_message")
        conf = 0.6

    # ── 5. calm ──────────────────────────────────────────────────
    else:
        affect = "calm"
        conf = _clamp(
            0.5
            + (sentiment  - SENTIMENT_THRESHOLD)  / 2.0
            + (REPETITION_THRESHOLD - repetition) / 2.0
            + (CONFUSION_THRESHOLD  - confusion)  / 2.0
            + (SADNESS_THRESHOLD    - sadness)    / 2.0
        )
        signals_used = []

    # ── Escalation smoother ──────────────────────────────────────
    affect, conf, escalation_rule = apply_escalation_rules(
        raw_affect=affect,
        raw_conf=conf,
        affect_window=clean_window,
        all_signals_fired=(negative and repetitive and confused_kw),
    )

    if escalation_rule:
        signals_used.append(f"escalation:{escalation_rule}")

    return affect, round(conf, 4), signals_used, escalation_rule


def apply_escalation_rules(
    raw_affect: str,
    raw_conf: float,
    affect_window: Optional[List[str]],
    all_signals_fired: bool,
) -> Tuple[str, float, Optional[str]]:

    # R4: disengaged after all-calm history → brevity, not disengagement
    if raw_affect == "disengaged":
        if not affect_window or all(a == "calm" for a in affect_window[-WINDOW_SIZE:]):
            return "calm", raw_conf * 0.9, "R4_calm_history_not_disengaged"
        return raw_affect, raw_conf, None

    # R5: low-confidence sadness immediately after a calm turn → don't over-escalate.
    # Prevents a mild NLP sadness score from persisting when the conversation has
    # naturally moved on (e.g. user mentions retirement → small model reads as sad).
    if raw_affect == "sad" and raw_conf < 0.65:
        if affect_window and affect_window[-1] == "calm":
            return "calm", raw_conf * 0.85, "R5_low_conf_sad_after_calm"

    if not affect_window:
        return raw_affect, raw_conf, None

    window = affect_window[-WINDOW_SIZE:]

    if raw_affect != "frustrated":
        return raw_affect, raw_conf, None

    non_calm_count = sum(1 for a in window if a != "calm")

    # R3: entirely calm history → frustrated not allowed
    if non_calm_count == 0:
        return "confused", raw_conf * 0.8, "R3_all_calm_history"

    # R1: insufficient trailing streak
    streak = _trailing_noncalm_streak(window)
    if streak < MIN_NONCALM_STREAK and not all_signals_fired:
        return "confused", raw_conf * 0.75, "R1_insufficient_streak"

    return raw_affect, raw_conf, None


# ── Context encoder ───────────────────────────────────────────────────────────

def encode_context_id(affect: str, *_args) -> int:
    """Returns affect index 0-4. Extra args ignored (backward compat)."""
    return AFFECT_MAP.get(affect, AFFECT_MAP["calm"])


def encode_context_features(affect: str, personality_or_clarity, pace: str = "normal") -> np.ndarray:
    """
    Encodes state into a 14D feature vector for LinUCB.

    Accepts either:
      encode_context_features(affect, PersonalityVector)   — new 14D form
      encode_context_features(affect, clarity_level, pace) — old 7D form (upgraded to 14D)

    Vector structure: [ One-Hot Affect (5D) | PersonalityVector (9D) ]
    """
    # One-hot affect (5D)
    affect_vec = np.zeros(len(AFFECT_MAP), dtype=float)
    affect_vec[AFFECT_MAP.get(affect, AFFECT_MAP["calm"])] = 1.0

    # Personality dims (9D) — duck-type check avoids import-path mismatch
    if hasattr(personality_or_clarity, "to_array"):
        pers_vec = personality_or_clarity.to_array()
    else:
        # Legacy path: reconstruct a PersonalityVector from old clarity/pace
        clarity_level = int(personality_or_clarity)
        clarity_val   = {1: 0.9, 2: 0.65, 3: 0.4}.get(clarity_level, 0.65)
        pace_val      = {"slow": 0.25, "normal": 0.5, "fast": 0.75}.get(pace, 0.5)
        pers_vec      = PV(clarity=clarity_val, pace=pace_val).to_array()

    return np.concatenate([affect_vec, pers_vec])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _trailing_noncalm_streak(window: List[str]) -> int:
    streak = 0
    for entry in reversed(window):
        if entry != "calm":
            streak += 1
        else:
            break
    return streak


def _blend(a: float, b: float) -> float:
    return _clamp((a + b) / 2.0)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
