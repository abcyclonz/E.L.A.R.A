"""
Personality Applier

Maps action_id → PersonalityVector delta.
Returns (new_personality, changes_dict, reason_str).

Actions (19):
  0  DO_NOTHING
  1  INCREASE_WARMTH        2  DECREASE_WARMTH
  3  INCREASE_HUMOR         4  DECREASE_HUMOR
  5  INCREASE_PLAYFULNESS   6  DECREASE_PLAYFULNESS
  7  INCREASE_FORMALITY     8  DECREASE_FORMALITY
  9  INCREASE_VERBOSITY    10  DECREASE_VERBOSITY
  11 INCREASE_PACE         12  DECREASE_PACE
  13 INCREASE_CLARITY      14  DECREASE_CLARITY
  15 INCREASE_PATIENCE     16  DECREASE_PATIENCE
  17 INCREASE_ASSERTIVENESS 18 DECREASE_ASSERTIVENESS
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

from .personality import PersonalityVector, DIMS, AFFECT_DEFAULTS

STEP = 0.10   # size of one bandit step

# Action IDs
DO_NOTHING = 0
ACTION_NAMES = {
    0:  "DO_NOTHING",
    1:  "INCREASE_WARMTH",       2:  "DECREASE_WARMTH",
    3:  "INCREASE_HUMOR",        4:  "DECREASE_HUMOR",
    5:  "INCREASE_PLAYFULNESS",  6:  "DECREASE_PLAYFULNESS",
    7:  "INCREASE_FORMALITY",    8:  "DECREASE_FORMALITY",
    9:  "INCREASE_VERBOSITY",   10:  "DECREASE_VERBOSITY",
    11: "INCREASE_PACE",        12:  "DECREASE_PACE",
    13: "INCREASE_CLARITY",     14:  "DECREASE_CLARITY",
    15: "INCREASE_PATIENCE",    16:  "DECREASE_PATIENCE",
    17: "INCREASE_ASSERTIVENESS", 18: "DECREASE_ASSERTIVENESS",
}

# Maps action_id → (dim_name, delta_sign)
# +1 = increase, -1 = decrease
_ACTION_MAP: Dict[int, Tuple[str, int]] = {
    1:  ("warmth",        +1),  2:  ("warmth",        -1),
    3:  ("humor",         +1),  4:  ("humor",         -1),
    5:  ("playfulness",   +1),  6:  ("playfulness",   -1),
    7:  ("formality",     +1),  8:  ("formality",     -1),
    9:  ("verbosity",     +1), 10:  ("verbosity",     -1),
    11: ("pace",          +1), 12:  ("pace",          -1),
    13: ("clarity",       +1), 14:  ("clarity",       -1),
    15: ("patience",      +1), 16:  ("patience",      -1),
    17: ("assertiveness", +1), 18:  ("assertiveness", -1),
}

_AFFECT_REASON = {
    "frustrated": "affect_frustrated_detected",
    "confused":   "affect_confused_detected",
    "sad":        "affect_sad_detected",
    "calm":       "affect_calm_no_change",
    "disengaged": "affect_disengaged_detected",
}


def apply_action(
    action_id: int,
    personality: PersonalityVector,
    affect: str,
) -> Tuple[PersonalityVector, Dict[str, Any], str]:
    """
    Returns (updated_personality, changes_dict, reason).
    changes_dict maps dim_name → new_value; empty when nothing changes.
    """
    changes: Dict[str, Any] = {}
    d = personality.model_dump()

    if action_id == DO_NOTHING:
        # Calm recovery: nudge one param at a time toward affect-appropriate defaults
        if affect == "calm":
            default = AFFECT_DEFAULTS["calm"]
            for dim in DIMS:
                curr    = d[dim]
                target  = getattr(default, dim)
                if abs(curr - target) > 0.01:
                    direction = 1 if target > curr else -1
                    new_val   = round(max(0.0, min(1.0, curr + direction * STEP * 0.5)), 3)
                    if new_val != curr:
                        d[dim]       = new_val
                        changes[dim] = new_val
                        break   # one parameter per turn

    elif action_id in _ACTION_MAP:
        dim, sign = _ACTION_MAP[action_id]
        curr    = d[dim]
        new_val = round(max(0.0, min(1.0, curr + sign * STEP)), 3)
        if new_val != curr:
            d[dim]       = new_val
            changes[dim] = new_val

    new_personality = PersonalityVector(**d)

    if not changes:
        reason = "no_change_needed_or_already_at_limit"
    elif affect == "calm" and action_id == DO_NOTHING:
        reason = "calm_recovery_step"
    else:
        reason = _AFFECT_REASON.get(affect, ACTION_NAMES.get(action_id, str(action_id)))

    return new_personality, changes, reason


# ── Legacy shim ───────────────────────────────────────────────────────────────
# Kept so any old call-sites that import CurrentConfig from here still work.

from .schemas import CurrentConfig   # noqa: E402  (after the main logic)

def personality_to_elara_config(p: PersonalityVector) -> CurrentConfig:
    """Derive the legacy 4-field ElaraConfig from a PersonalityVector."""
    pace = "slow" if p.pace < 0.35 else "fast" if p.pace > 0.65 else "normal"
    clarity_level = 1 if p.clarity > 0.75 else 3 if p.clarity < 0.45 else 2
    confirm = "high" if p.patience > 0.70 else "medium" if p.patience > 0.45 else "low"
    patience_mode = p.patience > 0.75 or p.warmth > 0.82
    return CurrentConfig(
        pace=pace,
        clarity_level=clarity_level,
        confirmation_frequency=confirm,
        patience_mode=patience_mode,
        personality=p,
    )
