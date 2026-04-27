"""
Personality Vector — 9D continuous style parameters for ELARA.

Three-layer pipeline per reply:
    P (Personal Profile)   — long-term baseline, slow EMA across sessions
    S (Session Bandit)     — per-session LinUCB factor (learned in-session)
    C (Context Gate)       — per-turn affect-derived caps/floors, not learned

Final params = clip(P * S, 0, 1), then each dim gated by C[affect].
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

DIMS = [
    "warmth",        # empathy / emotional support
    "humor",         # joke / playful language frequency
    "playfulness",   # childlike whimsy vs mature/serious
    "formality",     # casual ("hey!") vs formal ("Good afternoon")
    "verbosity",     # brief vs elaborate
    "pace",          # slow/deliberate vs quick
    "clarity",       # high = simpler language; low = richer vocabulary
    "patience",      # repetition, confirmations, elaboration
    "assertiveness", # gentle suggestions vs direct/confident advice
]
N_PERSONALITY_DIMS = len(DIMS)  # 9
DIM_IDX: Dict[str, int] = {name: i for i, name in enumerate(DIMS)}


class PersonalityVector(BaseModel):
    warmth:        float = Field(0.60, ge=0.0, le=1.0)
    humor:         float = Field(0.30, ge=0.0, le=1.0)
    playfulness:   float = Field(0.30, ge=0.0, le=1.0)
    formality:     float = Field(0.40, ge=0.0, le=1.0)
    verbosity:     float = Field(0.50, ge=0.0, le=1.0)
    pace:          float = Field(0.50, ge=0.0, le=1.0)
    clarity:       float = Field(0.70, ge=0.0, le=1.0)
    patience:      float = Field(0.50, ge=0.0, le=1.0)
    assertiveness: float = Field(0.30, ge=0.0, le=1.0)

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, d) for d in DIMS], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "PersonalityVector":
        vals = np.clip(arr, 0.0, 1.0)
        return cls(**{d: float(vals[i]) for i, d in enumerate(DIMS)})

    def apply_gate(self, affect: str) -> "PersonalityVector":
        """Clamp dims to affect-derived caps/floors. Returns a new vector."""
        caps, floors = CONTEXT_GATE.get(affect, ({}, {}))
        d = self.model_dump()
        for dim, cap in caps.items():
            d[dim] = min(d[dim], cap)
        for dim, floor_val in floors.items():
            d[dim] = max(d[dim], floor_val)
        return PersonalityVector(**d)


# ── Cold-start defaults per affect ────────────────────────────────────────────

AFFECT_DEFAULTS: Dict[str, PersonalityVector] = {
    "calm": PersonalityVector(
        warmth=0.60, humor=0.30, playfulness=0.30, formality=0.40,
        verbosity=0.50, pace=0.50, clarity=0.70, patience=0.50, assertiveness=0.30,
    ),
    "frustrated": PersonalityVector(
        warmth=0.75, humor=0.05, playfulness=0.10, formality=0.50,
        verbosity=0.40, pace=0.30, clarity=0.90, patience=0.85, assertiveness=0.20,
    ),
    "confused": PersonalityVector(
        warmth=0.65, humor=0.10, playfulness=0.20, formality=0.40,
        verbosity=0.55, pace=0.35, clarity=0.90, patience=0.80, assertiveness=0.20,
    ),
    "sad": PersonalityVector(
        warmth=0.90, humor=0.05, playfulness=0.15, formality=0.30,
        verbosity=0.50, pace=0.35, clarity=0.70, patience=0.85, assertiveness=0.10,
    ),
    "disengaged": PersonalityVector(
        warmth=0.70, humor=0.35, playfulness=0.40, formality=0.30,
        verbosity=0.40, pace=0.55, clarity=0.70, patience=0.60, assertiveness=0.25,
    ),
}


# ── Context Gate ──────────────────────────────────────────────────────────────
# (caps, floors) per affect state.
# caps  — upper bound on a dimension regardless of what the bandit learned
# floors — lower bound (ensures minimum empathy/patience in distressed states)

CONTEXT_GATE: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {
    "sad": (
        {"humor": 0.15, "assertiveness": 0.25, "playfulness": 0.20},
        {"warmth": 0.75, "patience": 0.70},
    ),
    "frustrated": (
        {"humor": 0.30, "assertiveness": 0.40},
        {"warmth": 0.55, "patience": 0.70, "clarity": 0.70},
    ),
    "confused": (
        {"humor": 0.35, "assertiveness": 0.30},
        {"patience": 0.70, "clarity": 0.65},
    ),
    "disengaged": (
        {"verbosity": 0.55},
        {"warmth": 0.60},
    ),
    "calm": ({}, {}),
}
