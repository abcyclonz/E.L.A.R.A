"""
Layer 3 — Discounted LinUCB Contextual Bandit

Per-action matrices (19 actions × 14 features):
  A[a]  — 14×14 covariance matrix   (init: identity)
  b[a]  — 14×1 reward-weighted feature vector (init: zeros)

Feature vector (14D):
  [ One-Hot Affect (5D) | PersonalityVector (9D) ]

  Affect one-hot positions: 0=frustrated, 1=confused, 2=sad, 3=calm, 4=disengaged

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

Selection:  θ[a] = A[a]⁻¹·b[a];  score = θᵀx + α√(xᵀA⁻¹x)
Update:     A[a] = γ·A[a] + (1−γ)·I + x·xᵀ;  b[a] = γ·b[a] + r·x
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List

from state_classifier import N_CONTEXTS, AFFECT_MAP

N_ACTIONS = 19

# Rule-based fallbacks: affect_idx → action_id (used during cold start)
RULE_BASED_DEFAULTS = {
    0: 12,   # frustrated   → DECREASE_PACE
    1: 13,   # confused     → INCREASE_CLARITY
    2: 15,   # sad          → INCREASE_PATIENCE
    3:  0,   # calm         → DO_NOTHING
    4:  3,   # disengaged   → INCREASE_HUMOR
}

# Only empathy actions allowed when user is sad
SAD_ALLOWED_ACTIONS = {0, 1, 15, 16}

_CALM_IDX = AFFECT_MAP["calm"]
_SAD_IDX  = AFFECT_MAP["sad"]


class LinUCBBandit:
    def __init__(self, A: np.ndarray, b: np.ndarray, alpha: float = 1.0, gamma: float = 0.99):
        self.A         = A.copy()
        self.b         = b.copy()
        self.alpha     = alpha
        self.gamma     = gamma
        self.d         = A.shape[1]
        self.n_actions = A.shape[0]

    def select_action(self, x: np.ndarray) -> Tuple[int, List[float]]:
        p = np.zeros(self.n_actions)
        x = x.reshape(-1, 1)

        for a in range(self.n_actions):
            A_inv       = np.linalg.inv(self.A[a])
            theta       = A_inv @ self.b[a].reshape(-1, 1)
            uncertainty = np.sqrt(x.T @ A_inv @ x)
            p[a]        = (theta.T @ x) + self.alpha * uncertainty

        action_id = int(np.argmax(p))
        return action_id, p.flatten().tolist()

    def update(self, x: np.ndarray, action_id: int, reward: float):
        x = x.reshape(-1, 1)
        self.A[action_id] = (
            self.gamma * self.A[action_id]
            + (1 - self.gamma) * np.eye(self.d)
            + x @ x.T
        )
        self.b[action_id] = self.gamma * self.b[action_id] + (reward * x).flatten()
