from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field


class CuriosityItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    question: str
    # Words in the user's message that make this question especially timely
    topic_triggers: List[str] = Field(default_factory=list)
    # Context words indicating this is the wrong moment to ask
    suppress_if_topics: List[str] = Field(default_factory=list)
    # Affect states where this question must be suppressed
    suppress_if_affects: List[str] = Field(default_factory=list)
    # 0.0 = safe anytime, 1.0 = only when user is calm and in good spirits
    emotional_sensitivity: float = 0.0
    # 0.0–1.0 base priority (insight value + warmth)
    priority: float = 0.5
    # How many times this has been asked already
    ask_count: int = 0
