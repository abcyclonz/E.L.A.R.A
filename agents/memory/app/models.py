from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# ── Inbound from Orchestrator ──────────────────────────────────────────────

class ProcessRequest(BaseModel):
    """What the orchestrator sends to the memory agent on every turn."""
    text: str
    speaker: str = "user"
    emotion: Optional[str] = None        # from video pipeline: 'happy', 'anxious'
    scene: Optional[str] = None          # scene context: 'office', 'home'
    metadata: Optional[dict] = {}


class RetrieveRequest(BaseModel):
    """What the orchestrator sends when it needs context before replying."""
    question: str
    top_n: int = 20


# ── Extracted Claims (internal) ────────────────────────────────────────────

class ClaimType(str, Enum):
    STATE = "STATE"
    BELIEF = "BELIEF"
    EVENT = "EVENT"
    IGNORE = "IGNORE"


class ExtractedClaim(BaseModel):
    type: ClaimType
    entity: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[str] = None
    confidence: float = 0.9
    topic: Optional[str] = None          # for frequency tracking
    observer: Optional[str] = "user"     # for beliefs
    entity_or_event: Optional[str] = None
    corrects_entity: Optional[str] = None  # old wrong entity to invalidate on correction


class ExtractionResult(BaseModel):
    claims: List[ExtractedClaim]


# ── Outbound to Orchestrator ───────────────────────────────────────────────

class ActiveState(BaseModel):
    entity: str
    attribute: str
    value: str
    confidence: float
    emotion: Optional[str]
    valid_from: datetime


class BeliefHistory(BaseModel):
    about: str
    attribute: str
    history: List[dict]
    current_value: Optional[str]


class RecentEvent(BaseModel):
    event_type: str
    actor: Optional[str]
    description: Optional[str] = None
    emotion: Optional[str]
    timestamp: datetime


class MemorySnapshot(BaseModel):
    """Clean resolved snapshot — this is ALL the LLM ever sees."""
    active_states: List[ActiveState]
    relevant_beliefs: List[BeliefHistory]
    recent_events: List[RecentEvent]
    last_5_turns: List[dict]
    intent: str


class ProcessResponse(BaseModel):
    status: str = "ok"
    claims_extracted: int = 0
    snapshot: Optional[MemorySnapshot] = None


# ── Episodic memory ────────────────────────────────────────────────────────

class EpisodeRequest(BaseModel):
    speaker_id: str = "user"
    user_turn: str
    assistant_turn: Optional[str] = None


class RecallRequest(BaseModel):
    question: str
    speaker_id: str = "user"
    top_k: int = 3


class Episode(BaseModel):
    id: int
    speaker_id: str
    user_turn: str
    assistant_turn: Optional[str]
    timestamp: datetime
    similarity: Optional[float] = None


class RecallResponse(BaseModel):
    episodes: List[Episode]
