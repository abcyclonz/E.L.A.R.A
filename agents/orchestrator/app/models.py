from pydantic import BaseModel
from typing import Optional


class AgentInput(BaseModel):
    """
    What your system sends to the orchestrator on every turn.
    Matches your described format exactly.
    """
    text: str
    emotion: Optional[str] = None        # from video pipeline: 'happy', 'angry', 'normal'
    scene: Optional[str] = None          # scene description: 'user sitting in a room'
    time: Optional[str] = None           # timestamp or natural time reference
    speaker: Optional[str] = "user"
    metadata: Optional[dict] = {}


class OrchestrationResult(BaseModel):
    """What the orchestrator returns — the final reply + what happened internally."""
    reply: str
    memory_used: bool
    memory_stored: bool
    intent: Optional[str] = None
    active_states: Optional[list] = []
    affect: Optional[str] = None
    caregiver_alert: bool = False
    tool_called: Optional[str] = None     # tool name if USE_TOOL was routed
    debug: Optional[dict] = {}
