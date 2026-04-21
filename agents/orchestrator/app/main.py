from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import AgentInput, OrchestrationResult
from app.agents import (
    route, store_and_retrieve, retrieve_only, elara_chat,
    detect_style_frustration, maybe_summarize, run_tool,
    _GREETING_RE,
)


app = FastAPI(
    title="Orchestrator",
    description="LLM-routed multi-agent system: memory agent + Elara conversation agent",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/input", response_model=OrchestrationResult)
def handle_input(req: AgentInput):
    """
    Main entry point. LLM router decides what to do:

    STORE_MEMORY       → store to memory agent, pass snapshot to Elara
    RETRIEVE_MEMORY    → retrieve from memory agent, pass snapshot to Elara
    STORE_AND_RETRIEVE → store + retrieve, pass to Elara
    USE_TOOL           → run a tool, pass result to Elara as context
    DIRECT_CHAT        → skip memory entirely, go straight to Elara
    """
    snapshot         = None
    memory_used      = False
    memory_stored    = False
    claims_extracted = 0
    tool_called      = None
    speaker_id       = req.speaker or "user"

    # ── Step 1: Implicit style frustration check ───────────────────────────
    # If the user didn't send an explicit emotion, check for frustration signals.
    # If detected, inject "frustrated" so Elara and the router both see it.
    emotion = req.emotion
    if not emotion or emotion.lower() in ("none", "normal", "neutral", "unknown"):
        if detect_style_frustration(req.text):
            emotion = "frustrated"
            print("[Orchestrator] Style frustration injected → emotion=frustrated")

    # ── Step 2: LLM router decides the action ─────────────────────────────
    decision = route(req.text, emotion)
    print(f"[Orchestrator] Route → {decision.action} | {decision.reason}")

    # ── Step 3: Execute the routed action ─────────────────────────────────

    if decision.action == "STORE_MEMORY":
        result = store_and_retrieve(_build_memory_payload(req, emotion))
        snapshot = result.get("snapshot")
        claims_extracted = result.get("claims_extracted", 0)
        memory_stored = claims_extracted > 0
        memory_used = True

    elif decision.action == "RETRIEVE_MEMORY":
        snapshot = retrieve_only(req.text)
        memory_used = bool(snapshot and snapshot.get("active_states"))

    elif decision.action == "STORE_AND_RETRIEVE":
        result = store_and_retrieve(_build_memory_payload(req, emotion))
        snapshot = result.get("snapshot")
        claims_extracted = result.get("claims_extracted", 0)
        memory_stored = claims_extracted > 0
        memory_used = True

    elif decision.action == "USE_TOOL":
        tool_called = decision.tool
        tool_result = run_tool(tool_called, req.text, speaker_id)
        print(f"[Orchestrator] Tool result: {tool_result}")
        # Pass tool output to Elara as memory context so it can formulate the reply
        snapshot = {"tool_result": tool_result, "active_states": [], "relevant_beliefs": [], "recent_events": []}
        memory_used = True

    elif decision.action == "DIRECT_CHAT":
        print("[Orchestrator] Direct chat — skipping memory agent")

    # ── Step 4: Send to Elara conversation agent ──────────────────────────
    # For tool results, inject tool output directly into snapshot-like context
    elara_snapshot = snapshot
    if decision.action == "USE_TOOL" and tool_called:
        # Wrap tool result so Elara sees it as context (format_memory_context handles dict)
        elara_snapshot = None  # avoid broken snapshot; pass tool result via separate context
        # We'll pass it through memory_context directly by abusing the snapshot path
        # elara_chat's _format_memory_context handles active_states etc; tool result needs special handling
        tool_context_snapshot = {
            "active_states": [{"entity": "tool", "attribute": tool_called, "value": snapshot["tool_result"]}],
            "relevant_beliefs": [],
            "recent_events": [],
        }
        elara_snapshot = tool_context_snapshot

    elara_result = elara_chat(
        user_text=req.text,
        snapshot=elara_snapshot,
        emotion=emotion,
        speaker_id=speaker_id,
        scene=req.scene,
        reset_history=bool(_GREETING_RE.match(req.text.strip())),
    )

    # ── Step 5: Trigger conversation summarization every N turns ──────────
    last_turns = elara_result.get("last_turns", [])
    maybe_summarize(speaker_id, last_turns)

    # ── Step 6: Build response ─────────────────────────────────────────────
    active_states = snapshot.get("active_states", []) if snapshot else []
    intent = snapshot.get("intent") if snapshot else None

    return OrchestrationResult(
        reply=elara_result["reply"],
        memory_used=memory_used,
        memory_stored=memory_stored,
        intent=intent,
        active_states=active_states,
        affect=elara_result["affect"],
        caregiver_alert=elara_result["caregiver_alert"],
        tool_called=tool_called,
        debug={
            "router_action": decision.action,
            "router_reason": decision.reason,
            "claims_extracted": claims_extracted,
            "emotion_used": emotion,
        }
    )


def _build_memory_payload(req: AgentInput, emotion: str = None) -> dict:
    return {
        "text": req.text,
        "speaker": req.speaker or "user",
        "emotion": emotion or req.emotion,
        "scene": req.scene,
        "metadata": {
            "time": req.time,
            **(req.metadata or {})
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator", "version": "4.0.0"}
