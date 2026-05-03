import re
import httpx
import requests as _requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.models import AgentInput, OrchestrationResult
from app.agents import (
    route, store_and_retrieve, retrieve_only, elara_chat,
    detect_style_frustration, maybe_summarize, run_tool,
    store_episode, recall_episodes, fetch_grounding, _location_from_grounding,
    _GREETING_RE, _AFFIRMATION_RE,
)
from app import auth, cache
from app.config import settings


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


@app.on_event("startup")
def startup_event():
    auth.init_auth_db()


# ---------------------------------------------------------------------------
# Auth request/response models
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""
    age: str = ""
    preferred_language: str = ""
    background: str = ""
    interests: list = []
    conversation_preferences: list = []
    technology_usage: str = ""
    conversation_goals: list = []
    additional_info: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    session_id: str          # == user_id from auth
    user_input: str
    user_token: str
    location: Optional[str] = None  # GPS-derived city string from browser, e.g. "Trivandrum, Kerala"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_profile_into_memory(user_id: str, req: "SignupRequest") -> None:
    """Store the signup profile as memory facts so Elara knows who the user is."""
    parts = []
    if req.full_name:
        parts.append(f"My name is {req.full_name}.")
    if req.age:
        parts.append(f"I am {req.age} years old.")
    if req.preferred_language:
        parts.append(f"I prefer to speak in {req.preferred_language}.")
    if req.background:
        parts.append(f"About my background: {req.background}.")
    if req.interests:
        parts.append(f"My interests include: {', '.join(req.interests)}.")
    if req.technology_usage:
        parts.append(f"My technology usage: {req.technology_usage}.")
    if req.conversation_goals:
        parts.append(f"My conversation goals: {', '.join(req.conversation_goals)}.")
    if req.additional_info:
        parts.append(req.additional_info)

    if not parts:
        return

    profile_text = " ".join(parts)
    try:
        _requests.post(
            f"{settings.memory_agent_url}/process",
            json={"text": profile_text, "speaker": user_id,
                  "metadata": {"source": "signup_profile"}},
            timeout=30,
        )
        print(f"[Auth] Seeded profile for {user_id}: {profile_text[:80]}...")
    except Exception as e:
        print(f"[Auth] Profile seed failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/signup")
def signup_endpoint(req: SignupRequest):
    try:
        result = auth.signup(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            age=req.age,
            preferred_language=req.preferred_language,
            background=req.background,
            interests=req.interests,
            conversation_preferences=req.conversation_preferences,
            technology_usage=req.technology_usage,
            conversation_goals=req.conversation_goals,
            additional_info=req.additional_info,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    user = auth.get_user(result["user_id"])
    _seed_profile_into_memory(result["user_id"], req)
    return {"user": user, "access_token": result["token"]}


@app.post("/auth/login")
def login_endpoint(req: LoginRequest):
    try:
        result = auth.login(req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return {"user": result["row"], "access_token": result["token"]}


# ---------------------------------------------------------------------------
# Chat endpoint (auth-gated wrapper around existing pipeline)
# ---------------------------------------------------------------------------

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Verify token
    user_id = auth.verify_token(req.user_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Use the session_id (== user UUID) as the speaker_id so memories are namespaced per user
    speaker_id = req.session_id

    # Run through the full existing pipeline (mirrors handle_input logic)
    agent_input = AgentInput(
        text=req.user_input,
        speaker=speaker_id,
        emotion=None,
        scene=None,
        metadata={"location": req.location} if req.location else {},
    )
    result = handle_input(agent_input)

    return {
        "ai_response": result.reply,
        "session_id": speaker_id,
        "turn_count": 0,  # turn count is tracked per-speaker inside cache
        "current_router_decision": result.debug.get("router_action", "") if result.debug else "",
    }


# ---------------------------------------------------------------------------
# Profile & Memory endpoints
# ---------------------------------------------------------------------------

@app.get("/get_profile/{user_id}")
def get_profile(user_id: str):
    user = auth.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_persona": user}


@app.get("/get_memories/{user_id}")
def get_memories(user_id: str):
    """
    Fetch memories for a user from the memory agent.
    Returns {facts: [...], summaries: [...]}.
    """
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{settings.memory_agent_url}/retrieve",
                json={"question": f"all facts and memories for {user_id}"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[get_memories] Failed to reach memory agent: {e}")
        return {"facts": [], "summaries": []}

    # Format active_states as flat fact strings
    active_states = data.get("active_states", [])
    facts = [
        {
            "document": f"{s.get('entity', '')} — {s.get('attribute', '')}: {s.get('value', '')}",
            **s,
        }
        for s in active_states
    ]

    # Pull summaries from recent_events if present
    recent_events = data.get("recent_events", [])
    summaries = [
        {"document": e.get("event_type", str(e))} for e in recent_events
    ]

    return {"facts": facts, "summaries": summaries}


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
    episodes         = None
    memory_used      = False
    memory_stored    = False
    claims_extracted = 0
    tool_called      = None
    speaker_id       = req.speaker or "user"

    # Fetch grounding facts once per turn — always present in Elara's context.
    grounding_facts = fetch_grounding(speaker_id)

    # Derive user location: GPS city from request metadata (web frontend) OR memory-stored location.
    user_location = (req.metadata or {}).get("location") or _location_from_grounding(grounding_facts)
    if user_location:
        print(f"[Location] Using: {user_location}")

    # ── Step 1: Implicit style frustration check ───────────────────────────
    emotion = req.emotion
    if not emotion or emotion.lower() in ("none", "normal", "neutral", "unknown"):
        # Only run the LLM style-frustration check on messages long enough to carry
        # implicit stylistic complaints (≥6 words).  Short messages and questions
        # produce too many false positives (e.g. "so what were we saying" → frustrated).
        if len(req.text.split()) >= 6 and detect_style_frustration(req.text):
            emotion = "frustrated"
            print("[Orchestrator] Style frustration injected → emotion=frustrated")

    # ── Step 2: LLM router decides the action ─────────────────────────────
    from app.agents import RouteDecision
    text_lower = req.text.strip().lower()

    # Fetch current session history so the router has conversational context.
    _current_session = cache.get_session(speaker_id)
    _recent_turns = _current_session.get("history", [])[-4:] if _current_session else []

    # Gratitude affirmations → simple canned reply (small LLMs hallucinate on these).
    # Matches messages that START with a gratitude phrase and are short (≤10 words).
    _GRATITUDE_RE = re.compile(
        r"^\s*(thanks|thank you|thank u|cheers|ta)\b",
        re.IGNORECASE,
    )
    if _GRATITUDE_RE.match(req.text.strip()) and len(req.text.split()) <= 10:
        elara_result = elara_chat(
            user_text=req.text, snapshot=None, emotion=emotion,
            speaker_id=speaker_id, scene=req.scene, reset_history=False,
            grounding_facts=grounding_facts,
        )
        return OrchestrationResult(
            reply="You're welcome! Is there anything else I can help you with?",
            memory_used=False, memory_stored=False,
            affect=elara_result["affect"], caregiver_alert=elara_result["caregiver_alert"],
            debug={"router_action": "DIRECT_CHAT", "router_reason": "gratitude affirmation"},
        )

    # Explicit search/find requests directed at Elara → USE_TOOL | web_search.
    # Small LLMs occasionally route "can you find X" to STORE_MEMORY on cold sessions.
    _EXPLICIT_SEARCH_RE = re.compile(
        r"^\s*(can you (search|find|look up|look for)|"
        r"(search|find) (me |us |for |up )?|"
        r"look (it |that |them )?(up|for me)|"
        r"(what.?s|what is|where (is|are)) .{3,50}\b(today|now|near(by| me)|in [a-z ]{3,20})\b|"
        r"find (me |us )?(some |a |an |nearby |the best )?[a-z ]{3,40}"
        r"(near(by| me| [a-z]{3,20})|in [a-z ]{3,20}))",
        re.IGNORECASE,
    )

    # Reminder keyword short-circuits — small LLMs frequently mis-route these to STORE_MEMORY.
    _REMINDER_SET_RE = re.compile(
        r"\b(remind me|set a reminder|don.?t let me forget|"
        r"add a reminder|create a reminder|make a reminder)\b",
        re.IGNORECASE,
    )
    _REMINDER_LIST_RE = re.compile(
        r"\b(what reminders|list (my )?reminders|show (my )?reminders|"
        r"do i have (any )?reminders|my reminders|check (my )?reminders)\b",
        re.IGNORECASE,
    )

    # Elara-complaint messages → DIRECT_CHAT (small LLMs route these to STORE_MEMORY).
    _ELARA_COMPLAINT_RE = re.compile(
        r"\b(i already told you|you never remember|you.?re not listening|"
        r"why do you keep|stop (talking|saying|asking|it|this|that)|you already asked|"
        r"i.?ve said this|you.?re repeating|you don.?t listen|"
        r"just stop|bro stop|i (didn.?t|never|haven.?t) said (that|this|any)|"
        r"i never said|that.?s not what i said|what are you (even )?(talking|saying) about|"
        r"you.?re (making this up|wrong about this)|i don.?t (know what|understand why) you.?re)\b",
        re.IGNORECASE,
    )

    # Memory retrieval questions → RETRIEVE_MEMORY (small LLMs mis-route these to STORE_MEMORY).
    _MEMORY_QUESTION_RE = re.compile(
        r"^\s*(do you remember\b|can you recall\b|what do you (know|remember) about me\b|"
        r"what (have|did) i (tell|told) you\b|tell me what you (know|remember)\b|"
        r"do you know (my|what i|where my|when i)\b|"
        r"(so\s+)?what (were|was) (we|i) (saying|talking about|discussing|on about)\b|"
        r"where were we\b|what (were|was) (we|i) (up to|on about)\b)",
        re.IGNORECASE,
    )

    # Elara-directed questions (opinion, wellbeing, etc.) → DIRECT_CHAT.
    _ELARA_OPINION_RE = re.compile(
        r"^\s*(what do you (think|feel|reckon) (of|about|regarding)\b|"
        r"how (have|are) you (been|doing|feeling|going)\b|"
        r"how.?s your (day|week|morning|evening)\b|"
        r"are you (ok(ay)?|alright|doing (ok|well|good))\b|"
        r"what have you been up to\b|"
        r"how are you (today|lately|these days)\b)",
        re.IGNORECASE,
    )

    # Apologies / social recoveries → DIRECT_CHAT (nothing worth storing).
    _APOLOGY_RE = re.compile(
        r"^\s*(sorry[,.]?\s+(i (didn.?t|just|am|was)|i.?m)|"
        r"i.?m sorry[,.]?\s+i\b|i apologize[,.]|forgive me[,.] i)\b",
        re.IGNORECASE,
    )

    # Positive social feedback about Elara → DIRECT_CHAT.
    _SOCIAL_POSITIVE_RE = re.compile(
        r"\b(good|great|wonderful|helpful|lovely|nice|brilliant) (company|companion|friend|assistant)\b|"
        r"\bi (really )?(enjoy|like|love) (talking|chatting|speaking) (to|with) you\b",
        re.IGNORECASE,
    )

    # Style preference requests → DIRECT_CHAT (not a storable personal fact).
    _STYLE_FEEDBACK_RE = re.compile(
        r"\b(speak (more )?(normally|naturally|simply|casually|like (a human|yourself))|"
        r"talk (more )?(normally|naturally|simply|casually)|"
        r"just (be |speak |talk )?(normal|natural|yourself|casual|relaxed|chill|simple|friendly)|"
        r"stop being so (formal|stiff|robotic|weird|strange)|"
        r"(too|very|so) (formal|stiff|robotic|weird|strange)|"
        r"(be|act|sound) (more )?(normal|natural|casual|human|friendly|yourself))\b",
        re.IGNORECASE,
    )

    # Implicit nearby-service search — "I could really use a coffee", "I could do with a taxi".
    # These express a tangible want/need that maps to a nearby search, not a personal fact.
    _IMPLICIT_SEARCH_RE = re.compile(
        r"\b(i could (really )?use (a |an |some )?|i could do with (a |an |some )?|"
        r"i('m| am) (really |so )?(craving|desperate for|dying for) (a |an |some )?)"
        r"(coffee|tea|food|pizza|burger|sandwich|snack|meal|breakfast|lunch|dinner|"
        r"taxi|cab|doctor|pharmacy|chemist|hospital|bakery|cafe|restaurant|shop|store|"
        r"drink|beer|juice|water|ice cream)\b",
        re.IGNORECASE,
    )

    # Tool-search correction — "oh sorry I meant restaurants, not hotels" after a prior USE_TOOL turn.
    # Detected only when the last router action was USE_TOOL to avoid false-positives on fact corrections.
    _TOOL_CORRECTION_RE = re.compile(
        r"\b((oh |so )?sorry[,]?\s+i meant|no[,]?\s+i meant|i meant .{1,40} not |"
        r"actually (search for|find|look for)|not (hotels?|restaurants?|cafes?|shops?|hospitals?)"
        r"[,]?\s+i meant)\b",
        re.IGNORECASE,
    )

    _last_action = cache.get_last_action(speaker_id)

    # Greetings → DIRECT_CHAT, skip memory entirely (also triggers history reset).
    if _GREETING_RE.match(req.text.strip()):
        decision = RouteDecision("DIRECT_CHAT", "greeting")
        print("[Orchestrator] Greeting short-circuit → DIRECT_CHAT")
    # Short-circuit for other single-word affirmations: always DIRECT_CHAT, no memory.
    elif _AFFIRMATION_RE.match(req.text.strip()):
        decision = RouteDecision("DIRECT_CHAT", "short affirmation")
        print("[Orchestrator] Affirmation short-circuit → DIRECT_CHAT")
    elif _EXPLICIT_SEARCH_RE.match(req.text.strip()):
        decision = RouteDecision("USE_TOOL", "explicit search/find directed at Elara", tool="web_search")
        print("[Orchestrator] Explicit search short-circuit → USE_TOOL | web_search")
    elif _REMINDER_SET_RE.search(req.text):
        decision = RouteDecision("USE_TOOL", "reminder keyword", tool="reminder")
        print("[Orchestrator] Reminder short-circuit → USE_TOOL | reminder")
    elif _REMINDER_LIST_RE.search(req.text):
        decision = RouteDecision("USE_TOOL", "list reminders keyword", tool="reminder")
        print("[Orchestrator] List-reminders short-circuit → USE_TOOL | reminder")
    elif _TOOL_CORRECTION_RE.search(req.text) and _last_action == "USE_TOOL":
        decision = RouteDecision("USE_TOOL", "correction of prior tool search", tool="web_search")
        print("[Orchestrator] Tool-correction short-circuit → USE_TOOL | web_search")
    elif _IMPLICIT_SEARCH_RE.search(req.text):
        decision = RouteDecision("USE_TOOL", "implicit nearby-service search", tool="web_search")
        print("[Orchestrator] Implicit search short-circuit → USE_TOOL | web_search")
    elif _ELARA_COMPLAINT_RE.search(req.text):
        decision = RouteDecision("DIRECT_CHAT", "Elara complaint — handle empathetically")
        print("[Orchestrator] Complaint short-circuit → DIRECT_CHAT")
    elif _MEMORY_QUESTION_RE.match(req.text.strip()):
        decision = RouteDecision("RETRIEVE_MEMORY", "memory retrieval question")
        print("[Orchestrator] Memory question short-circuit → RETRIEVE_MEMORY")
    elif _ELARA_OPINION_RE.match(req.text.strip()):
        decision = RouteDecision("DIRECT_CHAT", "Elara-directed opinion question")
        print("[Orchestrator] Opinion question short-circuit → DIRECT_CHAT")
    elif _APOLOGY_RE.match(req.text.strip()):
        decision = RouteDecision("DIRECT_CHAT", "apology / social recovery")
        print("[Orchestrator] Apology short-circuit → DIRECT_CHAT")
    elif _SOCIAL_POSITIVE_RE.search(req.text):
        decision = RouteDecision("DIRECT_CHAT", "positive social feedback")
        print("[Orchestrator] Social positive short-circuit → DIRECT_CHAT")
    elif _STYLE_FEEDBACK_RE.search(req.text):
        decision = RouteDecision("DIRECT_CHAT", "style preference request")
        print("[Orchestrator] Style feedback short-circuit → DIRECT_CHAT")
    else:
        decision = route(req.text, emotion, recent_turns=_recent_turns)
    print(f"[Orchestrator] Route → {decision.action} | {decision.reason}")

    # ── Step 3: Execute the routed action ─────────────────────────────────

    if decision.action == "STORE_MEMORY":
        result = store_and_retrieve(_build_memory_payload(req, emotion))
        snapshot = result.get("snapshot")
        claims_extracted = result.get("claims_extracted", 0)
        memory_stored = claims_extracted > 0
        memory_used = True

    elif decision.action == "RETRIEVE_MEMORY":
        snapshot = retrieve_only(req.text, speaker_id=speaker_id)
        memory_used = bool(snapshot and snapshot.get("active_states"))
        # Also search past episodes for episodic recall ("remember when...")
        episodes = recall_episodes(req.text, speaker_id)

    elif decision.action == "STORE_AND_RETRIEVE":
        result = store_and_retrieve(_build_memory_payload(req, emotion))
        snapshot = result.get("snapshot")
        claims_extracted = result.get("claims_extracted", 0)
        memory_stored = claims_extracted > 0
        memory_used = True

    elif decision.action == "USE_TOOL":
        tool_called = decision.tool
        tool_result = run_tool(tool_called, req.text, speaker_id, recent_turns=_recent_turns, user_location=user_location)
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
        episodes=episodes,
        grounding_facts=grounding_facts,
    )

    # ── Step 5: Store this turn as an episode (episodic memory) ──────────
    store_episode(speaker_id, req.text, elara_result.get("reply", ""))

    # ── Step 6: Trigger conversation summarization every N turns ──────────
    last_turns = elara_result.get("last_turns", [])
    maybe_summarize(speaker_id, last_turns)

    # Persist the router action so the next turn can detect tool corrections.
    cache.set_last_action(speaker_id, decision.action)

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
