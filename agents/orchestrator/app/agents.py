"""
Orchestrator agents module.

Capabilities:
  - LLM-based router (STORE_MEMORY | RETRIEVE_MEMORY | STORE_AND_RETRIEVE | DIRECT_CHAT | USE_TOOL)
  - Tool execution via MCP (web_search, reminder, calendar) with LLM param extraction
  - Query rewriting before memory retrieval
  - Conversation summarization every N turns
  - Implicit style-feedback detection → emotion injection
  - Elara session state management per speaker
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
import re
import requests
from app.config import settings
from app.tool_client import call_mcp_tool
from app import cache

# Clear conversation history if the last turn is older than this.
_SESSION_HISTORY_TTL = timedelta(hours=1)


# ── TOOL REGISTRY ──────────────────────────────────────────────────────────
# Each entry: "tool_id": "one-line description for the router"
# The router sees these descriptions when deciding whether to call a tool.

TOOL_REGISTRY: dict[str, str] = {
    "calendar":       "Schedule events or appointments on a specific date/time",
    "reminder":       "Set a reminder or check existing reminders (no specific date needed)",
    "web_search":     "Search the web for current information, news, facts, or weather",
    "health_monitor": "Check the user's health metrics or vitals from connected sensors",
}

# ── TOOL PARAMETER EXTRACTION ──────────────────────────────────────────────

_TOOL_SCHEMAS: dict[str, str] = {
    "web_search":  '{"query": "<concise search query derived from the user message>"}',
    "reminder":    '{"text": "<what to remind about>", "when": "<time or date, e.g. 8pm, tomorrow morning>"}',
    "calendar":    '{"title": "<event title>", "when": "<date and time>", "description": "<optional details>"}',
}

_PARAMS_PROMPT = """Extract the parameters for the tool call from the user's message.

Tool: {tool_name}
User message: "{text}"
Required JSON schema: {schema}

Rules:
- Output ONLY valid JSON. No explanation.
- If a field cannot be determined, use a sensible default or empty string.
- For "when" fields, preserve the exact time expression the user said (e.g. "8pm tonight", "next Monday").

JSON:"""


def extract_tool_params(tool_name: str, user_text: str) -> dict:
    """Use LLM to pull structured params from free-form user text."""
    schema = _TOOL_SCHEMAS.get(tool_name, '{"query": "<user message>"}')
    prompt = _PARAMS_PROMPT.format(tool_name=tool_name, text=user_text, schema=schema)
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0, "num_predict": 100}},
            timeout=20
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as e:
        print(f"[ToolParams] Extraction failed: {e}")
    return {"query": user_text}  # safe fallback


# ── ROUTER ─────────────────────────────────────────────────────────────────

_TOOLS_LIST = "\n".join(f"  - {k}: {v}" for k, v in TOOL_REGISTRY.items())

ROUTER_PROMPT = f"""You are a routing agent for a personal AI companion system.
Analyze the user's message and choose ONE action.

CRITICAL RULE: STORE_MEMORY is ONLY for declarative statements where the user is telling you
something about themselves. NEVER use STORE_MEMORY for questions of any kind.

ACTIONS:

STORE_MEMORY — user is asserting/stating personal facts, relationships, feelings, or preferences.
  Must be a statement, NOT a question.
  YES: "I hate my neighbour", "my son plays guitar", "I got a promotion", "I live in Kerala",
       "please keep replies short", "my name is John", "I feel anxious"
  NO:  "do you know X?", "what is X?", "search X", "can you find X"

RETRIEVE_MEMORY — user is asking whether you remember personal details they previously shared.
  Only for questions about THEIR OWN personal history, NOT general world knowledge.
  YES: "do you know my son's name?", "do you remember what I told you?",
       "what do you know about my neighbour?", "do you remember my family?"
  NO:  "do you know the capital of France", "what are the states of Kerala"

STORE_AND_RETRIEVE — user is correcting or updating previously stored personal information.
  YES: "actually my neighbour is fine now", "it was my son not my neighbour", "I moved to Mumbai"

USE_TOOL — use for ALL of these:
  (a) Questions about world facts, geography, science, history, news, weather, or current events
  (b) Explicit search requests: "search it", "search that", "look it up", "find out", "search [topic]"
  (c) Reminders and calendar requests
  Available tools:
{_TOOLS_LIST}
  YES: "what are the districts of Kerala?", "search it", "search Kerala",
       "remind me to take medicine at 8pm", "what's the weather today",
       "how many states does India have", "who is the prime minister", "look that up"
  Format: USE_TOOL | tool_name | reason

DIRECT_CHAT — greetings, small talk, questions TO ELARA, or simple replies that need nothing else.
  YES: "hey", "how are you", "tell me a joke", "ok", "thanks", "what's your name", "that's nice"

User message: "{{text}}"
Detected emotion: {{emotion}}

Output exactly ONE line:
STORE_MEMORY | <why>
RETRIEVE_MEMORY | <why>
STORE_AND_RETRIEVE | <why>
DIRECT_CHAT | <why>
USE_TOOL | <tool_name> | <why>"""

VALID_ACTIONS = {"STORE_MEMORY", "RETRIEVE_MEMORY", "STORE_AND_RETRIEVE", "USE_TOOL", "DIRECT_CHAT"}


@dataclass
class RouteDecision:
    action: str
    reason: str
    tool: Optional[str] = None


def route(text: str, emotion: str = None) -> RouteDecision:
    """LLM-based router. Falls back to STORE_MEMORY on failure."""
    prompt = ROUTER_PROMPT.format(text=text, emotion=emotion or "none")
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.0, "num_predict": 60}},
            timeout=30
        )
        response.raise_for_status()
        raw = response.json()["response"].strip()
        print(f"[Router] Raw: {raw}")

        # Scan every line — LLMs sometimes echo the format header before the answer
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            action = parts[0].upper().split()[0].rstrip(".,:")
            if action not in VALID_ACTIONS:
                continue

            if action == "USE_TOOL":
                tool   = parts[1] if len(parts) > 1 else "unknown"
                reason = parts[2] if len(parts) > 2 else ""
                if tool not in TOOL_REGISTRY:
                    closest = next((k for k in TOOL_REGISTRY if k in tool.lower()), None)
                    tool = closest or "web_search"
                return RouteDecision("USE_TOOL", reason, tool=tool)

            reason = parts[1] if len(parts) > 1 else ""
            return RouteDecision(action, reason)

        print(f"[Router] No valid action found in response, defaulting to STORE_MEMORY")
        return RouteDecision("STORE_MEMORY", "router fallback")

    except Exception as e:
        print(f"[Router] Failed: {e}")
        return RouteDecision("STORE_MEMORY", "router error fallback")


# ── STYLE FEEDBACK DETECTION ───────────────────────────────────────────────

_STYLE_PROMPT = """Does this message express frustration, annoyance, or dissatisfaction with how the AI is speaking — such as being too long, too wordy, too formal, too slow, or too much?
Answer YES or NO only.

Message: "{text}" """


def detect_style_frustration(text: str) -> bool:
    """
    Quick LLM check for implicit style feedback.
    Returns True if the user is frustrated with the communication style.
    Catches things like "you're talking for a year", "cut it out", "too much".
    """
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model,
                  "prompt": _STYLE_PROMPT.format(text=text),
                  "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 5}},
            timeout=15
        )
        response.raise_for_status()
        answer = response.json()["response"].strip().upper()
        result = answer.startswith("YES")
        if result:
            print("[StyleCheck] Style frustration detected — injecting frustrated emotion")
        return result
    except Exception as e:
        print(f"[StyleCheck] Failed: {e}")
        return False


# ── QUERY REWRITING ────────────────────────────────────────────────────────

_REWRITE_PROMPT = """Rewrite the following user question into a concise, keyword-focused search query for a memory database.
Focus on the core subject — who or what is being asked about.
Do not answer the question. Output ONLY the rewritten query, nothing else.

Examples:
  "do you remember what I told you about my neighbour?" → "user neighbour relationship"
  "how was I feeling last week?" → "user emotion feeling recent"
  "what did I say about my son?" → "user son"

Question: "{question}"
Search query:"""


def rewrite_query(question: str) -> str:
    """Rewrite a vague retrieval question into clean search terms."""
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model,
                  "prompt": _REWRITE_PROMPT.format(question=question),
                  "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 20}},
            timeout=15
        )
        response.raise_for_status()
        rewritten = response.json()["response"].strip().split("\n")[0]
        print(f"[QueryRewrite] '{question}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[QueryRewrite] Failed: {e}, using original")
        return question


# ── MEMORY AGENT CALLS ─────────────────────────────────────────────────────

def store_and_retrieve(payload: dict) -> dict:
    try:
        r = requests.post(f"{settings.memory_agent_url}/process",
                          json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Memory] Store failed: {e}")
        return {"status": "error", "snapshot": None, "claims_extracted": 0}


def retrieve_only(question: str) -> dict:
    """Retrieve with query rewriting for better results."""
    search_query = rewrite_query(question)
    try:
        r = requests.post(f"{settings.memory_agent_url}/retrieve",
                          json={"question": search_query}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Memory] Retrieve failed: {e}")
        return {}


# ── CONVERSATION SUMMARIZATION ─────────────────────────────────────────────

_SUMMARY_PROMPT = """Summarize the key facts, emotions, and events from this conversation segment.
Focus only on information worth remembering for future conversations — skip small talk.
Be concise and factual.

Conversation:
{turns}

Summary:"""

def maybe_summarize(speaker_id: str, last_turns: list) -> bool:
    """
    Increment turn counter for this speaker. If it hits the threshold,
    summarize the last N turns and store as an EVENT in memory.
    Returns True if summarization ran.
    """
    n = settings.summarize_every_n_turns
    if n <= 0 or not last_turns:
        return False

    count = cache.incr_turn_count(speaker_id)

    if count < n:
        return False

    cache.reset_turn_count(speaker_id)

    # Format turns for the summarizer
    # Elara history uses {"role": ..., "content": ...}
    turns_text = "\n".join(
        f"{t.get('role', t.get('speaker', 'unknown')).capitalize()}: {t.get('content', t.get('text', ''))}"
        for t in last_turns
        if isinstance(t, dict)
    )

    try:
        response = requests.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model,
                  "prompt": _SUMMARY_PROMPT.format(turns=turns_text),
                  "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 200}},
            timeout=45
        )
        response.raise_for_status()
        summary = response.json()["response"].strip()
        print(f"[Summarizer] Summary: {summary[:100]}...")

        # Store as a memory process call — the extractor will pull facts from it
        store_and_retrieve({
            "text": f"[Conversation summary]: {summary}",
            "speaker": speaker_id,
            "emotion": None,
            "scene": "conversation_summary",
            "metadata": {"source": "auto_summarizer"}
        })
        return True

    except Exception as e:
        print(f"[Summarizer] Failed: {e}")
        return False


# ── MEMORY FORMATTING ──────────────────────────────────────────────────────

def store_episode(speaker_id: str, user_turn: str, assistant_turn: str) -> None:
    """Fire-and-forget: store one turn-pair as an episode in the memory agent."""
    try:
        requests.post(
            f"{settings.memory_agent_url}/episode",
            json={"speaker_id": speaker_id, "user_turn": user_turn,
                  "assistant_turn": assistant_turn},
            timeout=15,
        )
    except Exception as e:
        print(f"[Episode] Store failed (non-fatal): {e}")


def recall_episodes(question: str, speaker_id: str, top_k: int = 3) -> list:
    """Semantic search over past episodes. Returns list of episode dicts."""
    try:
        r = requests.post(
            f"{settings.memory_agent_url}/recall",
            json={"question": question, "speaker_id": speaker_id, "top_k": top_k},
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("episodes", [])
    except Exception as e:
        print(f"[Episode] Recall failed (non-fatal): {e}")
        return []


def _format_memory_context(snapshot: dict, episodes: list = None) -> str:
    if not snapshot and not episodes:
        return ""
    parts = []
    active = snapshot.get("active_states", []) if snapshot else []
    if active:
        # Group by entity so the LLM can clearly distinguish facts about the user
        # from facts about third parties (son, neighbour, etc.).
        by_entity: dict[str, list[str]] = {}
        for s in active:
            entity = s["entity"]
            fact = f"{s['attribute']} = {s['value']}"
            by_entity.setdefault(entity, []).append(fact)

        for entity, facts in sorted(by_entity.items()):
            if entity == "user":
                label = "About you"
            elif entity in ("son", "daughter"):
                name = next(
                    (s["value"] for s in active
                     if s["entity"] == entity and s["attribute"] == "name"),
                    None,
                )
                label = f"About your {entity} ({name})" if name else f"About your {entity}"
            else:
                label = f"About your {entity}"
            parts.append(f"{label}: {', '.join(facts)}")

    for b in snapshot.get("relevant_beliefs", []):
        history = b.get("history", [])
        if history:
            current = b.get("current_value", history[-1].get("value", ""))
            parts.append(f"Your belief about {b['about']}: {current}")
    events = [e["event_type"] for e in snapshot.get("recent_events", []) if e.get("event_type")] if snapshot else []
    if events:
        parts.append("Recent events: " + ", ".join(events))

    if episodes:
        ep_lines = []
        for ep in episodes:
            ts = str(ep.get("timestamp", ""))[:10]
            u = ep.get("user_turn", "")[:120]
            a = ep.get("assistant_turn") or ""
            a = a[:120]
            ep_lines.append(f'[{ts}] You said: "{u}"')
            if a:
                ep_lines.append(f'         Elara replied: "{a}"')
        parts.append("Past conversations you may remember:\n" + "\n".join(ep_lines))

    return "\n".join(parts)


def _clear_stale_history(state: dict) -> dict:
    """Reset conversation history if the session has been idle for > SESSION_HISTORY_TTL.

    Bandit matrices and config are preserved — only history is cleared.
    """
    history = state.get("history", [])
    if not history:
        return state
    last_ts = history[-1].get("timestamp")
    if not last_ts:
        return state
    try:
        last_time = datetime.fromisoformat(last_ts)
        if datetime.now(timezone.utc) - last_time > _SESSION_HISTORY_TTL:
            state = dict(state)
            state["history"] = []
            state["consecutive_distress_turns"] = 0
            print("[Cache] Stale session history cleared — fresh conversation context")
    except Exception:
        pass
    return state


# ── ELARA SESSION STATE ────────────────────────────────────────────────────

_GREETING_RE = re.compile(
    r"^\s*(hey+|hi+|hello+|howdy|yo|sup|good\s+(morning|afternoon|evening|night))\s*[!.?]?\s*$",
    re.IGNORECASE,
)


def elara_chat(
    user_text: str,
    snapshot: dict = None,
    emotion: str = None,
    speaker_id: str = "user",
    scene: str = None,
    reset_history: bool = False,
    episodes: list = None,
) -> dict:
    memory_context = _format_memory_context(snapshot, episodes)
    if emotion and emotion.lower() not in ("normal", "neutral", "none", "unknown"):
        prefix = f"Sensor-detected emotion: {emotion}."
        memory_context = f"{prefix}\n{memory_context}" if memory_context else prefix

    session = cache.get_session(speaker_id)
    if session:
        if reset_history:
            session = dict(session)
            session["history"] = []
            session["consecutive_distress_turns"] = 0
            print("[Cache] History reset for greeting")
        else:
            session = _clear_stale_history(session)

    payload = {
        "message": user_text,
        "state": session,
        "backend": "ollama",
        "memory_context": memory_context or None,
    }
    try:
        r = requests.post(f"{settings.elara_url}/chat", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        cache.set_session(speaker_id, data["state"])
        diag = data.get("diagnostics", {})
        print(f"[Elara] affect={diag.get('affect')} action={diag.get('ucb_action_id')}")
        return {
            "reply": data["reply"],
            "affect": diag.get("affect", "unknown"),
            "caregiver_alert": diag.get("caregiver_alert", False),
            "last_turns": data["state"].get("history", []),
        }
    except Exception as e:
        print(f"[Elara] Failed: {e}")
        return {
            "reply": "I'm having a little trouble right now. Could you try again?",
            "affect": "unknown",
            "caregiver_alert": False,
            "last_turns": [],
        }


# ── TOOL EXECUTION ─────────────────────────────────────────────────────────

# Keywords that mean the user wants to LIST rather than SET
_LIST_KEYWORDS = ("list", "show", "what are", "do i have", "any reminders", "my reminders",
                  "what's on", "what is on", "upcoming", "scheduled")


def run_tool(tool_name: str, user_text: str, speaker_id: str = "user") -> str:
    """Execute a tool via MCP and return its result string."""
    print(f"[Tool] Running: {tool_name} for {speaker_id}")
    text_lower = user_text.lower()

    if tool_name == "web_search":
        args = extract_tool_params("web_search", user_text)
        return call_mcp_tool(settings.web_search_mcp_url, "search", args)

    if tool_name == "reminder":
        if any(kw in text_lower for kw in _LIST_KEYWORDS):
            return call_mcp_tool(settings.assistant_mcp_url, "list_reminders",
                                 {"speaker_id": speaker_id})
        args = extract_tool_params("reminder", user_text)
        args["speaker_id"] = speaker_id
        return call_mcp_tool(settings.assistant_mcp_url, "set_reminder", args)

    if tool_name == "calendar":
        if any(kw in text_lower for kw in _LIST_KEYWORDS):
            return call_mcp_tool(settings.assistant_mcp_url, "list_calendar_events",
                                 {"speaker_id": speaker_id})
        args = extract_tool_params("calendar", user_text)
        args["speaker_id"] = speaker_id
        return call_mcp_tool(settings.assistant_mcp_url, "add_calendar_event", args)

    if tool_name == "health_monitor":
        return "Health monitor is not yet connected to sensor data."

    return f"Unknown tool: {tool_name}"
