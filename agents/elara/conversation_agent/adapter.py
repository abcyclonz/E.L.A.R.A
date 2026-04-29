"""
conversation_agent/adapter.py

Stateless conversation turn handler. Bridges the Learning Agent pipeline
(NLP signals, affect classification, LinUCB bandit, personality update)
with the LLM reply generation.

Key changes from v1:
  - SessionState now carries a PersonalityVector (9D continuous style params)
    that the bandit learns and the prompt builder consumes.
  - NLPSignals replaces the old 4-tuple; personality_hints from the LLM
    are used to directly nudge the vector (supervised signal on top of bandit).
  - Style directive is injected into the system prompt from PersonalityVector.
  - ElaraConfig is kept for backward compat but is now derived from personality.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

from conversation_agent.rag import load_persona, build_persona_prompt, retrieve
from conversation_agent.llm import collect_stream
from conversation_agent.notifier import send_caregiver_alert
from learning_agent.personality import PersonalityVector, DIMS

_PACE_TOKENS: Dict[str, int] = {"slow": 160, "normal": 100, "fast": 60}

_CURIOSITY_REFRESH_INTERVAL = 10  # turns between curiosity queue refreshes

_DEFAULT_CONFIG: Dict[str, Any] = {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": False,
}

_PERSONA = load_persona()

DISTRESS_TURN_LIMIT = 7

# Fast pre-pipeline check for immediate reward (explicit thanks)
_POSITIVE_FEEDBACK_PATTERNS = re.compile(
    r"\b("
    r"thank you|thanks|that('s| is) (helpful|great|lovely|nice|perfect|clear)|"
    r"that helped|much better|i understand (now)?|that makes sense|"
    r"got it|perfect|brilliant|wonderful|that'?s? (good|better)"
    r")\b",
    re.IGNORECASE,
)

_IMMEDIATE_POSITIVE_REWARD = 1.0

# Step size for applying personality hints directly (supervised nudge)
_HINT_STEP = 0.08


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ElaraConfig(BaseModel):
    pace: str = "normal"
    clarity_level: int = 2
    confirmation_frequency: str = "low"
    patience_mode: bool = False


class BanditState(BaseModel):
    previous_affect:     Optional[str]        = None
    previous_action_id:  Optional[int]        = None
    previous_context_id: Optional[int]        = None
    previous_config:     Optional[ElaraConfig] = None
    affect_window:       List[str]             = Field(default_factory=list)


class SessionState(BaseModel):
    session_id:        str
    interaction_count: int = 0
    config:            ElaraConfig = Field(default_factory=ElaraConfig)
    personality:       PersonalityVector = Field(default_factory=PersonalityVector)
    bandit:            BanditState = Field(default_factory=BanditState)
    history:           List[ConversationTurn] = Field(default_factory=list)

    consecutive_distress_turns: int  = 0
    caregiver_alerted:          bool = False

    # Curiosity / proactive system
    curiosity_queue:            List[Dict[str, Any]] = Field(default_factory=list)
    curiosity_refresh_counter:  int   = 0
    last_proactive_turn:        int   = 0
    proactive_receptiveness:    float = 0.7   # EMA, learned from user reactions
    last_was_proactive:         bool  = False  # flag to capture next-turn feedback


class ChatRequest(BaseModel):
    message:        str
    state:          Optional[SessionState] = None
    backend:        str = "ollama"
    model:          Optional[str] = None
    memory_context: Optional[str] = None


class AdaptationDiagnostics(BaseModel):
    affect:                  str
    confidence:              float
    signals_used:            List[str]
    config_changes:          Dict[str, Any]
    reward_applied:          Optional[float]
    ucb_action_id:           int
    ucb_scores:              List[float]
    escalation_rule:         Optional[str] = None
    distress_turns:          int = 0
    caregiver_alert:         bool = False
    immediate_reward_applied: bool = False


class ChatResponse(BaseModel):
    reply:       str
    state:       SessionState
    diagnostics: AdaptationDiagnostics


# ── Adapter ───────────────────────────────────────────────────────────────────

class ConversationAdapter:

    MAX_HISTORY_TURNS = 10
    MAX_SERVICE_TURNS = 10

    def handle_turn(self, req: ChatRequest, learning_pipeline: Callable) -> ChatResponse:

        state = req.state or self._new_session()
        state.interaction_count += 1

        ts = datetime.now(timezone.utc).isoformat()
        state.history.append(ConversationTurn(role="user", content=req.message, timestamp=ts))

        # Pre-pipeline: immediate positive reward for explicit thanks
        immediate_reward_applied = False
        if (
            _POSITIVE_FEEDBACK_PATTERNS.search(req.message)
            and state.bandit.previous_action_id is not None
            and state.bandit.previous_config is not None
        ):
            self._apply_immediate_reward(state, _IMMEDIATE_POSITIVE_REWARD)
            immediate_reward_applied = True
            state.bandit.previous_action_id = None

        # Build Learning Agent request
        la_turns     = self._history_to_la_turns(state.history)
        la_config    = self._state_to_la_config(state)
        la_prev_cfg  = self._elara_to_la_config(state.bandit.previous_config, state.personality) \
                       if state.bandit.previous_config else None

        from learning_agent.schemas import AnalyseRequest, ConversationWindow
        la_req = AnalyseRequest(
            schema_version="1.1",
            session_id=state.session_id,
            conversation_window=ConversationWindow(turns=la_turns[-self.MAX_SERVICE_TURNS:]),
            current_config=la_config,
            previous_affect=state.bandit.previous_affect,
            previous_action_id=state.bandit.previous_action_id,
            previous_config=la_prev_cfg,
            affect_window=state.bandit.affect_window[-5:],
            interaction_count=state.interaction_count,
        )

        la_resp = learning_pipeline(la_req)

        # Update proactive receptiveness based on how user responded to last proactive turn
        if state.last_was_proactive:
            self._update_proactive_receptiveness(state, la_resp)
            state.last_was_proactive = False

        # Apply personality delta from bandit action
        if la_resp.updated_personality is not None:
            state.personality = la_resp.updated_personality

        # Apply personality hints (direct supervised nudge from NLPSignals)
        if hasattr(la_resp, "personality_hints") and la_resp.personality_hints:
            state.personality = self._apply_hints(state.personality, la_resp.personality_hints)

        # Apply context gate (caps/floors per affect)
        affect = la_resp.inferred_state.affect
        state.personality = state.personality.apply_gate(affect)

        # Sync legacy ElaraConfig from personality
        state.config = self._personality_to_elara_config(state.personality)

        # Track bandit state
        state.bandit.previous_config     = ElaraConfig(**state.config.model_dump())
        state.bandit.previous_affect     = affect
        state.bandit.previous_action_id  = la_resp.bandit_context.action_id
        state.bandit.previous_context_id = la_resp.bandit_context.context_id
        state.bandit.affect_window = (state.bandit.affect_window + [affect])[-5:]

        # Distress watchdog
        caregiver_alert = False
        if affect == "calm":
            state.consecutive_distress_turns = 0
            state.caregiver_alerted = False
        else:
            state.consecutive_distress_turns += 1
        caregiver_alert = self._check_distress_watchdog(state)

        # Curiosity: refresh queue periodically, then select a proactive question
        state.curiosity_refresh_counter += 1
        if req.memory_context and (
            not state.curiosity_queue
            or state.curiosity_refresh_counter >= _CURIOSITY_REFRESH_INTERVAL
        ):
            self._refresh_curiosity(state, req.memory_context, state.history)
            state.curiosity_refresh_counter = 0

        proactive_item = self._select_curiosity(state, req.message, affect)

        # Build system prompt with style directive
        system_prompt = build_persona_prompt(
            _PERSONA,
            req.message,
            state.config.model_dump(),
            req.memory_context,
            personality=state.personality,
        )

        if proactive_item:
            system_prompt += (
                f"\n\n[Organic curiosity — weave into reply naturally]: "
                f"After addressing the user's message, casually ask: "
                f"\"{proactive_item.question}\" — make it feel like a spontaneous thought, "
                f"not an announcement of a new topic. Keep it brief and warm."
            )

        has_prior_reply = any(t.role == "assistant" for t in state.history)
        if has_prior_reply:
            system_prompt += (
                "\n\n[IMPORTANT] Do NOT start this reply with 'Hello', 'Hi', or any greeting. "
                "The conversation is already in progress. Respond directly."
            )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.history[-(self.MAX_HISTORY_TURNS * 2):]:
            llm_role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": llm_role, "content": turn.content})

        max_tokens = _PACE_TOKENS.get(state.config.pace, 300)
        try:
            reply = collect_stream(
                messages, backend=req.backend, model=req.model,
                max_tokens=max_tokens, print_live=False,
            )
        except Exception as exc:
            log.error("LLM call failed: %s", exc)
            reply = "I'm sorry, I'm having a little trouble thinking right now. Please try again."

        # Strip spurious greeting Mistral emits after the first turn
        if has_prior_reply and reply:
            import re as _re
            reply = _re.sub(
                r"^(Hello|Hi|Hey)\b[,!\s]+(?:[A-Z][a-z]+(?: [A-Z][a-z]+)*[,!\s]+)?",
                "",
                reply.strip(),
            ).strip()

        if proactive_item:
            for d in state.curiosity_queue:
                if d.get("id") == proactive_item.id:
                    d["ask_count"] = d.get("ask_count", 0) + 1
                    break
            state.last_proactive_turn = state.interaction_count
            state.last_was_proactive  = True

        state.history.append(ConversationTurn(role="assistant", content=reply, timestamp=ts))
        if len(state.history) > self.MAX_HISTORY_TURNS * 2:
            state.history = state.history[-(self.MAX_HISTORY_TURNS * 2):]

        diag = AdaptationDiagnostics(
            affect=affect,
            confidence=la_resp.inferred_state.confidence,
            signals_used=la_resp.inferred_state.signals_used,
            config_changes=la_resp.config_delta.changes,
            reward_applied=la_resp.diagnostics.reward_applied,
            ucb_action_id=la_resp.bandit_context.action_id,
            ucb_scores=la_resp.diagnostics.ucb_scores,
            escalation_rule=la_resp.inferred_state.escalation_rule_applied,
            distress_turns=state.consecutive_distress_turns,
            caregiver_alert=caregiver_alert,
            immediate_reward_applied=immediate_reward_applied,
        )

        return ChatResponse(reply=reply, state=state, diagnostics=diag)

    # ── Curiosity / proactive system ──────────────────────────────────────────

    @staticmethod
    def _refresh_curiosity(state: "SessionState", memory_context: str, history: list) -> None:
        try:
            from curiosity_agent.generator import generate_curiosity_items
            recent = [{"role": t.role, "content": t.content} for t in history[-8:]]
            items  = generate_curiosity_items(memory_context, recent)
            if items:
                state.curiosity_queue = [item.model_dump() for item in items]
                log.info("curiosity queue refreshed: %d items for %s",
                         len(state.curiosity_queue), state.session_id)
        except Exception as exc:
            log.warning("curiosity refresh failed: %s", exc)

    @staticmethod
    def _select_curiosity(state: "SessionState", message: str, affect: str):
        if not state.curiosity_queue:
            return None
        try:
            from curiosity_agent.schemas import CuriosityItem
            from curiosity_agent.injector import select_proactive, llm_timing_check
            items = [CuriosityItem(**d) for d in state.curiosity_queue]
            item  = select_proactive(
                items, message, affect,
                state.interaction_count, state.last_proactive_turn,
                state.proactive_receptiveness,
            )
            # LLM timing veto for emotionally sensitive questions
            if item and item.emotional_sensitivity >= 0.6:
                if not llm_timing_check(item.question, message, affect):
                    log.debug("curiosity: LLM timing veto on '%s'", item.question[:50])
                    item = None
            return item
        except Exception as exc:
            log.warning("curiosity selection failed: %s", exc)
            return None

    @staticmethod
    def _update_proactive_receptiveness(state: "SessionState", la_resp) -> None:
        """Update the proactive receptiveness EMA based on how the user responded."""
        sentiment = la_resp.diagnostics.sentiment_score
        affect    = la_resp.inferred_state.affect
        ema       = state.proactive_receptiveness

        if affect in ("disengaged", "frustrated") or sentiment < -0.3:
            ema = max(0.0, round(ema - 0.25, 3))
        elif affect == "calm" and sentiment > 0.2:
            ema = min(1.0, round(ema + 0.10, 3))
        # neutral response → no change, just keep going

        state.proactive_receptiveness = ema
        log.debug("proactive_receptiveness → %.3f (affect=%s, sentiment=%.2f)",
                  ema, affect, sentiment)

    # ── Distress watchdog ─────────────────────────────────────────────────────

    @staticmethod
    def _check_distress_watchdog(state: SessionState) -> bool:
        if state.consecutive_distress_turns >= DISTRESS_TURN_LIMIT:
            if not state.caregiver_alerted:
                send_caregiver_alert(state.session_id, state.consecutive_distress_turns)
                state.caregiver_alerted = True
            return True
        return False

    # ── Immediate reward injection ────────────────────────────────────────────

    @staticmethod
    def _apply_immediate_reward(state: SessionState, reward: float) -> None:
        from learning_agent.storage import tables_locked
        from learning_agent.state_classifier import encode_context_features

        if (
            state.bandit.previous_action_id is None
            or state.bandit.previous_config  is None
            or state.bandit.previous_affect  is None
        ):
            return

        features = encode_context_features(
            state.bandit.previous_affect,
            state.personality,
        )

        from learning_agent.bandit import LinUCBBandit
        with tables_locked(user_id=state.session_id) as (A, b):
            bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)
            bandit.update(features, state.bandit.previous_action_id, reward)
            A[:] = bandit.A
            b[:] = bandit.b

    # ── Personality hint application ──────────────────────────────────────────

    @staticmethod
    def _apply_hints(personality: PersonalityVector, hints: dict) -> PersonalityVector:
        """Nudge personality toward explicit user preference signals."""
        d = personality.model_dump()
        for dim, target in hints.items():
            if target is None or dim not in d:
                continue
            curr = d[dim]
            direction = 1 if target > curr else -1
            d[dim] = round(max(0.0, min(1.0, curr + direction * _HINT_STEP)), 3)
        return PersonalityVector(**d)

    # ── ElaraConfig ↔ PersonalityVector ──────────────────────────────────────

    @staticmethod
    def _personality_to_elara_config(p: PersonalityVector) -> ElaraConfig:
        pace          = "slow" if p.pace < 0.35 else "fast" if p.pace > 0.65 else "normal"
        clarity_level = 1 if p.clarity > 0.75 else 3 if p.clarity < 0.45 else 2
        confirm       = "high" if p.patience > 0.70 else "medium" if p.patience > 0.45 else "low"
        patience_mode = p.patience > 0.75 or p.warmth > 0.82
        return ElaraConfig(
            pace=pace, clarity_level=clarity_level,
            confirmation_frequency=confirm, patience_mode=patience_mode,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _new_session() -> SessionState:
        import uuid
        return SessionState(
            session_id=f"elara-{uuid.uuid4().hex[:8]}",
            config=ElaraConfig(**_DEFAULT_CONFIG),
        )

    @staticmethod
    def _history_to_la_turns(history: List[ConversationTurn]):
        from learning_agent.schemas import Turn
        result = []
        for turn in history:
            la_role = "user" if turn.role == "user" else "agent"
            result.append(Turn(role=la_role, text=turn.content, timestamp=turn.timestamp))
        return result

    @staticmethod
    def _state_to_la_config(state: SessionState):
        from learning_agent.schemas import CurrentConfig
        return CurrentConfig(
            pace=state.config.pace,
            clarity_level=state.config.clarity_level,
            confirmation_frequency=state.config.confirmation_frequency,
            patience_mode=state.config.patience_mode,
            personality=state.personality,
        )

    @staticmethod
    def _elara_to_la_config(config: Optional[ElaraConfig], personality: PersonalityVector):
        if config is None:
            return None
        from learning_agent.schemas import CurrentConfig
        return CurrentConfig(
            pace=config.pace,
            clarity_level=config.clarity_level,
            confirmation_frequency=config.confirmation_frequency,
            patience_mode=config.patience_mode,
            personality=personality,
        )
