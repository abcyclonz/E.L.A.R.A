"""
app.py — ELARA Unified Microservice Entry Point
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import io
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel as _BaseModel

from learning_agent.schemas import (
    AnalyseRequest, AnalyseResponse, ConversationWindow,
    CurrentConfig, Turn,
)
from learning_agent.nlp_layer import extract_signals, NLPSignals
from learning_agent.state_classifier import (
    classify_state, encode_context_id, encode_context_features,
)
from learning_agent.bandit import LinUCBBandit
from learning_agent.config_applier import apply_action, personality_to_elara_config
from learning_agent.storage import tables_locked
from learning_agent.personality import PersonalityVector, AFFECT_DEFAULTS

from conversation_agent.adapter import (
    ConversationAdapter, ChatRequest, ChatResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="ELARA — Unified Elderly Care Companion Service",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Reward computation ────────────────────────────────────────────────────────

_AFFECT_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("frustrated", "calm"):       +1.0,
    ("frustrated", "confused"):   +0.3,
    ("frustrated", "frustrated"): -0.5,
    ("confused",   "calm"):       +1.0,
    ("confused",   "confused"):   -0.3,
    ("confused",   "frustrated"): -1.0,
    ("sad",        "calm"):       +1.0,
    ("sad",        "sad"):        -0.2,
    ("calm",       "calm"):        0.0,
    ("calm",       "confused"):   -0.5,
}


def _compute_reward(prev: str, curr: str, signals: NLPSignals) -> float:
    if curr == "disengaged":
        base = -1.0
    else:
        base = _AFFECT_REWARD_TABLE.get((prev, curr), 0.0)

    # Explicit user feedback shifts reward
    if signals.explicit_positive:
        base = min(1.0, base + 0.4)
    if signals.explicit_negative:
        base = max(-1.0, base - 0.4)

    return round(base, 4)


# ── Learning pipeline ─────────────────────────────────────────────────────────

def _run_learning_pipeline(req: AnalyseRequest) -> AnalyseResponse:
    t0 = time.time()

    turns = req.conversation_window.turns

    # Layer 1: NLP signal extraction (LLM-based)
    signals: NLPSignals = extract_signals(turns)

    last_user_text = next((t.text for t in reversed(turns) if t.role == "user"), "")

    # Layer 2: affect classification
    affect, confidence, signals_used, escalation_rule = classify_state(
        signals,
        last_user_text=last_user_text,
        affect_window=req.affect_window,
    )

    # Current personality (from request config, else affect default)
    current_personality: PersonalityVector = req.current_config.personality

    # Layer 3: feature encoding (14D) and bandit
    curr_features = encode_context_features(affect, current_personality)
    context_id    = encode_context_id(affect)

    reward_applied = None

    with tables_locked(user_id=req.session_id) as (A, b):
        bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)

        if (
            req.previous_affect    is not None
            and req.previous_action_id is not None
            and req.previous_config    is not None
        ):
            prev_features = encode_context_features(
                req.previous_affect,
                req.previous_config.personality,
            )
            reward = _compute_reward(req.previous_affect, affect, signals)
            bandit.update(prev_features, req.previous_action_id, reward)
            reward_applied = reward

        action_id, ucb_scores = bandit.select_action(curr_features)

        A[:] = bandit.A
        b[:] = bandit.b

    # Apply action → update personality
    new_personality, changes, reason = apply_action(action_id, current_personality, affect)

    # Apply personality hints directly (supervised nudge from user signals)
    if signals.personality_hints:
        d = new_personality.model_dump()
        for dim, target in signals.personality_hints.items():
            if target is not None and dim in d:
                curr = d[dim]
                direction = 1 if target > curr else -1
                d[dim] = round(max(0.0, min(1.0, curr + direction * 0.06)), 3)
                if d[dim] != curr:
                    changes[dim] = d[dim]
        new_personality = PersonalityVector(**d)

    # wants_shorter / wants_simpler signals → direct personality update
    if signals.wants_shorter > 0.5:
        d = new_personality.model_dump()
        d["verbosity"] = round(max(0.0, d["verbosity"] - 0.08), 3)
        changes["verbosity"] = d["verbosity"]
        new_personality = PersonalityVector(**d)
    if signals.wants_simpler > 0.5:
        d = new_personality.model_dump()
        d["clarity"] = round(min(1.0, d["clarity"] + 0.08), 3)
        changes["clarity"] = d["clarity"]
        new_personality = PersonalityVector(**d)

    elapsed_ms = round((time.time() - t0) * 1000)

    # Derive legacy CurrentConfig from personality for backward compat
    new_cfg = personality_to_elara_config(new_personality)

    from learning_agent.schemas import InferredState, ConfigDelta, BanditContext, Diagnostics

    return AnalyseResponse(
        schema_version="1.1",
        session_id=req.session_id,
        processing_time_ms=elapsed_ms,
        inferred_state=InferredState(
            affect=affect,
            confidence=confidence,
            context_id=context_id,
            signals_used=signals_used,
            escalation_rule_applied=escalation_rule,
        ),
        config_delta=ConfigDelta(apply=len(changes) > 0, changes=changes, reason=reason),
        bandit_context=BanditContext(context_id=context_id, action_id=action_id),
        diagnostics=Diagnostics(
            sentiment_score=round(signals.sentiment, 4),
            repetition_score=round(signals.repetition, 4),
            confusion_score=round(signals.confusion, 4),
            sadness_score=round(signals.sadness, 4),
            ucb_scores=[round(s, 4) for s in ucb_scores],
            reward_applied=reward_applied,
            total_tries=0,
        ),
        updated_personality=new_personality,
    )


# ── TTS ───────────────────────────────────────────────────────────────────────

class TTSRequest(_BaseModel):
    text:    str
    voice:   str   = "bf_emma"
    backend: str   = "kokoro"
    speed:   float = 0.9


_kokoro_lock:      threading.Lock        = threading.Lock()
_kokoro_pipelines: dict[str, object]     = {}


def _get_kokoro_pipeline(lang_code: str):
    with _kokoro_lock:
        if lang_code not in _kokoro_pipelines:
            from kokoro import KPipeline
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return _kokoro_pipelines[lang_code]


def _kokoro_sync(text: str, voice: str, speed: float) -> io.BytesIO:
    import numpy as np
    import soundfile as sf
    lang_code = "b" if voice.startswith("b") else "a"
    pipeline  = _get_kokoro_pipeline(lang_code)
    chunks    = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    buf = io.BytesIO()
    if chunks:
        sf.write(buf, np.concatenate(chunks), 24000, format="WAV")
        buf.seek(0)
    return buf


async def _tts_edge(req: TTSRequest) -> StreamingResponse:
    import edge_tts
    communicate = edge_tts.Communicate(req.text, req.voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/mpeg")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse, include_in_schema=False)
def ui():
    return FileResponse("index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "ELARA Unified", "version": "3.0.0"}


@app.post("/tts", include_in_schema=False)
async def tts(req: TTSRequest):
    if req.backend == "edge":
        return await _tts_edge(req)
    import asyncio
    buf = await asyncio.get_event_loop().run_in_executor(
        None, _kokoro_sync, req.text, req.voice, req.speed,
    )
    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest) -> AnalyseResponse:
    return _run_learning_pipeline(req)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    adapter = ConversationAdapter()
    return adapter.handle_turn(req, _run_learning_pipeline)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    SSE streaming variant of /chat.
    Streams LLM tokens then sends final ChatResponse JSON as last event.
    """
    import json as _json
    from conversation_agent.adapter import (
        ConversationAdapter, ChatResponse as _CR,
        ConversationTurn, _POSITIVE_FEEDBACK_PATTERNS,
        _IMMEDIATE_POSITIVE_REWARD, _PACE_TOKENS, _PERSONA,
        ElaraConfig, AdaptationDiagnostics,
    )

    adapter = ConversationAdapter()

    def _generate():
        from datetime import datetime, timezone as _tz

        state = req.state or adapter._new_session()
        state.interaction_count += 1
        ts = datetime.now(_tz.utc).isoformat()

        state.history.append(ConversationTurn(role="user", content=req.message, timestamp=ts))

        immediate_reward_applied = False
        if (
            _POSITIVE_FEEDBACK_PATTERNS.search(req.message)
            and state.bandit.previous_action_id is not None
            and state.bandit.previous_config is not None
        ):
            adapter._apply_immediate_reward(state, _IMMEDIATE_POSITIVE_REWARD)
            immediate_reward_applied = True
            state.bandit.previous_action_id = None

        la_turns   = adapter._history_to_la_turns(state.history)
        la_config  = adapter._state_to_la_config(state)
        la_prev_cfg = adapter._elara_to_la_config(state.bandit.previous_config, state.personality) \
                      if state.bandit.previous_config else None

        from learning_agent.schemas import AnalyseRequest as _LAReq, ConversationWindow as _CW
        la_req = _LAReq(
            schema_version="1.1",
            session_id=state.session_id,
            conversation_window=_CW(turns=la_turns[-adapter.MAX_SERVICE_TURNS:]),
            current_config=la_config,
            previous_affect=state.bandit.previous_affect,
            previous_action_id=state.bandit.previous_action_id,
            previous_config=la_prev_cfg,
            affect_window=state.bandit.affect_window[-5:],
            interaction_count=state.interaction_count,
        )

        la_resp = _run_learning_pipeline(la_req)

        if la_resp.updated_personality is not None:
            state.personality = la_resp.updated_personality
        if hasattr(la_resp, "personality_hints") and la_resp.personality_hints:
            state.personality = adapter._apply_hints(state.personality, la_resp.personality_hints)

        affect = la_resp.inferred_state.affect
        state.personality = state.personality.apply_gate(affect)
        state.config      = adapter._personality_to_elara_config(state.personality)

        state.bandit.previous_config     = ElaraConfig(**state.config.model_dump())
        state.bandit.previous_affect     = affect
        state.bandit.previous_action_id  = la_resp.bandit_context.action_id
        state.bandit.previous_context_id = la_resp.bandit_context.context_id
        state.bandit.affect_window       = (state.bandit.affect_window + [affect])[-5:]

        caregiver_alert = False
        if affect == "calm":
            state.consecutive_distress_turns = 0
        else:
            state.consecutive_distress_turns += 1
        caregiver_alert = adapter._check_distress_watchdog(state)

        from conversation_agent.rag import build_persona_prompt as _bpp
        system_prompt = _bpp(
            _PERSONA, req.message, state.config.model_dump(),
            req.memory_context, personality=state.personality,
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in state.history[-(adapter.MAX_HISTORY_TURNS * 2):]:
            llm_role = "assistant" if turn.role == "assistant" else "user"
            messages.append({"role": llm_role, "content": turn.content})

        max_tokens = _PACE_TOKENS.get(state.config.pace, 300)

        from conversation_agent.llm import stream_response
        reply_parts = []
        try:
            for chunk in stream_response(messages, backend=req.backend,
                                         model=req.model, max_tokens=max_tokens):
                reply_parts.append(chunk)
                yield f"data: {_json.dumps({'token': chunk})}\n\n"
        except Exception as exc:
            log.error("LLM stream failed: %s", exc)
            fallback = "I'm sorry, I'm having a little trouble thinking right now. Please try again."
            reply_parts = [fallback]
            yield f"data: {_json.dumps({'token': fallback})}\n\n"

        reply = "".join(reply_parts)

        state.history.append(ConversationTurn(role="assistant", content=reply, timestamp=ts))
        if len(state.history) > adapter.MAX_HISTORY_TURNS * 2:
            state.history = state.history[-(adapter.MAX_HISTORY_TURNS * 2):]

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

        final = _CR(reply=reply, state=state, diagnostics=diag)
        yield f"data: {_json.dumps({'done': True, **_json.loads(final.model_dump_json())})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
