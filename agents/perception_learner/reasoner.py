"""
reasoner.py — Cross-modal inference engine.

For each perception snapshot:
1. Retrieve relevant memory context from the memory agent
2. Ask Mistral to reason across both and generate a grounded inference
3. If meaningful, store the inference back to the memory agent as an EVENT
"""
import json
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

MEMORY_AGENT_URL = os.environ.get("MEMORY_AGENT_URL", "http://memory_agent:8000")
OLLAMA_URL       = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL", "mistral:latest")
TIMEOUT_S        = int(os.environ.get("REASONER_TIMEOUT_S", "120"))

_SYSTEM = (
    "You are a perceptive learning system. You receive a perception snapshot "
    "(what a camera observed: the user's emotion, the scene, subjects present) "
    "and memory context (facts known about the user from conversations). "
    "Your job is to produce ONE grounded inference sentence that meaningfully "
    "connects both sources — something neither source alone would reveal. "
    "Output ONLY the inference sentence, or the single word SKIP if no "
    "meaningful cross-modal connection exists."
)


def _retrieve_memory(user: str, query: str) -> str:
    try:
        with httpx.Client(timeout=15) as client:
            r = client.post(
                f"{MEMORY_AGENT_URL}/retrieve",
                json={"question": query, "speaker_id": user, "llm_rerank": False},
            )
            r.raise_for_status()
            snap = r.json()
    except Exception as e:
        log.warning("memory retrieve failed: %s", e)
        return ""

    parts = []
    for s in snap.get("active_states", []):
        parts.append(f"[fact] {s['entity']} {s['attribute']}: {s['value']}")
    for b in snap.get("relevant_beliefs", []):
        if b.get("current_value"):
            parts.append(f"[belief] {b['about']} {b['attribute']}: {b['current_value']}")
    for e in snap.get("recent_events", []):
        if e.get("description"):
            parts.append(f"[event] {e['description']}")

    return "\n".join(parts)


def _call_llm(prompt: str) -> str:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "stream":  False,
        "options": {"temperature": 0.3},
    }
    with httpx.Client(timeout=TIMEOUT_S) as client:
        r = client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def _store_inference(user: str, inference: str, snapshot: dict) -> bool:
    try:
        with httpx.Client(timeout=15) as client:
            r = client.post(
                f"{MEMORY_AGENT_URL}/process",
                json={
                    "text":     inference,
                    "speaker":  user,
                    "emotion":  snapshot.get("emotion"),
                    "scene":    snapshot.get("scene"),
                    "metadata": {
                        "source":      "perception_learner",
                        "snapshot_id": snapshot.get("id"),
                        "snapshot_ts": snapshot.get("ts"),
                    },
                },
            )
            r.raise_for_status()
            return True
    except Exception as e:
        log.error("store inference failed: %s", e)
        return False


def process_snapshot(snapshot: dict) -> Optional[str]:
    """
    Returns the stored inference string, or None if SKIP / error.
    Called in a thread from the async poll loop.
    """
    user     = snapshot["user"]
    emotion  = snapshot["emotion"]
    scene    = snapshot.get("scene", "")
    subjects = snapshot.get("subjects", [])
    affected = snapshot.get("emotion_affected", False)
    reason   = snapshot.get("reason", "")
    summary  = snapshot.get("summary", "")
    ts       = snapshot.get("ts", "")

    subject_str = ", ".join(subjects) if subjects else "none noted"
    query       = f"{user} {emotion} {subject_str} {scene[:80]}"

    memory_context = _retrieve_memory(user, query)
    if not memory_context:
        log.info("snapshot #%s: no memory context — skipping", snapshot.get("id"))
        return None

    prompt = f"""PERCEPTION SNAPSHOT  (timestamp: {ts})
- Person present : {user}
- Emotion        : {emotion}  (confidence: {snapshot.get('confidence', 0):.0%})
- Scene          : {scene}
- Subjects seen  : {subject_str}
- Emotion caused by scene: {affected}
- Reason         : {reason or 'none'}
- Summary        : {summary}

MEMORY CONTEXT  (what we know about {user} from conversations):
{memory_context}

Generate ONE factual, grounded inference sentence that meaningfully connects the perception snapshot with the memory context. Only state something new — do not restate what is already in memory or the snapshot. Be specific. Use past tense.

Good examples:
- "User appeared happy while their tall son was present in the scene holding a basketball, suggesting a positive shared sporting moment."
- "User looked sad near photographs, consistent with their known grief over a deceased spouse."

If no meaningful connection exists between what was seen and what is known, respond with exactly: SKIP"""

    try:
        inference = _call_llm(prompt)
    except Exception as e:
        log.error("LLM call failed for snapshot #%s: %s", snapshot.get("id"), e)
        return None

    if not inference or inference.strip().upper() == "SKIP":
        log.info("snapshot #%s: SKIP", snapshot.get("id"))
        return None

    log.info("snapshot #%s → inference: %s", snapshot.get("id"), inference[:140])

    if _store_inference(user, inference, snapshot):
        return inference
    return None
