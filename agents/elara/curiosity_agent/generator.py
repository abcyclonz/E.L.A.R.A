"""
Curiosity generator — passive LLM pass over the user's memory profile.

Produces a queue of CuriosityItems: questions Elara is genuinely curious to ask,
each annotated with when to ask them and when not to.

This runs every N turns (not on every turn) and is intentionally slow — it's a
reflective pass over what Elara knows, not a real-time classifier.
"""

import json
import logging
import os

import requests

from .schemas import CuriosityItem

log = logging.getLogger(__name__)

OLLAMA_URL  = os.environ.get("OLLAMA_URL",  "http://localhost:11434") + "/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")

_SYSTEM = """\
You are the curiosity engine inside ELARA, a warm AI companion for elderly users.
You have been given what ELARA knows about the user and their recent conversation.
Your job: generate 6-8 questions ELARA is genuinely curious to ask — things a caring
friend might wonder about based on what they know.

Good question examples:
  - "Hey, how's David doing these days?"
  - "Did your knee end up feeling any better this week?"
  - "You mentioned wanting to visit the coast — did that ever happen?"
  - "I think Steven needs a girlfriend, haha — is he still single?"
  - "You love cupcakes — did you find a good bakery nearby yet?"

Bad examples (too clinical, too survey-like):
  - "I see from your profile that you know David. How is he?"
  - "According to your records, you have knee pain. Has it improved?"

Rules:
- Make questions feel spontaneous, warm, and conversational
- Cover a mix of: people in their life, health check-ins, hobbies/interests, places, events
- Vary emotional weight (some light and fun, some gently caring)
- Do NOT ask about things clearly discussed in the last 5 turns

For each question, provide a JSON object with these exact fields:
  question              : string — the exact words ELARA would say
  topic_triggers        : list of strings — words in the user's message that make this timely
                          (empty list = ask at any point in conversation)
  suppress_if_topics    : list of strings — words indicating WRONG timing
                          (e.g. for a fun cupcake question: ["funeral","hospital","crying","grief","dead"])
  suppress_if_affects   : list from [sad, frustrated, confused, disengaged] — affects that block this
  emotional_sensitivity : float 0.0–1.0 (0=safe anytime, 1=only when user is calm and happy)
  priority              : float 0.0–1.0 (how valuable or caring this question is)

Return ONLY a valid JSON array of objects. No explanation, no markdown fences."""

_USER_TMPL = """\
User memory profile:
{memory_context}

Recent conversation (last 6 turns):
{recent_turns}

Generate the curiosity question list now."""


def generate_curiosity_items(
    memory_context: str,
    recent_turns: list,
) -> list[CuriosityItem]:
    formatted = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in recent_turns[-6:]
    ) or "(none yet)"

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER_TMPL.format(
            memory_context=memory_context or "(no stored memories yet)",
            recent_turns=formatted,
        )},
    ]

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "messages": messages,
                "stream":  False,
                "options": {"num_predict": 1000, "temperature": 0.75},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            log.warning("curiosity generator: no JSON array in LLM output")
            return []

        items = []
        for d in json.loads(raw[start:end]):
            try:
                items.append(CuriosityItem(**d))
            except Exception as e:
                log.debug("curiosity item parse error: %s — %s", e, d)

        log.info("curiosity generator: produced %d items", len(items))
        return items

    except Exception as exc:
        log.warning("curiosity generator failed: %s", exc)
        return []
