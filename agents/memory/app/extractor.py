import json
import math
import re
import requests
from app.config import settings
from app.models import ExtractedClaim, ExtractionResult, ClaimType
from app import embed_cache

_GARBAGE_VALUE = re.compile(
    r"^(corrects?_?\w*|old_entity|unknown_entity|placeholder|<[^>]+>)$",
    re.IGNORECASE,
)

EXTRACTION_PROMPT = """You are a memory extraction engine for an elderly care AI companion.

SPEAKER NAME (confirmed from memory, or "user" if not yet known): {speaker_name}
IMPORTANT: This name comes from stored memory, NOT from the session identifier. If the speaker
says their name in the current message and it is NOT in EXISTING MEMORY, you MUST extract it —
even if {speaker_name} already looks like a name. The session ID is not the same as stored memory.

EXISTING MEMORY (already stored — do not re-extract these):
{existing_memory}

RECENT CONVERSATION (for pronoun resolution and context):
{recent_context}

CURRENT MESSAGE: {text}
Emotion: {emotion}

Extract only NEW personal facts not already in memory. Classify each as:
- STATE: mutable personal fact (name, location, relationship, job, hobby, preference)
- BELIEF: subjective opinion ("I think", "I feel", "I hate", "I trust")
- EVENT: something that happened (meeting, fight, achievement)
- IGNORE: anything else

STRICT RULES:
1. Entity naming: Facts about the speaker → entity = "{speaker_name}". Facts about others → use their name.
2. No world knowledge: Never extract general facts (capitals, history, science, sports scores, news). Only personal facts about the user and people they know.
3. No transient states: Do NOT store current emotions, moods, or temporary physical states
   (tired, sad, happy, lonely, frustrated, anxious, sleepy, hungry, bored, excited, angry, etc.).
   These are momentary and not worth persisting. Only store PERMANENT or LONG-TERM personal facts.
4. No duplicates: If the fact is already in EXISTING MEMORY above, mark as IGNORE.
   EXCEPTION: if EXISTING MEMORY has no name entry for the speaker, ALWAYS extract a stated name.
5. Pronoun resolution: Use RECENT CONVERSATION to resolve "him", "her", "they" to actual names. Never use a pronoun as entity.
6. Minimal extraction: Only extract what is clearly stated. Do not infer or speculate.

Importance scale:
  0.9+ = fundamental (name, serious illness, key relationships, bereavement)
  0.6  = notable (job, hobby, address, significant preference)
  0.3  = minor (today's plan, casual remark)
  0.1  = trivial

Stability:
  permanent = never changes (name, birthplace, deceased relatives)
  stable    = changes rarely (job, home, long-term relationships)
  transient = changes within days/weeks (plans, current activities)

Return ONLY a JSON object:
{{"claims": [{{"type": "STATE", "entity": "{speaker_name}", "attribute": "location", "value": "Kerala", "confidence": 0.95, "importance": 0.6, "stability": "stable", "topic": "location"}}]}}

For BELIEF: add "observer": "{speaker_name}", "entity_or_event": "<subject>"
For EVENT:  add "entity_or_event": "<event_name>"
For corrections: add "corrects_entity": "<old wrong entity>"

JSON:"""

INTENT_PROMPT = """Classify the retrieval intent. Return ONE word only.

Options: CURRENT_STATE, PAST_BELIEF, EVENT, HISTORY, GENERAL

CURRENT_STATE = asking about current status
PAST_BELIEF   = asking about past feelings/opinions
EVENT         = asking what happened
HISTORY       = asking how something changed over time
GENERAL       = everything else

Question: {question}

Answer:"""

RELEVANCE_PROMPT = """You are a memory relevance judge for an elderly care AI.

Current question: {question}

Rate how relevant each memory item is to answering this question.
Score 0 = completely irrelevant, 10 = directly answers the question.

Memory items:
{items}

Return ONLY JSON: {{"scores": [n, n, ...]}} — one integer per item, same order."""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON in: {raw[:300]}")


def _call_ollama(prompt: str, max_tokens: int = 500) -> str:
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": max_tokens},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["response"]


def _summarise_existing_memory(snapshot: dict | None) -> str:
    if not snapshot:
        return "None"
    lines = []
    for s in snapshot.get("active_states", []):
        lines.append(f"  {s['entity']}.{s['attribute']} = {s['value']}")
    for b in snapshot.get("relevant_beliefs", []):
        lines.append(f"  belief: {b.get('entity_or_event','?')} — {b.get('value','?')}")
    return "\n".join(lines) if lines else "None"


def _summarise_recent_context(recent_turns: list | None) -> str:
    if not recent_turns:
        return "None"
    lines = []
    for t in recent_turns[-6:]:
        role = t.get("speaker", t.get("role", "user"))
        lines.append(f"  {role}: {t.get('text', t.get('content', ''))}")
    return "\n".join(lines)


def extract_claims(
    text: str,
    emotion: str = None,
    scene: str = None,
    speaker_name: str = "user",
    existing_snapshot: dict | None = None,
    recent_turns: list | None = None,
) -> ExtractionResult:
    try:
        prompt = EXTRACTION_PROMPT.format(
            text=text,
            emotion=emotion or "none",
            speaker_name=speaker_name,
            existing_memory=_summarise_existing_memory(existing_snapshot),
            recent_context=_summarise_recent_context(recent_turns),
        )
        raw  = _call_ollama(prompt)
        print(f"[Extractor] Raw: {raw[:300]}")
        data = _parse_json(raw)

        claims = []
        for c in data.get("claims", []):
            try:
                claim = ExtractedClaim(**c)
                if claim.type == ClaimType.IGNORE:
                    continue
                if _GARBAGE_VALUE.match(str(claim.value).strip()):
                    print(f"[Extractor] Rejecting garbage claim: {c}")
                    continue
                # Clamp fields to valid ranges
                claim.importance = max(0.0, min(1.0, claim.importance))
                if claim.stability not in ("permanent", "stable", "transient"):
                    claim.stability = "stable"
                # Intent/goal attributes are inherently short-lived — never persist them long
                if str(getattr(claim, "attribute", "")).lower() in (
                    "intent", "goal", "looking_for", "searching_for", "wants"
                ):
                    claim.stability = "transient"
                    claim.importance = min(claim.importance, 0.4)
                claims.append(claim)
            except Exception as e:
                print(f"[Extractor] Skipping claim {c}: {e}")

        print(f"[Extractor] Extracted {len(claims)} claims")
        return ExtractionResult(claims=claims)

    except Exception as e:
        print(f"[Extractor] Failed: {e}")
        return ExtractionResult(claims=[])


def classify_intent(question: str) -> str:
    try:
        raw    = _call_ollama(INTENT_PROMPT.format(question=question), max_tokens=10)
        intent = raw.strip().upper().split()[0].rstrip(".,:")
        valid  = {"CURRENT_STATE", "PAST_BELIEF", "EVENT", "HISTORY", "GENERAL"}
        result = intent if intent in valid else "GENERAL"
        print(f"[Intent] {result}")
        return result
    except Exception as e:
        print(f"[Intent] Failed: {e}")
        return "GENERAL"


def score_relevance(question: str, candidates: list[dict]) -> list[float]:
    """
    LLM rates each candidate's relevance to the question (0-10).
    Returns list of floats in [0, 1] (divided by 10), same order as candidates.
    Falls back to [1.0, ...] on any failure so callers always get a valid list.
    """
    if not candidates:
        return []

    lines = []
    for i, c in enumerate(candidates, 1):
        age   = c.get("age_days", 0)
        label = (
            f"{i}. {c.get('entity', '?')}.{c.get('attribute', '?')} = "
            f"\"{c.get('value', '?')}\"  "
            f"(age: {age:.0f}d, importance: {c.get('importance', 0.5):.1f}, "
            f"stability: {c.get('stability', 'stable')})"
        )
        lines.append(label)

    prompt = RELEVANCE_PROMPT.format(
        question=question,
        items="\n".join(lines),
    )
    try:
        raw  = _call_ollama(prompt, max_tokens=80)
        data = _parse_json(raw)
        scores = data.get("scores", [])
        if len(scores) == len(candidates):
            return [max(0.0, min(1.0, float(s) / 10.0)) for s in scores]
    except Exception as e:
        print(f"[Relevance] LLM scoring failed ({e}) — using neutral scores")

    return [1.0] * len(candidates)


def embed_text(text: str) -> list[float]:
    cached = embed_cache.get(text)
    if cached is not None:
        print("[Embedder] Cache hit")
        return cached
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        vector = response.json()["embedding"]
        embed_cache.put(text, vector)
        return vector
    except Exception as e:
        print(f"[Embedder] Failed: {e}")
        return []
