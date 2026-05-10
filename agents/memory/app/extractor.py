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

━━━ CONTEXT ━━━
SPEAKER NAME (from stored memory, "user" if unknown): {speaker_name}
NOTE: If the speaker states their name and it is NOT in EXISTING MEMORY, extract it — the session
ID is not the same as a confirmed stored name.

EXISTING MEMORY (skip re-extracting anything already here):
{existing_memory}

RECENT CONVERSATION (use to resolve pronouns like "he/she/they" to real names):
{recent_context}

CURRENT MESSAGE: {text}
Emotion: {emotion}

━━━ YOUR TASK ━━━
Extract ALL new personal facts from the message. A message may contain MULTIPLE facts —
return one claim object per fact. Skip anything already in EXISTING MEMORY.

Claim types:
  STATE  — a personal fact about a person (name, location, job, health, relationship, hobby, preference)
  BELIEF — use this ONLY when the speaker expresses a direct subjective opinion using words like
           "I think", "I feel", "I believe", "I love", "I hate", "I trust", "in my opinion".
           Fields required: observer, entity_or_event (the subject of the opinion), attribute, value.
  EVENT  — something that happened (a visit, meeting, achievement, incident)
  IGNORE — world knowledge, duplicate, or transient emotion (tired, happy, sad, frustrated)

━━━ IMPORTANCE & STABILITY ━━━
importance 0.95, stability "permanent"  → name, bereavement, serious illness, birthplace
importance 0.85, stability "permanent"  → key relationship (spouse, child) with identity details
importance 0.70, stability "stable"     → job, home address, long-term health condition
importance 0.60, stability "stable"     → hobby, regular activity, significant preference
importance 0.30, stability "transient"  → today's plan, recent activity
importance 0.10, stability "transient"  → trivial remark

━━━ RULES ━━━
1. Entity: facts about the speaker → entity = "{speaker_name}". Facts about another person → entity = their name.
2. Multiple facts in one message → return multiple claim objects in the "claims" array.
3. Pronoun resolution: resolve "he/she/they/him/her" using RECENT CONVERSATION. Never store a pronoun as entity.
4. No world knowledge (news, history, science, sports scores). Only personal facts.
5. No transient emotions/moods (tired, sad, lonely, anxious). Only permanent or long-term facts.
6. Corrections: if user corrects a stored fact, set "corrects_entity" to the old wrong entity name.

━━━ JSON SCHEMA ━━━
STATE  fields: type, entity, attribute, value, confidence, importance, stability, topic
BELIEF fields: type, observer ("{speaker_name}"), entity_or_event, attribute, value, confidence, importance, stability
EVENT  fields: type, entity_or_event, attribute ("event_description"), value, confidence, importance, stability

━━━ FEW-SHOT EXAMPLES ━━━

Example A — "My name is Robert and I was born in Dublin."
{{"claims": [
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "name", "value": "Robert", "confidence": 0.99, "importance": 0.95, "stability": "permanent", "topic": "identity"}},
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "birthplace", "value": "Dublin", "confidence": 0.95, "importance": 0.95, "stability": "permanent", "topic": "identity"}}
]}}

Example B — "My wife Sarah passed away two years ago. I miss her every day."
{{"claims": [
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "spouse_name", "value": "Sarah", "confidence": 0.99, "importance": 0.95, "stability": "permanent", "topic": "family"}},
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "spouse_status", "value": "deceased", "confidence": 0.99, "importance": 0.95, "stability": "permanent", "topic": "bereavement"}}
]}}

Example C — "I have two sons — Michael lives in London and Peter is in Edinburgh."
{{"claims": [
  {{"type": "STATE", "entity": "Michael", "attribute": "relation_to_{speaker_name}", "value": "son", "confidence": 0.99, "importance": 0.85, "stability": "permanent", "topic": "family"}},
  {{"type": "STATE", "entity": "Michael", "attribute": "location", "value": "London", "confidence": 0.95, "importance": 0.60, "stability": "stable", "topic": "family"}},
  {{"type": "STATE", "entity": "Peter", "attribute": "relation_to_{speaker_name}", "value": "son", "confidence": 0.99, "importance": 0.85, "stability": "permanent", "topic": "family"}},
  {{"type": "STATE", "entity": "Peter", "attribute": "location", "value": "Edinburgh", "confidence": 0.95, "importance": 0.60, "stability": "stable", "topic": "family"}}
]}}

Example D — "I think the local council is doing a poor job. I love gardening though."
{{"claims": [
  {{"type": "BELIEF", "observer": "{speaker_name}", "entity_or_event": "local council", "attribute": "opinion", "value": "doing a poor job", "confidence": 0.90, "importance": 0.30, "stability": "transient"}},
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "hobby", "value": "gardening", "confidence": 0.95, "importance": 0.60, "stability": "stable", "topic": "hobby"}}
]}}

Example E — "I visited my daughter Emma in Bristol last weekend. We went to a museum."
{{"claims": [
  {{"type": "STATE", "entity": "Emma", "attribute": "relation_to_{speaker_name}", "value": "daughter", "confidence": 0.99, "importance": 0.85, "stability": "permanent", "topic": "family"}},
  {{"type": "STATE", "entity": "Emma", "attribute": "location", "value": "Bristol", "confidence": 0.90, "importance": 0.60, "stability": "stable", "topic": "family"}},
  {{"type": "EVENT", "entity_or_event": "visit_to_Emma", "attribute": "event_description", "value": "visited daughter Emma in Bristol, went to a museum", "confidence": 0.95, "importance": 0.30, "stability": "transient"}}
]}}

Example F — "My doctor is Dr. Ahmed. He says I need to watch my blood pressure."
{{"claims": [
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "doctor", "value": "Dr. Ahmed", "confidence": 0.95, "importance": 0.70, "stability": "stable", "topic": "health"}},
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "health_condition", "value": "high blood pressure (monitoring required)", "confidence": 0.90, "importance": 0.85, "stability": "stable", "topic": "health"}}
]}}

Example G (correction) — "Actually I moved to Manchester, not Sheffield."
EXISTING MEMORY has: user.location = Sheffield
{{"claims": [
  {{"type": "STATE", "entity": "{speaker_name}", "attribute": "location", "value": "Manchester", "confidence": 0.99, "importance": 0.60, "stability": "stable", "topic": "location", "corrects_entity": "user"}}
]}}

━━━ NOW EXTRACT FROM THE CURRENT MESSAGE ━━━
Return ONLY a JSON object with a "claims" array. No explanation, no markdown fences.

JSON:"""

INTENT_PROMPT = """Classify the retrieval intent. Return ONE word only — no explanation.

Options:
  CURRENT_STATE = asking about someone's current status, attributes, or facts ("where does X live?", "what is X's job?")
  PAST_BELIEF   = asking about feelings, opinions, or what someone thinks/thought ("what does X think about Y?")
  EVENT         = asking what happened, what someone did, or about a specific incident ("what did X do last week?")
  HISTORY       = broad request for everything known, all memories, or the full profile ("tell me everything", "what do you know about X?", "give me all memories")
  GENERAL       = anything else

Question: {question}

Answer (ONE word):"""

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
        timeout=90,
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
        raw  = _call_ollama(prompt, max_tokens=1200)
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
