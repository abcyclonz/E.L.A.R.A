import json
import re
import requests
from app.config import settings
from app.models import ExtractedClaim, ExtractionResult, ClaimType

EXTRACTION_PROMPT = """You are a memory extraction engine. Extract structured memory claims from the user's message.

Classify each claim as:
- STATE: mutable fact that can change (age, relationship, job, location, mood)
- BELIEF: subjective opinion or feeling ("I think", "I feel", "I hate", "I trust")
- EVENT: something discrete that happened (fight, meeting, achievement)
- IGNORE: small talk, greetings, filler

CRITICAL RULES:
1. Pronoun resolution: Never use "he", "she", "they", "it" as the entity. Always resolve to the actual person/role.
   Example: "he plays guitar" after "I'm angry with my son" → entity = "son", NOT "he".

2. Correction detection: If the user corrects a previous statement (phrases like "actually", "I meant", "not X, it's Y", "you got it wrong", "wrong, it's"), extract the corrected fact AND add "corrects_entity" with the old wrong entity name.
   Example: "actually my son plays guitar, not my neighbour" → entity="son", corrects_entity="neighbour", attribute="activity", value="plays guitar"

Return ONLY a JSON object, no markdown, no explanation:
{{"claims": [{{"type": "STATE", "entity": "neighbour", "attribute": "relationship", "value": "hate", "confidence": 0.95, "topic": "neighbour"}}]}}

For BELIEF claims add: "observer": "user", "entity_or_event": "<subject>"
For EVENT claims add: "entity_or_event": "<event_name>"
For corrections add: "corrects_entity": "<old wrong entity>"

Message: {text}
Emotion: {emotion}
Scene: {scene}

JSON:"""

INTENT_PROMPT = """Classify the retrieval intent. Return ONE word only.

Options: CURRENT_STATE, PAST_BELIEF, EVENT, HISTORY, GENERAL

CURRENT_STATE = asking about current status
PAST_BELIEF = asking about past feelings/opinions  
EVENT = asking what happened
HISTORY = asking how something changed over time
GENERAL = everything else

Question: {question}

Answer:"""


def _parse_json(raw: str) -> dict:
    """Robustly extract JSON from any LLM response."""
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Find { ... } boundaries
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in: {raw[:300]}")


def _call_ollama(prompt: str) -> str:
    """Call local Ollama instance."""
    response = requests.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,    # Low temp for consistent JSON
                "top_p": 0.9,
                "num_predict": 500
            }
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_claims(text: str, emotion: str = None, scene: str = None) -> ExtractionResult:
    """Extract typed memory claims from raw text using local LLM."""
    try:
        prompt = EXTRACTION_PROMPT.format(
            text=text,
            emotion=emotion or "unknown",
            scene=scene or "unknown"
        )
        raw = _call_ollama(prompt)
        print(f"[Extractor] Raw: {raw[:300]}")

        data = _parse_json(raw)
        claims = []
        for c in data.get("claims", []):
            try:
                claim = ExtractedClaim(**c)
                if claim.type != ClaimType.IGNORE:
                    claims.append(claim)
            except Exception as e:
                print(f"[Extractor] Skipping claim {c}: {e}")
                continue

        print(f"[Extractor] Extracted {len(claims)} claims")
        return ExtractionResult(claims=claims)

    except Exception as e:
        print(f"[Extractor] Failed: {e}")
        return ExtractionResult(claims=[])


def classify_intent(question: str) -> str:
    """Classify retrieval intent using local LLM."""
    try:
        prompt = INTENT_PROMPT.format(question=question)
        raw = _call_ollama(prompt)
        intent = raw.strip().upper().split()[0].rstrip(".,:")
        valid = {"CURRENT_STATE", "PAST_BELIEF", "EVENT", "HISTORY", "GENERAL"}
        result = intent if intent in valid else "GENERAL"
        print(f"[Intent] {result}")
        return result
    except Exception as e:
        print(f"[Intent] Failed: {e}")
        return "GENERAL"


def embed_text(text: str) -> list[float]:
    """Generate embedding using nomic-embed-text via Ollama."""
    try:
        response = requests.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"[Embedder] Failed: {e}")
        return []