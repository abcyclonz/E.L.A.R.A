# Memory Agent

Versioned multimodal memory for your multiagent system.
Based on the PDF architecture: immutable log + typed memory layers + intent-aware retrieval.

## Architecture

```
Orchestrator
    │
    ├── POST /process   → on every user turn (write + snapshot)
    └── POST /retrieve  → before generating a reply (read-only)

Memory Layers:
    memory_logs     → immutable tape (never queried during chat)
    state_memory    → versioned facts (age, relationship, job)
    belief_memory   → versioned opinions (feelings, perceptions)
    event_memory    → immutable events + semantic embeddings
    topic_frequency → deterministic priority tracking
```

## Setup

### 1. Clone and configure
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Start everything
```bash
docker-compose up --build
```

### 3. Verify
```bash
curl http://localhost:8000/health
```

---

## API Usage

### POST /process
Call this on every user turn. Logs, extracts, writes, and returns snapshot.

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My neighbour and I had a huge fight, I really hate him now",
    "speaker": "user",
    "emotion": "angry",
    "scene": "home"
  }'
```

### POST /retrieve
Call this before generating a reply to get clean context.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I feel about my neighbour?"
  }'
```

### Response snapshot shape
```json
{
  "active_states": [
    {"entity": "neighbour", "attribute": "relationship", "value": "hate", ...}
  ],
  "relevant_beliefs": [...],
  "recent_events": [...],
  "last_5_turns": [...],
  "intent": "CURRENT_STATE"
}
```

---

## Debug Endpoints (dev only)
```
GET /debug/states    → all active states
GET /debug/beliefs   → full belief history
GET /debug/logs      → raw conversation log
```

---

## How Your Orchestrator Uses This

```python
import httpx

MEMORY_URL = "http://memory_agent:8000"

async def on_user_message(text, emotion=None, scene=None):
    # 1. Process and store
    process_resp = await httpx.post(f"{MEMORY_URL}/process", json={
        "text": text,
        "emotion": emotion,
        "scene": scene
    })
    snapshot = process_resp.json()["snapshot"]

    # 2. Build LLM prompt with clean snapshot
    prompt = f"""
    You are a helpful AI partner.
    
    What you know about the user right now:
    {snapshot['active_states']}
    
    Recent conversation:
    {snapshot['last_5_turns']}
    
    User says: {text}
    """

    # 3. Call your main LLM with this prompt
    # response = await your_llm(prompt)
    return prompt
```

---

## Memory Layer Rules (from the PDF)

| Layer | Mutable? | Use for |
|---|---|---|
| memory_logs | Never | Audit trail, raw tape |
| state_memory | Versioned (SCD Type 2) | Facts that change over time |
| belief_memory | Versioned per observer | Opinions, feelings, perceptions |
| event_memory | Immutable | Things that happened |
| topic_frequency | Incrementing count | Priority / relevance scoring |
