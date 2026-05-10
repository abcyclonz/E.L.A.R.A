# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**E.L.A.R.A.** — a real-time, affect-adaptive AI companion designed for elderly users. The system combines voice input, versioned memory, contextual learning, external tool access, and a physical camera eye into a multi-agent pipeline.

The system has two parallel sensory channels feeding a shared memory store:

- **Voice channel** (real-time): Pi mic → raw PCM over WebSocket → RunPod audio gateway (VAD + STT + speaker ID) → orchestrator → memory agent + tools → Elara conversation agent → Kokoro TTS → WAV back to Pi speaker
- **Vision channel** (passive, background): Raspberry Pi camera → face tracking + identity → RunPod perception monitor → emotion + scene analysis → perception learner → cross-modal inferences → same memory store

The Pi is a thin audio terminal — all ML (VAD, STT, speaker ID, TTS) runs on RunPod. The two channels never block each other. The vision side enriches memory asynchronously; the conversation side reads from that same memory naturally.

The two channels never block each other. The vision side enriches memory asynchronously; the conversation side reads from that same memory naturally.

---

## Running the System

### Full stack (all agents via Docker) — privileged pod / local machine
```bash
docker compose up --build
```

Requires a `.env` file in the project root:
```
TAVILY_API_KEY=your_key_here
```

### Full stack (no Docker) — non-privileged RunPod container
```bash
./start.sh
```
Starts all services natively: PostgreSQL, Redis, Ollama, Memory Agent, Orchestrator, Elara, Web Search Tool, Assistant Tool, Perception Learner, and Next.js frontend. Handles venv creation, DB setup, and health checks automatically.

### Pi thin client (audio terminal — copy to Raspberry Pi)
```bash
cd pi_client
cp .env.example .env
# Edit .env: set ORCHESTRATOR_WS_URL=ws://<pod-id>-8001.proxy.runpod.net/ws/audio
bash run.sh
```
Installs portaudio, creates venv, connects to RunPod audio gateway over WebSocket.

### MIC module (legacy — standalone local voice processing, no longer primary path)
```bash
cd MIC
python main.py
```

### Individual MIC modules for isolated testing
```bash
python mic_stream.py    # Test mic input (5-second capture with volume bars)
python vad_module.py    # Test VAD (prints SPEECH DETECTED or dots for silence)
python stt_module.py    # Test Whisper model download and init
```

### Interactive test client
```bash
python test_chat.py       # Sends text directly to orchestrator at :8003
python test_rigorous.py   # Multi-run rigorous test suite (3 runs × 5 scripts, 50 turns each)
```

---

## Service Map

| Service | Port | Container | Description |
|---------|------|-----------|-------------|
| Memory Agent | 8000 | `memory_agent` | PostgreSQL + pgvector memory store |
| Orchestrator | 8001 (Docker) / 8003 (native) | `orchestrator` | LLM router + audio gateway + workflow coordinator |
| Elara | 8002 | `elara` | Conversation + affect-adaptive learning |
| Web Search Tool | 8010 | `web_search_tool` | MCP server wrapping Tavily API |
| Assistant Tool | 8011 | `assistant_tool` | MCP server for reminders + calendar (SQLite) |
| Perception Learner | 8012 | `perception_learner` | Cross-modal inference: perception × memory → EVENT memories |
| PostgreSQL | 5432 | `memory_db` | pgvector/pgvector:pg16 |
| Redis | 6379 | `redis` | Session state store (per-speaker Elara sessions) |
| Frontend | 3000 | `frontend` | Web UI |
| Perception Monitor | — | `perception_monitor` | Pulls frames from Pi, runs emotion+scene analysis, writes timeline.db |
| Eye (Raspberry Pi) | 8765 | *(on-device)* | Camera capture, face detection, pan-tilt servo tracking, identity |
| Pi Audio Client | — | *(on-device)* | Thin terminal: streams raw PCM to orchestrator /ws/audio, plays WAV back |

---

## Architecture

### Full data flow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━  VOICE CHANNEL (real-time)  ━━━━━━━━━━━━━━━━━━━━━━━━
Pi thin client  (pi_client/audio_client.py — runs on Raspberry Pi)
  PyAudio mic callback (16kHz, mono, int16, 512-frame chunks)
    ↓ raw PCM binary frames
    ↓ WebSocket /ws/audio → RunPod
  Audio Gateway  (agents/orchestrator/app/audio_ws.py)
  [AudioPipeline — loaded once at startup in background thread]
    ↓
  VADFilter (Silero VAD via torch.hub, float32 normalized)
    ↓ speech detected → phrase accumulation → silence timeout (0.8s)
  [parallel threads]
    ├── Transcriber (Faster-Whisper base.en, CPU int8)
    └── SpeakerManager (SpeechBrain ECAPA-TDNN, cosine similarity)
         └── embeddings persisted to /data/speaker_embeddings.json
    ↓ {"speaker": "User_1", "text": "...", "confidence": 0.92}
    ↓ internal call → handle_input()
  Orchestrator (:8001 Docker / :8003 native)
    ├── style frustration detection (Ollama)
    ├── LLM router → STORE_MEMORY | RETRIEVE_MEMORY | STORE_AND_RETRIEVE | USE_TOOL | DIRECT_CHAT
    ├── [if USE_TOOL] extract_tool_params (Ollama) → call_mcp_tool()
    │     ├── web_search_tool:8010  (Tavily API)
    │     └── assistant_tool:8011   (SQLite reminders + calendar)
    ├── [if memory action] memory_agent:8000
    │     ├── /process  → claim extraction (Ollama) → STATE/BELIEF/EVENT layers
    │     └── /retrieve → intent classification → pgvector semantic search
    └── elara:8002
          ├── NLP signals (VADER sentiment, Jaccard repetition, keyword patterns)
          ├── Affect classification → CALM | FRUSTRATED | CONFUSED | SAD | DISENGAGED
          ├── LinUCB contextual bandit (7 actions, per-user matrices on disk)
          ├── Config application (clarity, pace, patience, confirmation frequency)
          ├── LLM reply (Ollama/Groq streaming)
          ├── Distress watchdog (7 consecutive non-calm turns → caregiver alert)
          └── POST /tts → Kokoro TTS → 24kHz WAV bytes
                ↓ WAV sent back over WebSocket to Pi
  Pi plays WAV through speaker (_playing flag mutes mic during playback)

━━━━━━━━━━━━━━━━━━━━━━  VISION CHANNEL (passive, background)  ━━━━━━━━━━━━━━━━━
Face_login_Elara (Raspberry Pi 5)          RunPod
  picamera2 → Haar cascade (20fps)
  PID → ESP32 → pan-tilt servos             perception_monitor (Docker)
  dlib face encoding (every 3s)    ←HTTP→    polls GET /frames/event (every 60s)
  identity confirmed → /frames/event          ├── HSEmotion (EfficientNet-B0)
                                              │     → emotion + 8-class scores
                                              ├── Moondream VLM (Ollama)
                                              │     → scene description
                                              ├── Mistral LLM (Ollama)
                                              │     → structured JSON + causal reason
                                              └── timeline.db (SQLite, Docker volume)
                                                        ↓ (every 10 min)
                                              perception_learner (:8012)
                                                ├── reads new snapshots from timeline.db
                                                ├── calls memory_agent:8000/retrieve
                                                │     → relevant facts/beliefs/events
                                                ├── Mistral (Ollama) cross-modal reasoning
                                                │     snapshot + memory → inference sentence
                                                │     e.g. "User was happy while their tall
                                                │     son held a basketball"
                                                └── calls memory_agent:8000/process
                                                      → stored as EVENT memory
                                                        ↓
                                              memory_agent:8000  ←─────────────────────┐
                                              (PostgreSQL + pgvector)                   │
                                              STATE | BELIEF | EVENT layers             │
                                              ← read by Orchestrator on every turn ────┘
```

### Audio gateway detail (primary voice path)

```
pi_client/audio_client.py (Raspberry Pi)
  PyAudio mic → raw int16 PCM chunks (512 frames, 16kHz)
    ↓ WebSocket binary frames → ws://<pod>-8001.proxy.runpod.net/ws/audio
agents/orchestrator/app/audio_ws.py (RunPod)
  AudioPipeline.load() [once at startup, background thread]
    ├── Silero VAD (torch.hub)
    ├── Faster-Whisper "base.en" (CPU int8)
    └── SpeechBrain ECAPA-TDNN (cosine, threshold 0.25)
  handle_audio_ws() per connection:
    ├── VAD accumulates speech frames
    ├── Silence timeout (0.8s) → complete phrase
    ├── Parallel: STT thread + Speaker ID thread
    │     └── auto-register new speaker if audio ≥ 1.5s + no match
    ├── handle_input() → full orchestrator pipeline
    └── GET elara:8002/tts → Kokoro WAV → send back over WebSocket
  Speaker embeddings: /data/speaker_embeddings.json (persists restarts)
```

### MIC module detail (legacy — local processing path, still functional)

```
MicrophoneStream (PyAudio callback → Queue)
    ↓ raw bytes (int16)
VADFilter (Silero VAD via torch.hub)
    ↓ bool: is_speech
[buffer accumulation + silence timeout in VoiceAssistantCore.run()]
    ↓ complete phrase (raw bytes)
VoiceAssistantCore.process_audio()
    ├── Thread 1 → Transcriber (Faster-Whisper "base.en", CPU int8)
    └── Thread 2 → SpeakerManager (SpeechBrain ECAPA-TDNN, cosine similarity)
                       └── auto-registers new speakers if audio ≥ 1.5s
```

---

## Agent Details

### Pi Thin Client (`pi_client/`)
- Dependencies: only pyaudio, websockets, python-dotenv — no ML models on Pi
- Connects to `ORCHESTRATOR_WS_URL` (set in `.env`); RunPod proxy URL format: `ws://<pod-id>-8001.proxy.runpod.net/ws/audio`
- `_playing` threading.Event mutes mic while TTS WAV is playing (prevents echo)
- Max WebSocket message size: 10MB (handles long TTS responses)
- `run.sh` installs portaudio system dep, creates venv, installs deps, starts client

### Audio Gateway (`agents/orchestrator/app/audio_ws.py`)
- `AudioPipeline` singleton: loads Silero VAD + Faster-Whisper + SpeechBrain once, reused per connection
- Loaded in background daemon thread at orchestrator startup (non-blocking — health check passes immediately)
- `SILENCE_TIMEOUT = 0.8s`, `MIN_AUDIO_LEN = 0.5s`, `SAMPLE_RATE = 16kHz`
- Speaker embeddings persisted to `/data/speaker_embeddings.json` — survive pod restarts (not pod deletion)
- After pipeline runs, calls `elara:8002/tts` for Kokoro synthesis (24kHz WAV, best quality)

### MIC (`MIC/`) — legacy local processing path
- Audio: 16kHz, mono, int16 from PyAudio; converted to float32 normalized [-1, 1] before ML
- `SILENCE_TIMEOUT = 0.8s` — marks end of phrase
- `MIN_AUDIO_LEN = 0.5s` — minimum phrase length to run inference
- Speaker embeddings are **in-memory only** — lost on restart. Similarity threshold: 0.25
- Speaker IDs assigned as `User_1`, `User_2`, etc.
- GPU: change `device="cpu"` → `"cuda"`, `compute_type="int8"` → `"float16"` in `main.py`

### Memory Agent (`agents/memory/`)
- Three memory layers stored in PostgreSQL:
  - `STATE` — mutable facts with versioning (valid_from / valid_to)
  - `BELIEF` — subjective user opinions, also versioned
  - `EVENT` — immutable events with 768D embeddings (nomic-embed-text) for semantic search
- `/process` — context-aware claim extraction via Ollama (uses speaker name, existing memory, and recent turns to deduplicate and resolve pronouns); writes to appropriate layer
- `/retrieve` — classifies intent, queries relevant layer(s), assembles snapshot
- `/grounding/{speaker_id}` — returns high-importance permanent facts always carried in context
- `/debug/states`, `/debug/beliefs`, `/debug/logs`, `/debug/salience` — inspection endpoints

### Orchestrator (`agents/orchestrator/`)
- 5-way LLM router: `STORE_MEMORY | RETRIEVE_MEMORY | STORE_AND_RETRIEVE | USE_TOOL | DIRECT_CHAT`
- Router fallback (on unparseable LLM output) is `DIRECT_CHAT` — never `STORE_MEMORY`
- Pre-routing short-circuits before the LLM router runs:
  - Single-word affirmations (`ok`, `yes`, `sure`, `alright`, etc.) → always `DIRECT_CHAT`
  - Gratitude words (`thanks`, `thank you`, `cheers`) → bypass LLM entirely, return canned "You're welcome!" reply
- Tool execution uses `extract_tool_params()` (Ollama) to parse structured args from free-form text, then `call_mcp_tool()` to call the right MCP server
- List vs set intent for reminder/calendar is detected via keyword matching (no extra LLM call)
- Summarizes conversation every N turns (default 5) and stores as EVENT in memory
- Per-speaker session state stored in **Redis** (persists across container restarts within a deployment)
- Auth endpoints: `POST /auth/signup`, `POST /auth/login`, `POST /chat` (token-gated)
- **Audio WebSocket**: `GET /ws/audio` — Pi thin client connects here; full audio pipeline runs inside the orchestrator process
- **Location priority** in `handle_input()`: `req.metadata["location"]` (browser GPS) → memory grounding facts → ipinfo.io fallback (2s timeout)

### Elara (`agents/elara/`)
- Conversation adapter manages per-turn session state (history, config, bandit tracking)
- **LLM**: qwen2.5:32b via Ollama (current default on RunPod). Groq also supported (`backend=groq`). TTS: Kokoro (local) or Edge TTS (cloud)
- **NLP signals** (from `nlp_layer.py`): VADER sentiment, Jaccard repetition, confusion keywords, sadness keywords
- **Affect states**: CALM | FRUSTRATED | CONFUSED | SAD | DISENGAGED (priority order, checked top-down)
- **Escalation smoother**: 4 rules prevent sudden affect jumps — Rule R4 specifically fires on empty `affect_window` (first turns / post-greeting-reset) to prevent short messages like "Hi" from being classified as `disengaged`
- **Greeting reset** (`reset_history=True`): clears conversation history AND `bandit.affect_window` / previous bandit state so old emotional context never bleeds into a new session
- **Post-processing**: leading `Hello [Name]` greetings are stripped from replies after the first turn (Mistral habitually emits them despite prompt instructions)
- **LinUCB bandit** (7 actions × 7D features): DO_NOTHING, INCREASE_CLARITY, DECREASE_CLARITY, INCREASE_PACE, ENABLE_PATIENCE, DECREASE_CLARITY_AND_PACE, CLARITY_AND_CONFIRMATION
- Bandit matrices persisted per-user to `elara_bandit_tables` Docker volume
- **Distress watchdog**: 7 consecutive non-calm turns → `caregiver_alert: true` (SendGrid email via notifier)
- Max tokens per pace: slow=160, normal=100, fast=60

### Eye — Face_login_Elara (`Face_login_Elara-master/`, Raspberry Pi 5)
- **FastAPI server** on port 8765 — camera feed, face registration/login, tracking status
- **Two device modes**: `pi` (picamera2) and `laptop` (browser-injected frames via POST `/track/feed`)
- **FaceTracker** runs two daemon threads:
  - Main loop (20fps): Haar cascade face detection → PID error → ESP32 servo command
  - Recognition loop (every 3s): dlib face encoding → match against registered user → set `identified_present`
- **Identity**: faces registered as base64 frames via `/register`, encodings stored in `faces/db.json`
- **Servo**: ESP32 via USB serial (115200 baud), protocol `"P<pan> T<tilt>\n"`. Falls back to simulate mode if no ESP32
- **PID**: discrete controller with anti-windup + deadband (30px). Pan/tilt limits: 30°–150°
- **Key endpoints**: `GET /frames/event` (frame + identity + bounding box as JSON), `GET /track/stream` (MJPEG), `POST /login`
- Face encodings (`faces/db.json`) and timeline (`faces/timeline.db`) are in-process — not shared over the network

### Perception Monitor (`Face_login_Elara-master/monitor.py`, runs on RunPod)
- Polls `GET /frames/event` from the Pi every `MONITOR_PERIOD_S` seconds (default 60s)
- Only processes frames where an **identified** user is present (skips `Unknown`)
- Three-stage analysis pipeline:
  1. **HSEmotion** (EfficientNet-B0, ONNX) — face crop → 8-class emotion scores
  2. **Moondream VLM** (Ollama) — full frame → natural language scene description
  3. **Mistral LLM** (Ollama) — emotion + scene + previous snapshot → structured JSON with `emotion_affected_by_scene`, `reason`, `summary`
- Smart deduplication before writing: only records if emotion changed, affected flag flipped, or >50% subjects swapped (Jaccard < 0.5)
- Writes to `timeline.db` (SQLite, `perception_data` Docker volume) at `/app/faces/timeline.db`
- Env vars: `PI_URL`, `MONITOR_PERIOD_S`, `OLLAMA_BASE_URL`, `ANALYZER_VLM_MODEL`, `ANALYZER_LLM_MODEL`

### Perception Learner (`agents/perception_learner/`, port 8012)
- **Entirely passive** — never involved in the live conversation pipeline
- **Separate from Elara's learning system** (which handles user preferences from conversation). This agent specifically fuses *what the camera saw* with *what was said in conversation* to generate new memories
- **Poll loop** (every `POLL_INTERVAL_S` seconds, default 600): reads new rows from `timeline.db` since last processed ID
- **Per-snapshot reasoning**:
  1. Calls `memory_agent:8000/retrieve` with the snapshot's user + scene/subjects as query
  2. If no relevant memory found → SKIP (nothing to cross-reference)
  3. Sends snapshot + memory context to Mistral (Ollama) with a cross-modal reasoning prompt
  4. LLM either returns a grounded inference sentence or `SKIP`
  5. If inference: calls `memory_agent:8000/process` → stored as EVENT memory with `source: perception_learner` in metadata
- **State tracking**: own SQLite (`learner_state.db`) records every processed snapshot ID + the inference text (or null if SKIP) — guarantees each snapshot is reasoned over exactly once, survives restarts
- **Volumes**: reads `perception_data` (read-only at `/data/perception/timeline.db`), writes `learner_state` (at `/data/learner_state.db`)
- **Endpoints**: `GET /health`, `GET /status` (counts: total snapshots, processed, inferences generated, pending)
- The inferences land in the EVENT layer of the memory agent — indistinguishable from conversation-derived events, retrieved normally by the orchestrator

### MCP Tool Servers (`agents/tools/`)

**Web Search** (`agents/tools/web_search/`, port 8010)
- MCP tool: `search(query, max_results=5)`
- Calls Tavily API; requires `TAVILY_API_KEY` env var
- Returns Tavily's answer summary + top result excerpts

**Assistant Tools** (`agents/tools/assistant/`, port 8011)
- MCP tools: `set_reminder`, `list_reminders`, `complete_reminder`, `add_calendar_event`, `list_calendar_events`
- Backed by SQLite at `/data/assistant.db` inside Docker volume `assistant_data`
- Data persists across container restarts

---

## Key Configuration Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `SILENCE_TIMEOUT` | `audio_ws.py` / `MIC/main.py` | 0.8s | Silence → end of phrase |
| `MIN_AUDIO_LEN` | `audio_ws.py` / `MIC/main.py` | 0.5s | Minimum phrase length |
| `SIMILARITY_THRESHOLD` | `audio_ws.py` / `speaker_manager.py` | 0.25 | Speaker cosine match |
| `SUMMARIZE_EVERY_N_TURNS` | `orchestrator/config.py` | 5 | Conversation summarization |
| `DISTRESS_TURN_LIMIT` | `elara/adapter.py` | 7 | Consecutive non-calm before alert |
| `BANDIT_GAMMA` | `elara/bandit.py` | 0.95 | Discount factor |
| `BANDIT_ALPHA` | `elara/bandit.py` | 0.8 | UCB exploration coefficient |
| `MAX_HISTORY_TURNS` | `elara/adapter.py` | 10 | Conversation window kept (20 turns stored, last 10×2 sent to LLM) |
| `DEAD_ZONE_PX` | `Face_login_Elara-master/config.py` | 30px | Servo ignores face offsets smaller than this |
| `IDENTITY_CHECK_INTERVAL_S` | `Face_login_Elara-master/config.py` | 3.0s | How often dlib re-confirms identity |
| `IDENTITY_TOLERANCE` | `Face_login_Elara-master/config.py` | 0.50 | Face match threshold (lower = stricter) |
| `MONITOR_PERIOD_S` | env var | 60s | How often perception_monitor polls the Pi |
| `POLL_INTERVAL_S` | env var | 600s | How often perception_learner checks for new snapshots |

---

## Dependencies & Environment

### MIC module
Virtual environment at `MIC/.venv/`. Models downloaded on first run:
- Silero VAD → Torch Hub cache
- SpeechBrain ECAPA-TDNN → `MIC/tmp_model/`
- Faster-Whisper → Hugging Face cache

### Docker services
Each agent has its own `Dockerfile` and `requirements.txt`. The orchestrator additionally requires `mcp>=1.3.0`, `httpx>=0.27.0` for MCP client calls, and `torch`, `torchaudio`, `faster-whisper`, `speechbrain`, `websockets` for the audio gateway.

### Environment variables (`.env` in project root)
```
TAVILY_API_KEY=tvly-...       # Required for web search

# Vision channel
PI_URL=http://<pi-ip>:8765    # Raspberry Pi address for perception_monitor
MONITOR_PERIOD_S=60           # How often to pull a frame from the Pi (seconds)
POLL_INTERVAL_S=600           # How often perception_learner checks for new snapshots
ANALYZER_VLM_MODEL=moondream  # VLM for scene description (Ollama model name)
ANALYZER_LLM_MODEL=mistral:latest  # LLM for perception reasoning
```
Docker Compose passes `TAVILY_API_KEY` to `web_search_tool` and `orchestrator`. Vision vars are passed to `perception_monitor` and `perception_learner`.

---

## Known Gaps / Future Work

- **DB table ownership (CRITICAL)**: `init.sql` creates tables owned by `postgres` superuser; memory agent runs as `memory_user`. PostgreSQL ownership (not just privileges) is required for ALTER TABLE — so all startup migrations fail on a fresh pod, leaving `importance`, `speaker_id`, `is_grounding` columns missing and all memory writes silently broken. Fix: add `OWNER memory_user` to every `CREATE TABLE` in `init.sql`, OR add `ALTER TABLE ... OWNER TO memory_user` in `start.sh` after init.sql runs.
- **PostgreSQL persistence**: DB lives on pod-local disk — wiped on pod deletion. Mount RunPod Network Volume at `/var/lib/postgresql/data` before accumulating real user data.
- **Speaker embeddings persistence**: `/data/speaker_embeddings.json` survives restarts but not pod deletion. Include in Network Volume mount.
- **Speaker ID persistence (MIC legacy path)**: `known_speakers` in MIC/ is in-memory; lost on restart
- **Session state persistence**: now stored in Redis — survives container restarts but not full stack teardown (`docker compose down -v` wipes it)
- **Distress escalation**: `send_caregiver_alert()` in `notifier.py` is wired up; requires SendGrid credentials in env
- **Health monitor tool**: registered in tool registry but not connected to any sensor data source
- **npm not in PATH**: `start.sh` frontend section silently fails on fresh RunPod pods if Node.js isn't installed. Fix: auto-detect and install Node.js 20 via nodesource, or document the manual step.
- **Audio WebSocket not yet hardware-tested**: `pi_client/` code is written but end-to-end test with real Pi hardware hasn't been done.
- **Orchestrator port mismatch**: Docker exposes 8001, native `start.sh` uses 8003. Pi client `.env.example` documents the Docker (RunPod proxy) port.
- **Authentication**: `/auth/signup` + `/auth/login` endpoints exist; `speaker_id` on the legacy `/input` endpoint is still client-provided with no validation
- **Bandit cold start**: first ~20 turns do exploration; no population warm-start
- **LLM quality ceiling**: qwen2.5:32b (current RunPod default, 20GB). llama3.3:70b would be better but requires ~43GB disk. Groq API is the easiest path to a larger model without local disk constraints.
- **Eye face encodings persistence**: `faces/db.json` on the Pi is not backed up — re-registration needed if Pi is wiped
- **Perception learner cold start**: if perception snapshots accumulate before the learner runs, all will be processed in bulk on first poll — no throttle or rate limiting on bulk catch-up
- **Cross-user inference isolation**: perception_learner reasons per snapshot's `user` field, but if multiple users are registered on the Pi, each is handled independently — no cross-user inference is ever made (intentional by design)
