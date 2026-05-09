# E.L.A.R.A. — Deep System Analysis

**Date:** 2026-05-09
**Scope:** No code changed. Findings only.
**Method:** Four parallel exploration agents (core triad, vision, infra/tools, cross-cutting bugs) + direct verification of high-risk claims by reading source.
**Source of truth precedence used here:** Code > `SYSTEM_OVERVIEW.txt` > `CLAUDE.md` (the latter is in `.gitignore` and has drifted hardest).

---

## How to use this doc in the next session

If you are a future Claude instance picking this up cold: read this file end-to-end and you have enough to continue without re-running the four exploration agents. You do **not** need to re-read every Python file. The minimum bootstrap is:

1. § 1 (what's actually running, with verified file:line) — your ground truth.
2. § 2 (CLAUDE.md vs reality) — anything CLAUDE.md asserts that contradicts here, trust here.
3. § 3 (bugs B1–B30) — every entry is independently actionable.
4. § 4 (upgrades U1–U22) — every entry is tagged `[Quick win]`, `[Realistic]`, or `[Ambitious]`. Default to picking from Quick win and Realistic; only pull an Ambitious item if the user asks for it by number.
5. § 5 (conflicts) — read before applying any two upgrades together.
6. § 6 (sequenced roadmap) — if the user says "just start", start at item 1.

Re-verify with `git log --since=<date-of-this-doc>` before acting; the codebase moves.

---

## 0. Executive Summary

E.L.A.R.A. is functionally rich (voice + vision + memory + bandit + tools + curiosity) but documentation has decoupled from code, and several deploy-blocking conflicts hide in the seams between Docker / native / docs / tests. Nothing here is "broken in production" today, because production is a single-user RunPod, but the system would not survive a multi-tenant or internet-facing deployment without ~1–2 days of hardening.

The three biggest issues are:

1. **Port disagreement** between Docker Compose (`orchestrator: 8001`) and everything else that talks to the orchestrator (`start.sh`, `test_chat.py`, `test_rigorous.py`, `frontend/Dockerfile build env`, `CLAUDE.md` → all `8003`). Stack-via-Docker silently sends the frontend to a dead port.
2. **Documentation drift** is severe. CLAUDE.md describes a 7-action × 7-D bandit, qwen2.5:32b default, a SendGrid notifier, and a 4-rule escalation smoother. None of those are true: the bandit is 19 × 14, four different services default to four different Ollama models (`mistral`, `mistral:latest`, `qwen2.5:1.5b`), the notifier is Gmail SMTP, and there are 5 escalation rules (R5 was added).
3. **Auth/security holes** that are *known* but live: `/input` accepts any `speaker_id`, every `/debug/*` on the memory agent is unauthenticated, `/grounding/{speaker_id}` is enumerable, JWT signing falls back to a literal `"elara-dev-secret-change-in-prod"` default, CORS is `*` everywhere.

Nothing on this list requires architectural change — they're all surgical fixes. The system itself is solid; the docs and the perimeter are not.

---

## 1. What's actually running (verified)

### 1.1 Service map (verified against code)

| Service | Docker port | start.sh port | Comments |
|---|---|---|---|
| memory_agent | 8000 | 8000 | ✓ aligned |
| orchestrator | **8001** | **8003** | **MISMATCH** |
| elara | 8002 | 8002 | ✓ aligned |
| web_search_tool | 8010 | 8010 | ✓ aligned |
| assistant_tool | 8011 | 8011 | ✓ aligned |
| perception_learner | 8012 | 8012 | ✓ aligned |
| perception_monitor | (no port) | — | background poller |
| Eye / face_login | 8765 (on Pi, off-stack) | — | not in compose |
| frontend | 3000 | 3000 | builds with `BACKEND_URL=http://orchestrator:8001` (compose) vs `http://localhost:8003` (start.sh) — **BACKEND_URL is wired into a static next build, so changing it requires rebuilding the frontend image** |
| postgres (pgvector) | 5432 | 5432 | ✓ |
| redis | 6379 | 6379 | ✓ |
| ollama | 11434 | 11434 | ✓ |

Evidence:
- `docker-compose.yml:122-124` → orchestrator publishes `8001:8001`, frontend has `BACKEND_URL: http://orchestrator:8001` at `:157`.
- `start.sh:199-213` → orchestrator is started on `8003`, frontend build env is `BACKEND_URL=http://localhost:8003`.
- `test_chat.py:13` → `ORCHESTRATOR = "http://localhost:8003"`.
- `test_rigorous.py:11` → `ORCH = "http://localhost:8003"`.

### 1.2 Endpoints, by service

**Orchestrator** (`agents/orchestrator/app/main.py`):
- `POST /auth/signup` (:120) — bcrypt + JWT, seeds profile to memory_agent
- `POST /auth/login` (:144)
- `POST /chat` (:158) — token-gated wrapper around the pipeline
- `POST /input` (:235) — **legacy, no auth, accepts client-supplied `speaker`** (CLAUDE.md gap is real)
- `GET /get_profile/{user_id}` (:190) — **no auth**
- `GET /get_memories/{user_id}` (:198) — **no auth**, returns full memory snapshot for any user_id
- `GET /health` (:560)
- `POST /sync-otp` (:637) — Pi → orchestrator (API-key gated, line 638 verifies `x_api_key`)
- `POST /verify-otp` (:651) — frontend → orchestrator (rate-limited 5/min via slowapi, :652)
- `GET /connection-status` (:673)

**Memory Agent** (`agents/memory/app/main.py`):
- `POST /process` (:107), `POST /retrieve` (:188), `POST /episode` (:214), `POST /recall` (:227)
- `GET /grounding/{speaker_id}` (:249) — **no auth**, returns importance≥0.85 permanent facts
- `GET /health` (:263)
- `GET /debug/states` (:274), `/debug/beliefs` (:295), `/debug/logs` (:317), `/debug/salience` (:351) — **all unauthenticated**
- `DELETE /debug/states/purge` (:330) — **destructive, unauthenticated**

**Elara** (`agents/elara/app.py`):
- `POST /chat` (:269), `POST /chat/stream` (:275)
- `POST /analyse` (:264) — internal-only; never wired to a client (dead from outside)
- `POST /tts` (:253), `GET /` (:243), `GET /health` (:248)

**Eye / face_login_Elara** (`Face_login_Elara-master/main.py`, runs on Pi):
- 20 endpoints across `/register`, `/login`, `/track/*`, `/frames/*`, `/camera/*`, `/debug/last-frame` — all unauthenticated, all CORS `*` (`:47-50`)

**MCP Tool Servers** (`agents/tools/{web_search,assistant}/server.py`):
- POST `/call/<tool_name>` style — plain HTTP, NOT actual MCP SSE/stdio. `call_mcp_tool()` in `agents/orchestrator/app/tool_client.py` is just `httpx.post("/call/<tool>", json=params)` with a 20 s timeout.

### 1.3 Where each LLM call lives (the full inventory)

| # | Caller | Model env var | Default model | Prompt location | Fallback on failure |
|---|---|---|---|---|---|
| 1 | Orchestrator router | `OLLAMA_MODEL` | `mistral:latest` (config.py:11) | `agents/orchestrator/app/agents.py:106-203` | `DIRECT_CHAT` (:264) ✓ |
| 2 | Style frustration check | same | same | `agents.py:273-276` | returns False (:302) |
| 3 | Memory query rewrite | same | same | `agents.py:307-317` | original question (:337) |
| 4 | Tool-param extraction | same | same | `agents.py:47-65` | `{"query": text}` (:99) |
| 5 | Claim extraction | same | `mistral` (memory/app/config.py:9) | `memory/app/extractor.py:14-60` | empty claims (:198) |
| 6 | Intent classification | same | same | `extractor.py:62-74` | `GENERAL` (:211) |
| 7 | Relevance reranking | same | same | `extractor.py:76-86` | all-1.0 scores (:247) — **dead code: `llm_rerank=False` is hard-coded everywhere** |
| 8 | Conversation summary | same | same | `agents.py:368-375` | silently skip (:426) |
| 9 | Curiosity generator | (env) | `qwen2.5:1.5b` (elara/conversation_agent/llm.py:23) | `agents/elara/curiosity_agent/generator.py` | none — empty queue |
| 10 | NLP signal extractor | same | `qwen2.5:1.5b` (nlp_layer.py:26) | `nlp_layer.py:55-78` | VADER + keyword scan (:171-192) |
| 11 | Main Elara reply | same | `qwen2.5:1.5b` | `rag.py: build_persona_prompt` | "I'm sorry, I'm having trouble..." (:358) |
| 12 | Perception scene VLM | `ANALYZER_VLM_MODEL` | `moondream` | `Face_login_Elara-master/analyzer.py:188-194` | `_empty()` |
| 13 | Perception structurer | `ANALYZER_LLM_MODEL` | `mistral:latest` | `analyzer.py:203-251` | `_empty()` |
| 14 | Perception cross-modal | `OLLAMA_MODEL` | `mistral:latest` (perception_learner/reasoner.py:20) | `reasoner.py:122-140` | mark snapshot processed with null inference |

**Default-model drift across services (CRITICAL doc problem):**

```
memory_agent          → mistral
orchestrator          → mistral:latest
elara conversation    → qwen2.5:1.5b
elara nlp_layer       → qwen2.5:1.5b
perception_learner    → mistral:latest
perception_monitor    → mistral:latest (LLM) + moondream (VLM)
```

Docker Compose passes `${OLLAMA_MODEL:-mistral:latest}` to memory_agent, elara, orchestrator, and perception_learner — so in Docker mode, **everything is unified to `mistral:latest`**. In native mode (start.sh), the env var is not set, so each service uses *its own default*, which means **elara silently runs a 1.5 B model while everything else runs Mistral 7 B**. CLAUDE.md says the default is `qwen2.5:32b` — wrong everywhere.

`start.sh` actually pulls `qwen2.5:32b` and `nomic-embed-text` (`start.sh:76,83`) but never sets `OLLAMA_MODEL`, so the pulled 32 B model is never selected by any agent.

---

## 2. Documentation conflicts (CLAUDE.md vs reality)

These are statements in `CLAUDE.md` that the code contradicts. (Note: `CLAUDE.md` is itself in `.gitignore` — it's a private dev artifact, which is why it has rotted.)

| # | CLAUDE.md says | Reality | Evidence |
|---|---|---|---|
| 1 | Orchestrator on port 8003 | Docker → 8001, native → 8003 | `docker-compose.yml:122` |
| 2 | "qwen2.5:32b is current default on RunPod" | Four different defaults across services; never 32b | see § 1.3 |
| 3 | "LinUCB bandit (7 actions × 7D features)" | **19 actions × 14D** (5-D affect one-hot + 9-D personality) | `agents/elara/learning_agent/storage.py:25-26`, `bandit.py:35`, `state_classifier.py:57` |
| 4 | bandit `BANDIT_GAMMA = 0.95`, `BANDIT_ALPHA = 0.8` | Code defaults to `alpha=1.0, gamma=0.99` | `agents/elara/learning_agent/bandit.py:54` |
| 5 | "Escalation smoother: 4 rules" | **5 rules** — R5 added 2026-04-29 (low-conf sad after calm → calm) | `state_classifier.py:202-207` |
| 6 | Affect signals = "VADER, Jaccard, confusion keywords, sadness keywords" | Now an **LLM call** with VADER+keyword as fallback only | `nlp_layer.py:83-140` |
| 7 | Caregiver alert via SendGrid | Gmail SMTP + app password | `agents/elara/conversation_agent/notifier.py:20,29,63-67` |
| 8 | Personality is implicit / undocumented | **9-D PersonalityVector** with EMA + bandit + context gate | `personality.py:17-29` |
| 9 | (no mention of curiosity) | Full curiosity agent: generator + injector + EMA receptiveness | `agents/elara/curiosity_agent/` |
| 10 | (no mention of OTP face-login bridge) | `/sync-otp`, `/verify-otp`, `/connection-status` exist; Pi pushes OTP, frontend verifies | `agents/orchestrator/app/main.py:637-690` |
| 11 | "leading 'Hello [Name]' stripped after first turn" | ✓ verified | `adapter.py:367-373` |
| 12 | "20 turns stored, last 10×2 sent to LLM" | ✓ verified | `adapter.py:231-232,346` |
| 13 | "Max tokens: slow=160, normal=100, fast=60" | ✓ verified | `adapter.py:33` |
| 14 | "Router fallback is DIRECT_CHAT, never STORE_MEMORY" | ✓ verified | `agents.py:264` |
| 15 | "Redis persists across container restarts" | ✓ (compose has named volume `redis_data`, `appendonly yes`) | `docker-compose.yml:2-14` |
| 16 | timeline.db at `/app/faces/timeline.db` | Path is **relative** (`Path("faces/timeline.db")`); resolves to `/app/faces/timeline.db` only because `WORKDIR /app` in Dockerfile.monitor — fine in Docker, fragile if WORKDIR changes | `Face_login_Elara-master/timeline.py:24` |
| 17 | "speaker_id is added by migrations" — **NOT documented anywhere** | `init.sql` does NOT have `speaker_id`; it's added by `main.py:35-39`. Old DBs that skipped the migration would have data without speaker_id and silently default everything to `'user'`, leaking memory across speakers. | `agents/memory/sql/init.sql:20-33` vs `main.py:35-70` |

CLAUDE.md is also internally inconsistent with `SYSTEM_OVERVIEW.txt` (the older but more current changelog): SYSTEM_OVERVIEW says `qwen2.5:32b` is pulled by start.sh and `Mistral 7B` is the actual conversation LLM; CLAUDE.md just says `qwen2.5:32b is current default`.

---

## 3. Bugs and concrete defects

Ordered roughly by severity × likelihood.

### 3.1 Critical (security / data integrity)

**B1 — `/input` accepts any speaker_id without auth.**
`agents/orchestrator/app/main.py:235`. Anyone reachable on the orchestrator port can inject memories for any user, retrieve memories of any user, and trigger LLM calls. CLAUDE.md acknowledges this as a "known gap"; it is still live.

**B2 — All `/debug/*` endpoints are open.**
`agents/memory/app/main.py:274-373`. `/debug/logs` returns raw user transcripts. `/debug/states/purge` is a destructive DELETE. No auth, no IP restriction, no env-flag gate.

**B3 — `/grounding/{speaker_id}` is enumerable and returns sensitive permanent facts.**
`agents/memory/app/main.py:249`. Caller can iterate `User_1`, `User_2`, … and read every user's high-importance permanent facts (deceased relatives, medical conditions).

**B4 — `/get_profile/{user_id}` and `/get_memories/{user_id}` are unauthenticated.**
`agents/orchestrator/app/main.py:190,198`. Same enumerability problem.

**B5 — JWT secret falls back to a literal hardcoded string.**
`agents/orchestrator/app/auth.py:25` → `os.environ.get("JWT_SECRET_KEY", "elara-dev-secret-change-in-prod")`. If the env var is missing — which is silent — every signed token is forgeable by anyone reading the code.

**B6 — CORS is `*` on every service that has CORS middleware.**
`orchestrator/main.py:34-38`, `memory/main.py:98-101`, `elara/app.py:49-52`, web_search/server.py:14, assistant/server.py:32, Face_login/main.py:47-50, perception_learner/main.py:176-179. Combined with B1–B4, this enables browser-based CSRF from any tab on the user's machine.

**B7 — `init.sql` ships a schema without `speaker_id`; it's only added by Python migration code at startup.**
If the DB is initialized once and the migration code path is later removed, or if a fresh DB is started against an older `main.py`, `speaker_id` defaults to `'user'` and all memory cross-pollutes between speakers. This is a footgun for any greenfield deploy.

### 3.2 High (functional / latency)

**B8 — `requests.get("https://ipinfo.io/json")` has no timeout, runs on EVERY turn.**
`agents/orchestrator/app/main.py:258`. If ipinfo is slow or blocked, the entire `/input` request hangs indefinitely. Also adds 200–500 ms to every turn even when fast.
Fix: `timeout=2`, cache result in Redis with 1 h TTL keyed by speaker_id.

**B9 — Frontend BACKEND_URL is baked at build time, then disagrees with Docker port.**
`docker-compose.yml:157` builds the frontend image with `BACKEND_URL=http://orchestrator:8001`. When you change the port to 8003 (to match start.sh and tests), the frontend image must be rebuilt. There is no warning if the orchestrator listens on a different port — the frontend will silently fail every request.

**B10 — LLM relevance reranker is fully implemented but never invoked.**
`agents/memory/app/extractor.py:214-247` defines `score_relevance()`, and `assemble_snapshot()` accepts `llm_rerank: bool` — but every caller passes `False`. Either delete the dead code (~35 lines) or wire it into the `/retrieve` path.

**B11 — `/track/feed` (laptop-mode injection) accepts unbounded base64 frames with no size cap.**
`Face_login_Elara-master/main.py:270-279`. A malicious browser tab (CORS = `*`) could POST a 100 MB image and exhaust memory.

**B12 — Perception monitor's `box` field is requested by `monitor.py` but never set by Eye's `/frames/event`.**
`monitor.py:104` reads `box = event.get("box")`; `main.py:473-517` never includes one. So HSEmotion always falls back to centre-crop (`analyzer.py:79-82`), which materially degrades emotion accuracy when the user is off-centre.

**B13 — `extract_tool_params` always queries `ipinfo.io` then concatenates the result into the LLM prompt.**
Same hang as B8, doubled — and the location string is built from data the system already has in memory (the user's stored location). Two HTTP calls per tool turn for a value already known.

**B14 — Concurrent `/process` calls on the same `entity.attribute` race.**
`agents/memory/app/memory.py:70-86` reads existing rows, then writes a new row + `valid_to` on the old. No transaction around the read+write, no `SELECT ... FOR UPDATE`. Two simultaneous extractions of the same fact create overlapping rows. Low likelihood today (single voice channel, sequential), high likelihood once vision-channel inferences and voice both flow into `/process`.

**B15 — `next.config.ts` silences both TypeScript errors and ESLint during build.**
`frontend/next.config.ts:6-10` → `eslint.ignoreDuringBuilds: true`, `typescript.ignoreBuildErrors: true`. The frontend is shipping with whatever type errors and lint violations exist; CI would not catch them because there is no CI.

### 3.3 Medium

**B16 — Bandit α/γ default mismatch.**
`bandit.py:54` defaults to `alpha=1.0, gamma=0.99`. Both CLAUDE.md and SYSTEM_OVERVIEW.txt say the project uses `0.8 / 0.95`. Whoever instantiates the bandit may or may not be passing the correct values; this is worth tracing.

**B17 — `/auth/signup` and `/auth/login` are not rate-limited.**
Only `/verify-otp` (`@limiter.limit("5/minute")` at `:652`) is rate-limited. Trivial to brute-force email enumeration / passwords.

**B18 — Bandit table per-user `fcntl.flock` has no timeout.**
`agents/elara/learning_agent/storage.py:51-67`. Concurrent requests for the same user from two workers will block forever. Single-worker today, fine; multi-worker, deadlock risk.

**B19 — Perception learner snapshot watermark is read-then-write without an atomic CAS.**
Two concurrent poll cycles (e.g. crash + supervisor restart races) can both fetch the same batch and process them twice. `INSERT OR IGNORE` (`main.py:59`) prevents corruption but not redundant LLM spend.

**B20 — Cross-modal events from perception_learner are indistinguishable from voice events at retrieval time.**
Perception inferences are stored as `EVENT` with `metadata.source: "perception_learner"`, but `/retrieve` does not surface metadata. If both channels infer the same fact, it is stored twice with no dedup — orchestrator will then surface both to Elara verbatim.

**B21 — `health_monitor` is in `TOOL_REGISTRY` but has no implementation.**
`agents/orchestrator/app/agents.py:32-37`. If the LLM router ever picks `health_monitor` (intent: "I feel dizzy"), `call_mcp_tool` will hit a non-existent server and return an error string straight to Elara. CLAUDE.md acknowledges this; the registry entry should be removed until the tool exists.

**B22 — Tool calls have no idempotency.**
A retry of the orchestrator after a network blip will create duplicate reminders / duplicate calendar events / duplicate Tavily searches.

**B23 — Speaker ID drift across restarts.**
MIC's speaker store is in-memory (`MIC/speaker_manager.py`). Memories written under `User_1` today are orphaned if `User_1` is re-registered as `User_2` after a restart. Also creates an opportunity for cross-user memory bleed if speaker IDs are reused.

### 3.4 Low / cosmetic

**B24 — `print()` in production paths.** Many LLM error paths use `print()` rather than `logging.warning()`. No log levels, no structured JSON, no shipping.

**B25 — `agents/tools/*/.venv/` and `agents/perception_learner/.venv/` show up as untracked.**
`.gitignore` covers `agents/*/.venv/` but the tool venvs live a level deeper (`agents/tools/<name>/.venv`) and are not ignored. Add `agents/tools/*/.venv/` to `.gitignore`.

**B26 — `data/` directory now holds host-mounted volume contents (`assistant`, `learner`, `perception`, `redis`).**
Git status lists `data/` as untracked. The dir is ~11 MB of runtime data and should be in `.gitignore`.

**B27 — `elara_web/` (`auth.html`, `dashboard.html`) is a stale legacy frontend; replaced by Next.js `frontend/`.**
Either delete or move to `archive/` to avoid confusion for new contributors.

**B28 — `MAX_HISTORY_TURNS` doc says "10 turns kept (20 stored, last 10×2 sent)".**
The arithmetic is verified, but the wording has caused confusion. Suggest renaming the constant to `HISTORY_CAP_PAIRS`.

**B29 — Recognition worker comment is wrong.**
`tracker.py:578` says "skip dlib's slow HOG step" but Haar locations are passed in directly, so HOG is already bypassed; `num_jitters=0` skips re-encoding, not HOG.

**B30 — `Face_login_Elara-master/main.py` is a single 600+ line file that mixes camera control, identity, OTP bridge, debug endpoints, and the FastAPI app.** Refactor candidate; not urgent.

---

## 4. Architectural opportunities (upgrades worth considering)

These are deliberate "what could be better" calls — not bug fixes.

**Ambition legend:** `[Quick win]` ≈ < 1 day, contained blast radius. `[Realistic]` ≈ 1–3 days, single subsystem. `[Ambitious]` ≈ multi-day, cross-cutting, often needs new infra or operational changes — pursue only if the user asks for it by number.

### 4.1 Routing & orchestration

**U1 [Ambitious] — Collapse the regex short-circuit layer behind a stronger LLM.**
There are now ~15 hard-coded regexes (`_GRATITUDE_RE`, `_AFFIRMATION_RE`, `_GREETING_RE`, `_EXPLICIT_SEARCH_RE`, `_REMINDER_SET_RE`, `_REMINDER_LIST_RE`, `_TOOL_CORRECTION_RE`, `_IMPLICIT_SEARCH_RE`, `_ELARA_COMPLAINT_RE`, `_MEMORY_QUESTION_RE`, `_ELARA_OPINION_RE`, `_APOLOGY_RE`, `_SOCIAL_POSITIVE_RE`, `_STYLE_FEEDBACK_RE`, plus `_GRIEF_SENTENCE_RE` post-filter). Every one of them exists because Mistral 7B can't be trusted with multi-conditional instructions. SYSTEM_OVERVIEW.txt:106-115 already lists this as the planned upgrade. Concretely:
- Move router + reply LLM to Llama-3.1-70B-Instruct on Groq (or Claude Haiku 4.5 via API).
- Keep the short-circuit regexes as a *safety net*, not a primary path.
- Remove the grief sentence-level post-filter once the LLM is reliable.
- Removing redundancy gains ~200 lines of code and 1–2 LLM calls per turn.

**U2 [Realistic] — Merge style-frustration check + NLP signal extraction + router into ONE LLM call.**
Today three separate Ollama calls run sequentially per turn (`agents.py:282`, `agents.py:230`, plus one in `nlp_layer.py:96`). A single structured-output call returns all three signals + the router decision. Saves 600–1500 ms per turn.

**U3 [Ambitious] — Cache the router prompt template and the persona prompt via Anthropic prompt caching.**
`ROUTER_PROMPT` is ~100 lines. `build_persona_prompt` injects ~30 lines of persona + style directives every turn. If we move to the Claude API, prompt caching takes both to ~0 cost on the cache-hit side.

**U4 [Ambitious] — Make the bandit warm-start from a population prior.**
Cold-start (the first ~20 turns) is currently uniform exploration. Save the average A/b across all existing users and warm-start new users from that prior. Cuts time-to-stable-personality by ~5×.

**U5 [Quick win] — Tool-call idempotency.**
Add a content-hash idempotency key on every `/call/<tool>` invocation; reminders/events store the key, and a duplicate call is a no-op. Eliminates the duplicate-reminder failure mode entirely.

### 4.2 Memory

**U6 [Realistic] — Run salience + similarity in the same query.**
Today salience is computed in Python after a wide `SELECT` (`memory.py:216-222`). Push it into SQL via a generated column or materialized view; index `(speaker_id, salience DESC)`. At ~10⁵ rows you'll feel this; at 10⁶ you'll need it.

**U7 [Quick win] — Add `(speaker_id, valid_to)` compound indexes on `state_memory` and `belief_memory`.**
Today only `(entity, attribute)` is indexed. `WHERE speaker_id = $1 AND valid_to IS NULL` does an index scan plus filter; a partial index keyed on speaker_id wins.

**U8 [Realistic] — Distinguish perception-derived events from voice-derived events at retrieval.**
Either add `metadata.source` to the `/retrieve` payload, or split into two layers (`event_voice`, `event_perception`). Lets Elara modulate trust ("we *saw* you near photographs" vs "you *said* you were sad").

**U9 [Realistic] — Use embeddings for the "implicit search" classification.**
The `_IMPLICIT_SEARCH_RE` regex catches "I want a coffee" but misses "could murder a coffee" or "fancy a brew". Train a tiny intent classifier on labelled examples or — cheaper — embed the message and compare against a small set of seed phrases in pgvector. Drop the regex once recall is comparable.

### 4.3 Vision

**U10 [Ambitious] — Run HSEmotion on the Pi, send only the 8-class scores to RunPod.**
HSEmotion is a 5 MB ONNX, easily Pi-5 capable. Network drops to a few hundred bytes per snapshot; Moondream and Mistral run on RunPod as today.

**U11 [Quick win] — Have the Eye populate the `box` field in `/frames/event`.**
Already detected by Haar in the tracker; just include `box: {x,y,w,h}`. Fixes B12 and meaningfully improves emotion accuracy.

**U12 [Realistic] — Add an importance score to perception snapshots.**
A snapshot of the user smiling on a normal afternoon is low-signal. A snapshot showing emotion change + new subject is high-signal. Score it cheaply (emotion variance + Jaccard distance from previous + affected flag), have the perception_learner prioritise high-importance snapshots and *down-sample* low-importance ones during catch-up.

**U13 [Quick win] — Cross-modal prompt: ask the LLM for *disconfirming* evidence too.**
Today the prompt asks for a confirming inference. Adding "if the snapshot disconfirms a stored belief, say so" turns the learner into a *belief revisor*, not just a generator.

### 4.4 Frontend & Auth

**U14 [Quick win] — Move BACKEND_URL from build-time to runtime via a `/api/config` endpoint or `NEXT_PUBLIC_*` runtime env.**
Eliminates the rebuild-required-on-port-change footgun (B9).

**U15 [Realistic] — Add proper auth middleware to `/input`, `/get_profile/*`, `/get_memories/*`, all `/debug/*`, and `/grounding/*`.**
B1–B4 all collapse into one fix: a `Depends(get_current_user)` on every non-`/auth/*` endpoint, with the speaker_id bound to the JWT subject.

**U16 [Quick win] — JWT secret: fail-fast, no default.**
`JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]` (raises KeyError if missing). Add a docker-compose check that fails the orchestrator container at startup if the var is empty.

**U17 [Ambitious] — Per-user encryption-at-rest of bandit tables.**
Today they're plain `.npy` files in a Docker volume. If the volume is exfiltrated, behavioural fingerprints leak. Encrypt with a key derived from the user's password (PBKDF2) and reload on auth.

### 4.5 Operations

**U18 [Realistic] — Real CI.**
There is no `.github/workflows/`, no `.gitlab-ci.yml`. `run_tests.py` (43-check live-stack suite) and `test_rigorous.py` (3×5 scripts × ~20 turns) exist but are run manually. Wire them into a nightly job against a docker-compose'd test stack.

**U19 [Realistic] — Structured logging + log shipping.**
Replace every `print()` in agent code with `log.info/warning/error`. JSON formatter. Aggregate with Loki / OpenSearch / whatever — but *one* place to grep when something breaks.

**U20 [Quick win] — Health checks for every service in docker-compose.**
Today only redis, db, ollama have healthchecks. memory_agent / orchestrator / elara have none → `depends_on` only enforces start order, not readiness.

**U21 [Quick win] — Restart policies.**
No service has `restart: unless-stopped`. Any crash = manual `docker compose up`.

**U22 [Quick win] — Resource limits.**
No CPU/memory caps on any non-Ollama service. One runaway agent OOMs the pod.

---

## 5. Conflicts between recommendations (think before you act)

These are places where two of the suggestions above would step on each other and you have to pick.

**C1 — U1 (bigger LLM) vs U2 (merge LLM calls).**
If you adopt a 70 B model on Groq, latency per call goes up, so merging three calls into one becomes more attractive — but the merge prompt becomes harder to engineer because the model has to emit a strict JSON structure with router action + style flag + NLP signals. Pick: (a) keep three small calls on Mistral, *or* (b) merge into one call against a more capable model. Doing both is a wash: a single call against Mistral often fails JSON parsing; three calls against Llama-70B is 3× the cost.

**C2 — U7 (compound indexes on memory) vs U6 (salience as SQL).**
Both touch the same hot tables. Plan them in the same migration so you don't reindex twice.

**C3 — U10 (HSEmotion on Pi) vs U12 (importance scoring).**
If HSEmotion runs on the Pi and only scores are sent, the importance scorer also has to run on the Pi (it needs the emotion). Either both move, or both stay.

**C4 — U15 (auth on /input) vs MIC's current contract.**
The MIC module today posts to `/input` with no token. Adding auth means the MIC module needs to authenticate first. If the MIC runs on the same trusted machine as the orchestrator, the cleanest path is a static API key in the `.env`, not a JWT (no login UX in a voice client).

**C5 — Removing dead code (B10, B21, B27) vs keeping safety nets.**
The relevance reranker (`score_relevance`) is "dead" but high-signal — it's the place to plug in better re-ranking if the LLM upgrade lands. Don't delete; wire up. Same for `health_monitor`: the registry entry is dead, but removing it forecloses the obvious extension point.

**C6 — Bandit α/γ "fix" (B16) vs in-flight learning.**
If existing user matrices were updated under `gamma=0.99` and you change to `0.95`, the discount applied to old learning will be slightly different going forward. Fine for new users, mildly disruptive for established ones. Decide whether to (a) leave it alone and update the docs, (b) change the default and accept a one-time drift, or (c) gate by user creation date.

**C7 — U17 (encrypt bandit tables) vs U4 (population warm-start).**
Per-user encryption breaks population-prior warm-start, because you can't average matrices you can't read in aggregate. Either bandit tables stay readable to the server (best) or population priors are computed offline at training time and stored unencrypted alongside the encrypted user-specific deltas.

---

## 6. Suggested sequencing (if you wanted to act on this list)

A conservative roadmap, ranked by ratio of safety-impact-to-effort:

| Order | Items | Rough effort | Why first |
|---|---|---|---|
| 1 | B5, B6, B17 | 1 h | Auth gate on perimeter; trivial. |
| 2 | B1, B2, B3, B4, U15 | 3–4 h | Close the open endpoints. Same JWT middleware. |
| 3 | B8, B13 | 30 min | Single timeout + Redis cache; eliminates a class of hangs. |
| 4 | B9, U14 | 2 h | Frontend port mismatch is a real production blocker. Move to runtime config. |
| 5 | Update CLAUDE.md to match § 1 + § 2 | 1 h | Stops new contributors building wrong mental models. |
| 6 | B25, B26, B27 | 30 min | Repo hygiene; trivial. |
| 7 | B12 + U11 | 1 h | Real accuracy win on emotion detection. |
| 8 | B10 (delete or wire up) | 30 min | Decisive. |
| 9 | U2 (merge router/style/NLP into one call) | 1 day | Big latency win. |
| 10 | U1 (LLM upgrade to 70B / Claude Haiku 4.5) | 1–2 days | Big quality win, allows deletion of regex layer. |
| 11 | U18 (CI) | 1 day | Future-proofing. |
| 12 | Everything else | as needed | |

Total to get to "would not be embarrassed to deploy this in front of strangers": about 2 working days.

---

## 7. What this analysis is *not*

- **Not a plan.** Nothing here is an instruction to change code. Every item above is a finding or a candidate.
- **Not exhaustive.** The four agents read every Python file in the agent directories but did not read the frontend in depth, did not run the tests, and did not load the live Docker stack to verify runtime behaviour.
- **Not a security audit.** § 3.1 is a fast pass over obvious holes. A real audit would also check session fixation, CSRF tokens, password reset flows (none exist), token revocation (none exists), and dependency CVEs.
- **Not opinionated about the bandit math.** The 19×14 architecture is documented as effective in SYSTEM_OVERVIEW.txt; whether it's the *right* parameterisation for elderly-care affect is an experiment, not an analysis.

The single most useful next step is to update CLAUDE.md so it stops actively misleading. After that, the security perimeter. After that, the Mistral 7B → larger-model upgrade is the lever that simplifies the most other code at once.
