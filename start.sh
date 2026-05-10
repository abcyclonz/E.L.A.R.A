#!/usr/bin/env bash
# E.L.A.R.A — single-command startup (no Docker)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/logs"
DATA="$ROOT/data"
mkdir -p "$LOGS" "$DATA/redis" "$DATA/assistant"

# ── Colors ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1;34m'; N='\033[0m'
log()  { echo -e "${G}[ELARA]${N} $*"; }
warn() { echo -e "${Y}[WARN] ${N} $*"; }
die()  { echo -e "${R}[ERROR]${N} $*"; exit 1; }
section() { echo -e "\n${B}━━━ $* ━━━${N}"; }

# ── PID tracking ─────────────────────────────────────────────────────────────
declare -a BGPIDS=()

cleanup() {
    echo ""
    log "Shutting down all services..."
    for pid in "${BGPIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pg_ctlcluster 16 main stop 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    log "All services stopped."
}
trap cleanup EXIT INT TERM

# ── Health-check helper ───────────────────────────────────────────────────────
wait_url() {
    local url="$1" name="$2" max="${3:-60}"
    log "Waiting for $name..."
    for i in $(seq 1 "$max"); do
        curl -sf "$url" &>/dev/null && { log "$name is ready"; return 0; }
        sleep 1
    done
    die "$name did not become healthy in ${max}s — check $LOGS/${name}.log"
}

# ─────────────────────────────────────────────────────────────────────────────
section "1 / 6  System Packages"
# ─────────────────────────────────────────────────────────────────────────────

MISSING=()
command -v psql        &>/dev/null || MISSING+=(postgresql-16 postgresql-16-pgvector)
command -v redis-server &>/dev/null || MISSING+=(redis-server)

if [ ${#MISSING[@]} -gt 0 ]; then
    log "Installing: ${MISSING[*]}"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${MISSING[@]}"
fi
log "System packages OK"

# ─────────────────────────────────────────────────────────────────────────────
section "2 / 6  Ollama"
# ─────────────────────────────────────────────────────────────────────────────

if ! command -v ollama &>/dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama daemon if not already running
if ! curl -sf http://localhost:11434 &>/dev/null; then
    log "Starting Ollama service..."
    OLLAMA_HOST=0.0.0.0 ollama serve >> "$LOGS/ollama.log" 2>&1 &
    BGPIDS+=($!)
    sleep 5
fi

# Pull required models (skips if already present)
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:32b}"
EMBED_MODEL="nomic-embed-text"

log "Checking model: $OLLAMA_MODEL"
ollama pull "$OLLAMA_MODEL"

log "Checking model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"

# ─────────────────────────────────────────────────────────────────────────────
section "3 / 6  PostgreSQL + Redis"
# ─────────────────────────────────────────────────────────────────────────────

# PostgreSQL
log "Starting PostgreSQL..."
if ! pg_ctlcluster 16 main status &>/dev/null; then
    pg_ctlcluster 16 main start
fi
sleep 2

# Idempotent DB/user setup
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='memory_user'" \
    | grep -q 1 || sudo -u postgres psql -c "CREATE USER memory_user WITH PASSWORD 'memory_pass';"

sudo -u postgres psql -lqt | cut -d'|' -f1 | grep -qw memory_db || \
    sudo -u postgres psql -c "CREATE DATABASE memory_db OWNER memory_user;"

sudo -u postgres psql -d memory_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || true
sudo -u postgres psql -d memory_db -f "$ROOT/agents/memory/sql/init.sql" >> "$LOGS/db_init.log" 2>&1 || true
sudo -u postgres psql -d memory_db -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO memory_user; GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO memory_user;" 2>/dev/null || true
log "PostgreSQL ready"

# Redis
log "Starting Redis..."
if ! redis-cli ping &>/dev/null; then
    redis-server --daemonize yes \
        --appendonly yes \
        --dir "$DATA/redis" \
        --logfile "$LOGS/redis.log"
    sleep 1
fi
log "Redis ready"

# ─────────────────────────────────────────────────────────────────────────────
section "4 / 6  Python Virtual Environments"
# ─────────────────────────────────────────────────────────────────────────────

setup_venv() {
    local name="$1" dir="$2" req="$3"
    if [ ! -d "$dir/.venv" ]; then
        log "Creating venv: $name"
        python3 -m venv "$dir/.venv"
        "$dir/.venv/bin/pip" install -q --upgrade pip wheel
        "$dir/.venv/bin/pip" install -q -r "$req"
    else
        log "Venv exists: $name (skipping)"
    fi
}

# Run in parallel — track PIDs explicitly so we don't wait on ollama/redis/etc.
setup_venv memory_agent   "$ROOT/agents/memory"           "$ROOT/agents/memory/requirements.txt" &  _v1=$!
setup_venv orchestrator   "$ROOT/agents/orchestrator"     "$ROOT/agents/orchestrator/requirements.txt" &  _v2=$!
setup_venv elara          "$ROOT/agents/elara"            "$ROOT/agents/elara/requirements.docker.txt" &  _v3=$!
setup_venv web_search     "$ROOT/agents/tools/web_search" "$ROOT/agents/tools/web_search/requirements.txt" &  _v4=$!
setup_venv assistant_tool "$ROOT/agents/tools/assistant"  "$ROOT/agents/tools/assistant/requirements.txt" &  _v5=$!
wait $_v1 $_v2 $_v3 $_v4 $_v5
# perception_learner reuses orchestrator venv (same deps: fastapi, uvicorn, httpx)
ln -sfn "$ROOT/agents/orchestrator/.venv" "$ROOT/agents/perception_learner/.venv"
log "All Python venvs ready"

# ─────────────────────────────────────────────────────────────────────────────
section "5 / 6  Python Services"
# ─────────────────────────────────────────────────────────────────────────────

# Memory Agent
log "Starting Memory Agent (port 8000)..."
pushd "$ROOT/agents/memory" > /dev/null
DATABASE_URL="postgresql://memory_user:memory_pass@localhost:5432/memory_db" \
OLLAMA_URL="http://localhost:11434" \
OLLAMA_MODEL="$OLLAMA_MODEL" \
REDIS_URL="redis://localhost:6379/0" \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 >> "$LOGS/memory_agent.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null
wait_url "http://localhost:8000/health" "memory_agent"

# Web Search Tool
log "Starting Web Search Tool (port 8010)..."
pushd "$ROOT/agents/tools/web_search" > /dev/null
TAVILY_API_KEY="${TAVILY_API_KEY:-}" \
PORT=8010 \
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8010 >> "$LOGS/web_search.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null

# Assistant Tool
log "Starting Assistant Tool (port 8011)..."
pushd "$ROOT/agents/tools/assistant" > /dev/null
DB_PATH="$DATA/assistant/assistant.db" \
GOOGLE_AUTH_DIR="$ROOT/agents/tools/assistant/google_auth" \
CALENDAR_TIMEZONE="Asia/Kolkata" \
PORT=8011 \
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8011 >> "$LOGS/assistant_tool.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null

wait_url "http://localhost:8010/health" "web_search_tool" 30
wait_url "http://localhost:8011/health" "assistant_tool" 30

# Elara Agent
log "Starting Elara Agent (port 8002)..."
pushd "$ROOT/agents/elara" > /dev/null
OLLAMA_URL="http://localhost:11434" \
OLLAMA_MODEL="$OLLAMA_MODEL" \
GMAIL_APP_PASSWORD="${GMAIL_APP_PASSWORD:-}" \
ALERT_FROM_EMAIL="${ALERT_FROM_EMAIL:-}" \
ALERT_TO_EMAIL="${ALERT_TO_EMAIL:-}" \
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8002 >> "$LOGS/elara.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null
wait_url "http://localhost:8002/health" "elara" 60

# Orchestrator
log "Starting Orchestrator (port 8003)..."
pushd "$ROOT/agents/orchestrator" > /dev/null
MEMORY_AGENT_URL="http://localhost:8000" \
ELARA_URL="http://localhost:8002" \
OLLAMA_URL="http://localhost:11434" \
OLLAMA_MODEL="$OLLAMA_MODEL" \
WEB_SEARCH_MCP_URL="http://localhost:8010" \
ASSISTANT_MCP_URL="http://localhost:8011" \
REDIS_URL="redis://localhost:6379/0" \
JWT_SECRET_KEY="${JWT_SECRET_KEY:-elara-dev-secret-change-in-prod}" \
TAVILY_API_KEY="${TAVILY_API_KEY:-}" \
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8003 >> "$LOGS/orchestrator.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null
wait_url "http://localhost:8003/health" "orchestrator" 60

# Perception Learner (passive; skips if no perception DB yet)
log "Starting Perception Learner (port 8012)..."
pushd "$ROOT/agents/perception_learner" > /dev/null
mkdir -p "$DATA/perception" "$DATA/learner"
MEMORY_AGENT_URL="http://localhost:8000" \
OLLAMA_URL="http://localhost:11434" \
OLLAMA_MODEL="$OLLAMA_MODEL" \
PERCEPTION_DB_PATH="$DATA/perception/timeline.db" \
POLL_INTERVAL_S="${POLL_INTERVAL_S:-600}" \
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8012 >> "$LOGS/perception_learner.log" 2>&1 &
BGPIDS+=($!)
popd > /dev/null
wait_url "http://localhost:8012/health" "perception_learner" 30

# ─────────────────────────────────────────────────────────────────────────────
section "7 / 7  Frontend"
# ─────────────────────────────────────────────────────────────────────────────

log "Setting up Next.js frontend..."
pushd "$ROOT/frontend" > /dev/null
[ -d node_modules ] || npm install --legacy-peer-deps --silent
if BACKEND_URL="http://localhost:8003" PI_CAMERA_URL="${PI_URL:-http://localhost:8765}" npm run build >> "$LOGS/frontend_build.log" 2>&1; then
    log "Frontend built — starting..."
    BACKEND_URL="http://localhost:8003" \
    PI_CAMERA_URL="${PI_URL:-http://localhost:8765}" \
    NODE_ENV=production \
    npm start >> "$LOGS/frontend.log" 2>&1 &
    BGPIDS+=($!)
    popd > /dev/null
    wait_url "http://localhost:3000" "frontend" 60
else
    popd > /dev/null
    warn "Frontend build failed — backend services are still running. Check logs/frontend_build.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${G}╔══════════════════════════════════════════════════════╗${N}"
echo -e "${G}║          E.L.A.R.A  is  running  🚀                 ║${N}"
echo -e "${G}╠══════════════════════════════════════════════════════╣${N}"
echo -e "${G}║  Frontend      →  http://localhost:3000              ║${N}"
echo -e "${G}║  Orchestrator  →  http://localhost:8003              ║${N}"
echo -e "${G}║  Elara         →  http://localhost:8002              ║${N}"
echo -e "${G}║  Memory Agent  →  http://localhost:8000              ║${N}"
echo -e "${G}║  Ollama        →  http://localhost:11434             ║${N}"
echo -e "${G}╠══════════════════════════════════════════════════════╣${N}"
echo -e "${G}║  Logs: ./logs/<service>.log                          ║${N}"
echo -e "${G}║  Press Ctrl+C to stop all services                   ║${N}"
echo -e "${G}╚══════════════════════════════════════════════════════╝${N}"
echo ""

# Block until Ctrl+C
wait
