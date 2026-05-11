#!/bin/bash
# E.L.A.R.A. — Pi full startup script
# Starts: Face Login (camera + servos), LCD/OTP, Audio Client

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── CONFIGURE THESE if your folder names differ ───────────────────────────────
FACE_LOGIN_DIR="$SCRIPT_DIR/Face_login_Elara-master"   # camera + servo server
LCD_DIR="$SCRIPT_DIR/ras-LCD"                           # LCD + OTP script
AUDIO_DIR="$SCRIPT_DIR/pi_client"                       # mic + speaker client
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── 1. Face Login (camera + servo server, port 8765) ─────────────────────────
echo "[1/3] Starting Face Login server..."
cd "$FACE_LOGIN_DIR"

if command -v uv &>/dev/null; then
    uv run python main.py >> "$LOG_DIR/face_login.log" 2>&1 &
else
    echo "  uv not found — falling back to system python"
    python3 main.py >> "$LOG_DIR/face_login.log" 2>&1 &
fi
PIDS+=($!)
echo "  PID $! → logs/face_login.log"

# ── 2. LCD / OTP ─────────────────────────────────────────────────────────────
echo "[2/3] Starting LCD / OTP..."
cd "$LCD_DIR"

if [ ! -d venv ]; then
    echo "  Creating venv for ras-LCD..."
    python3 -m venv venv
    venv/bin/pip install -q --upgrade pip
    venv/bin/pip install -q -r requirements.txt
fi

venv/bin/python rasb-otp.py >> "$LOG_DIR/lcd_otp.log" 2>&1 &
PIDS+=($!)
echo "  PID $! → logs/lcd_otp.log"

# ── 3. Audio Client (mic + speaker) ──────────────────────────────────────────
echo "[3/3] Starting Audio Client..."
cd "$AUDIO_DIR"

if [ ! -f .env ]; then
    echo "ERROR: pi_client/.env not found."
    echo "Copy pi_client/.env.example to pi_client/.env and set ORCHESTRATOR_WS_URL."
    cleanup
fi

if ! python3 -c "import pyaudio" 2>/dev/null; then
    echo "  Installing portaudio..."
    sudo apt-get install -y portaudio19-dev
fi

if [ ! -d venv ]; then
    echo "  Creating venv for pi_client..."
    python3 -m venv venv
    venv/bin/pip install -q --upgrade pip
    venv/bin/pip install -q -r requirements.txt
fi

echo ""
echo "All services started. Logs in: $SCRIPT_DIR/logs/"
echo "Press Ctrl+C to stop everything."
echo ""

source venv/bin/activate
python audio_client.py
