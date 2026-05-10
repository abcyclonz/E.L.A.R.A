#!/bin/bash
set -e

# ── E.L.A.R.A. Pi Audio Client ───────────────────────────────────────────────
# Run this once to set up, then use:  python audio_client.py
# ─────────────────────────────────────────────────────────────────────────────

# 1. Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and set your RunPod URL."
    exit 1
fi

# 2. Install system dep for PyAudio (Raspberry Pi OS / Debian)
if ! python3 -c "import pyaudio" 2>/dev/null; then
    echo "Installing portaudio (needed by PyAudio)..."
    sudo apt-get install -y portaudio19-dev
fi

# 3. Create venv if it doesn't exist
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 4. Install Python deps
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 5. Run
echo ""
echo "Starting E.L.A.R.A. audio client..."
echo "Speak into the mic. Press Ctrl+C to stop."
echo ""
python audio_client.py
