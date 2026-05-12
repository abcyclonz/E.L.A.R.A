#!/usr/bin/env bash
# Keeps cloudflared tunnel alive. On each (re)start, writes the new WSS URL
# to /workspace/E.L.A.R.A/logs/tunnel_url.txt so you can always find it.

LOG=/workspace/E.L.A.R.A/logs/cloudflared.log
URL_FILE=/workspace/E.L.A.R.A/logs/tunnel_url.txt
TARGET="http://localhost:8003"

while true; do
    echo "[watchdog] Starting cloudflared tunnel → $TARGET" | tee -a "$LOG"

    # Run cloudflared, tee output so we can grep the URL in real time
    cloudflared tunnel --url "$TARGET" --no-autoupdate 2>&1 | tee -a "$LOG" | while IFS= read -r line; do
        if [[ "$line" =~ https://([a-z0-9-]+\.trycloudflare\.com) ]]; then
            url="wss://${BASH_REMATCH[1]}/ws/audio"
            echo "$url" > "$URL_FILE"
            echo ""
            echo "╔══════════════════════════════════════════════════════════════╗"
            echo "║  TUNNEL URL (update Pi .env with this)                       ║"
            echo "║  ORCHESTRATOR_WS_URL=$url"
            echo "╚══════════════════════════════════════════════════════════════╝"
            echo ""
        fi
    done

    echo "[watchdog] cloudflared exited — restarting in 5s…" | tee -a "$LOG"
    sleep 5
done
