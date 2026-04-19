"""
Interactive test client for the full multi-agent system.
Run: python3 test_chat.py
"""

import json
import requests
import sys

ORCHESTRATOR = "http://localhost:8001"
MEMORY_AGENT  = "http://localhost:8000"
ELARA         = "http://localhost:8002"

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
MAGENTA= "\033[35m"
DIM    = "\033[2m"

def section(title: str, color: str = CYAN):
    print(f"\n{color}{BOLD}{'─' * 50}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'─' * 50}{RESET}")

def kv(key: str, value, color: str = RESET):
    print(f"  {DIM}{key}:{RESET} {color}{value}{RESET}")

def check_services():
    ok = True
    for name, url in [("Orchestrator", ORCHESTRATOR), ("Memory Agent", MEMORY_AGENT), ("Elara", ELARA)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            print(f"  {GREEN}✓{RESET} {name} ({url})")
        except Exception:
            print(f"  {RED}✗{RESET} {name} — not reachable at {url}")
            ok = False
    return ok

def send(text: str, speaker: str, emotion: str = None) -> dict:
    payload = {"text": text, "speaker": speaker}
    if emotion:
        payload["emotion"] = emotion
    r = requests.post(f"{ORCHESTRATOR}/input", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()

def display(result: dict, text: str):
    # ── Router decision ────────────────────────────────────────────────────
    debug = result.get("debug", {})
    action = debug.get("router_action", "—")
    action_color = {
        "STORE_MEMORY": MAGENTA, "RETRIEVE_MEMORY": CYAN,
        "STORE_AND_RETRIEVE": YELLOW, "DIRECT_CHAT": GREEN,
        "USE_TOOL": YELLOW,
    }.get(action, RESET)
    section(f"ROUTER → {action}", action_color)
    kv("Reasoning", debug.get("router_reason", "—"), DIM)
    kv("Emotion used", debug.get("emotion_used") or "—", DIM)

    # ── Tool result ────────────────────────────────────────────────────────
    tool_called = result.get("tool_called")
    if tool_called:
        section(f"TOOL → {tool_called}", YELLOW)
        active = result.get("active_states", [])
        tool_val = next((s["value"] for s in active if s.get("entity") == "tool"), None)
        if tool_val:
            kv("Result", tool_val, DIM)

    # ── Memory decisions ───────────────────────────────────────────────────
    if action not in ("DIRECT_CHAT", "USE_TOOL"):
        section("MEMORY AGENT", MAGENTA)
        kv("Memory stored",     result.get("memory_stored"), GREEN if result.get("memory_stored") else DIM)
        kv("Claims extracted",  debug.get("claims_extracted", 0))
        kv("Memory retrieved",  result.get("memory_used"), GREEN if result.get("memory_used") else DIM)
        kv("Intent classified", result.get("intent") or "—")

    active = result.get("active_states", [])
    if active and action not in ("DIRECT_CHAT", "USE_TOOL"):
        print(f"\n  {MAGENTA}Active memory states:{RESET}")
        for s in active:
            print(f"    • {s['entity']}.{s['attribute']} = {BOLD}{s['value']}{RESET}"
                  f"  {DIM}(conf: {s.get('confidence', '?')}, emotion: {s.get('emotion', '?')}){RESET}")

    # ── Elara / Learning Agent ─────────────────────────────────────────────
    section("ELARA — LEARNING AGENT", YELLOW)
    affect = result.get("affect", "unknown")
    affect_color = {"calm": GREEN, "frustrated": RED, "confused": YELLOW,
                    "sad": MAGENTA, "disengaged": RED}.get(affect, RESET)
    kv("Inferred affect",   affect, affect_color)
    kv("Caregiver alert",   result.get("caregiver_alert", False),
       RED if result.get("caregiver_alert") else DIM)

    # ── Reply ──────────────────────────────────────────────────────────────
    section("ELARA — REPLY", GREEN)
    reply = result.get("reply", "")
    print(f"\n  {BOLD}{reply}{RESET}\n")

def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   ELARA Multi-Agent System — Interactive Test  ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}\n")

    print("Checking services...")
    if not check_services():
        print(f"\n{RED}Some services are down. Run: docker compose up{RESET}")
        sys.exit(1)

    print(f"\n{DIM}Commands:{RESET}")
    print(f"  {DIM}Just type to chat as User_1{RESET}")
    print(f"  {DIM}Prefix with @name  to switch speaker:  @User_2 hello{RESET}")
    print(f"  {DIM}Prefix with !emo   to set emotion:     !angry I hate this{RESET}")
    print(f"  {DIM}Type 'quit' to exit{RESET}\n")

    speaker = "User_1"
    emotion = None

    while True:
        try:
            raw = input(f"{CYAN}{BOLD}{speaker}{RESET}{CYAN}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # Parse speaker switch: @Name message
        if raw.startswith("@"):
            parts = raw.split(" ", 1)
            speaker = parts[0][1:]
            raw = parts[1] if len(parts) > 1 else ""
            if not raw:
                print(f"  {DIM}Switched to speaker: {speaker}{RESET}")
                continue

        # Parse emotion override: !emotion message
        if raw.startswith("!"):
            parts = raw.split(" ", 1)
            emotion = parts[0][1:]
            raw = parts[1] if len(parts) > 1 else ""
            print(f"  {DIM}Emotion set to: {emotion}{RESET}")
            if not raw:
                continue
        else:
            emotion = None

        try:
            result = send(raw, speaker, emotion)
            display(result, raw)
        except requests.exceptions.ConnectionError:
            print(f"{RED}  Connection failed — is docker compose up?{RESET}")
        except Exception as e:
            print(f"{RED}  Error: {e}{RESET}")

if __name__ == "__main__":
    main()
