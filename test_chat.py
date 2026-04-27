"""
Interactive test client for the ELARA multi-agent system.

Two modes:
  python3 test_chat.py           — full pipeline via Orchestrator (port 8001)
  python3 test_chat.py --elara   — direct Elara chat (port 8002), shows personality
"""

import json
import sys
import requests

ORCHESTRATOR = "http://localhost:8001"
MEMORY_AGENT  = "http://localhost:8000"
ELARA         = "http://localhost:8002"

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[36m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
RED     = "\033[31m"
MAGENTA = "\033[35m"
BLUE    = "\033[34m"
DIM     = "\033[2m"

ACTION_NAMES = {
    0:  "DO_NOTHING",
    1:  "INCREASE_WARMTH",       2:  "DECREASE_WARMTH",
    3:  "INCREASE_HUMOR",        4:  "DECREASE_HUMOR",
    5:  "INCREASE_PLAYFULNESS",  6:  "DECREASE_PLAYFULNESS",
    7:  "INCREASE_FORMALITY",    8:  "DECREASE_FORMALITY",
    9:  "INCREASE_VERBOSITY",   10:  "DECREASE_VERBOSITY",
    11: "INCREASE_PACE",        12:  "DECREASE_PACE",
    13: "INCREASE_CLARITY",     14:  "DECREASE_CLARITY",
    15: "INCREASE_PATIENCE",    16:  "DECREASE_PATIENCE",
    17: "INCREASE_ASSERTIVENESS", 18: "DECREASE_ASSERTIVENESS",
}

PERSONALITY_DIMS = [
    "warmth", "humor", "playfulness", "formality",
    "verbosity", "pace", "clarity", "patience", "assertiveness",
]


def section(title: str, color: str = CYAN):
    print(f"\n{color}{BOLD}{'─' * 52}{RESET}")
    print(f"{color}{BOLD}  {title}{RESET}")
    print(f"{color}{BOLD}{'─' * 52}{RESET}")


def kv(key: str, value, color: str = RESET):
    print(f"  {DIM}{key}:{RESET} {color}{value}{RESET}")


def bar(val: float, width: int = 20) -> str:
    filled = round(val * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {val:.2f}"


# ── Service health check ──────────────────────────────────────────────────────

def check_services(direct_elara: bool = False):
    targets = [("Elara", ELARA)] if direct_elara else [
        ("Orchestrator", ORCHESTRATOR),
        ("Memory Agent", MEMORY_AGENT),
        ("Elara", ELARA),
    ]
    ok = True
    for name, url in targets:
        try:
            requests.get(f"{url}/health", timeout=3)
            print(f"  {GREEN}✓{RESET} {name} ({url})")
        except Exception:
            print(f"  {RED}✗{RESET} {name} — not reachable at {url}")
            ok = False
    return ok


# ── Orchestrator mode ─────────────────────────────────────────────────────────

def send_orchestrator(text: str, speaker: str, emotion: str = None) -> dict:
    payload = {"text": text, "speaker": speaker}
    if emotion:
        payload["emotion"] = emotion
    r = requests.post(f"{ORCHESTRATOR}/input", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def display_orchestrator(result: dict):
    debug  = result.get("debug", {})
    action = debug.get("router_action", "—")
    action_color = {
        "STORE_MEMORY": MAGENTA, "RETRIEVE_MEMORY": CYAN,
        "STORE_AND_RETRIEVE": YELLOW, "DIRECT_CHAT": GREEN,
        "USE_TOOL": YELLOW,
    }.get(action, RESET)
    section(f"ROUTER → {action}", action_color)
    kv("Reasoning",   debug.get("router_reason", "—"), DIM)
    kv("Emotion used", debug.get("emotion_used") or "—", DIM)

    if result.get("tool_called"):
        section(f"TOOL → {result['tool_called']}", YELLOW)
        active = result.get("active_states", [])
        tool_val = next((s["value"] for s in active if s.get("entity") == "tool"), None)
        if tool_val:
            kv("Result", tool_val, DIM)

    if action not in ("DIRECT_CHAT", "USE_TOOL"):
        section("MEMORY AGENT", MAGENTA)
        kv("Memory stored",    result.get("memory_stored"), GREEN if result.get("memory_stored") else DIM)
        kv("Claims extracted", debug.get("claims_extracted", 0))
        kv("Memory retrieved", result.get("memory_used"), GREEN if result.get("memory_used") else DIM)
        kv("Intent classified", result.get("intent") or "—")

        active = result.get("active_states", [])
        if active:
            print(f"\n  {MAGENTA}Active memory states:{RESET}")
            for s in active:
                print(f"    • {s['entity']}.{s['attribute']} = {BOLD}{s['value']}{RESET}"
                      f"  {DIM}(conf: {s.get('confidence','?')}, emotion: {s.get('emotion','?')}){RESET}")

    section("ELARA — LEARNING AGENT", YELLOW)
    affect = result.get("affect", "unknown")
    affect_color = {"calm": GREEN, "frustrated": RED, "confused": YELLOW,
                    "sad": MAGENTA, "disengaged": RED}.get(affect, RESET)
    kv("Inferred affect",  affect, affect_color)
    kv("Caregiver alert",  result.get("caregiver_alert", False),
       RED if result.get("caregiver_alert") else DIM)

    section("ELARA — REPLY", GREEN)
    print(f"\n  {BOLD}{result.get('reply', '')}{RESET}\n")


# ── Direct Elara mode ─────────────────────────────────────────────────────────

def send_elara(text: str, state: dict | None, backend: str = "ollama") -> dict:
    payload = {"message": text, "backend": backend}
    if state:
        payload["state"] = state
    r = requests.post(f"{ELARA}/chat", json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def display_elara(result: dict):
    diag  = result.get("diagnostics", {})
    state = result.get("state", {})

    # ── NLP Signals ──────────────────────────────────────────────────────────
    section("NLP SIGNALS (LLM-extracted)", BLUE)
    kv("Sentiment",       f"{diag.get('sentiment_score', 0.0):+.3f}")
    kv("Confusion",       f"{diag.get('confusion_score', 0.0):.3f}")
    kv("Sadness",         f"{diag.get('sadness_score', 0.0):.3f}")
    kv("Repetition",      f"{diag.get('repetition_score', 0.0):.3f}")
    sigs = diag.get("signals_used", [])
    if sigs:
        kv("Signals fired",   ", ".join(sigs), YELLOW)

    # ── Affect & Learning ─────────────────────────────────────────────────────
    section("LEARNING AGENT", YELLOW)
    affect = diag.get("affect", "unknown")
    affect_color = {"calm": GREEN, "frustrated": RED, "confused": YELLOW,
                    "sad": MAGENTA, "disengaged": RED}.get(affect, RESET)
    kv("Affect",          affect, affect_color)
    kv("Confidence",      f"{diag.get('confidence', 0.0):.3f}")

    action_id = diag.get("ucb_action_id", 0)
    kv("Bandit action",   f"{ACTION_NAMES.get(action_id, str(action_id))} (id={action_id})")

    reward = diag.get("reward_applied")
    if reward is not None:
        r_color = GREEN if reward > 0 else RED if reward < 0 else DIM
        kv("Reward applied",  f"{reward:+.3f}", r_color)

    if diag.get("escalation_rule"):
        kv("Escalation rule", diag["escalation_rule"], DIM)
    if diag.get("immediate_reward_applied"):
        kv("Immediate reward", "+1.0 (explicit positive feedback)", GREEN)
    if diag.get("caregiver_alert"):
        kv("Caregiver alert", f"YES — {diag.get('distress_turns', 0)} consecutive non-calm turns", RED)

    # ── Personality Vector ────────────────────────────────────────────────────
    section("PERSONALITY VECTOR", CYAN)
    pers = state.get("personality", {})
    if pers:
        for dim in PERSONALITY_DIMS:
            val = pers.get(dim, 0.0)
            bar_color = GREEN if val > 0.6 else RED if val < 0.3 else YELLOW
            print(f"  {DIM}{dim:<14}{RESET}  {bar_color}{bar(val)}{RESET}")
    else:
        print(f"  {DIM}(not available){RESET}")

    # ── Config changes ────────────────────────────────────────────────────────
    changes = diag.get("config_changes", {})
    if changes:
        section("PERSONALITY CHANGES THIS TURN", MAGENTA)
        for k, v in changes.items():
            print(f"  {MAGENTA}{k}{RESET} → {BOLD}{v:.3f}{RESET}")

    # ── Reply ─────────────────────────────────────────────────────────────────
    section("ELARA — REPLY", GREEN)
    print(f"\n  {BOLD}{result.get('reply', '')}{RESET}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    direct_elara = "--elara" in sys.argv

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    if direct_elara:
        print(f"{BOLD}{CYAN}║  ELARA Direct Mode — Personality Diagnostics  ║{RESET}")
    else:
        print(f"{BOLD}{CYAN}║   ELARA Multi-Agent System — Interactive Test  ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}\n")

    print("Checking services...")
    if not check_services(direct_elara):
        if direct_elara:
            print(f"\n{RED}Elara is not running. Start it with: docker compose up elara{RESET}")
        else:
            print(f"\n{RED}Some services are down. Run: docker compose up{RESET}")
        sys.exit(1)

    if direct_elara:
        print(f"\n{DIM}Direct Elara mode — full personality + signal diagnostics shown{RESET}")
        print(f"{DIM}Type 'quit' to exit{RESET}\n")

        elara_state = None
        while True:
            try:
                raw = input(f"{CYAN}{BOLD}you{RESET}{CYAN}> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            try:
                result     = send_elara(raw, elara_state)
                elara_state = result.get("state")
                display_elara(result)
            except requests.exceptions.ConnectionError:
                print(f"{RED}  Connection failed — is Elara running?{RESET}")
            except Exception as e:
                print(f"{RED}  Error: {e}{RESET}")

    else:
        print(f"\n{DIM}Commands:{RESET}")
        print(f"  {DIM}Just type to chat as User_1{RESET}")
        print(f"  {DIM}@name message  — switch speaker{RESET}")
        print(f"  {DIM}!emotion message  — set emotion override{RESET}")
        print(f"  {DIM}quit — exit{RESET}\n")

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

            if raw.startswith("@"):
                parts  = raw.split(" ", 1)
                speaker = parts[0][1:]
                raw    = parts[1] if len(parts) > 1 else ""
                if not raw:
                    print(f"  {DIM}Switched to speaker: {speaker}{RESET}")
                    continue

            if raw.startswith("!"):
                parts   = raw.split(" ", 1)
                emotion = parts[0][1:]
                raw     = parts[1] if len(parts) > 1 else ""
                print(f"  {DIM}Emotion set to: {emotion}{RESET}")
                if not raw:
                    continue
            else:
                emotion = None

            try:
                result = send_orchestrator(raw, speaker, emotion)
                display_orchestrator(result)
            except requests.exceptions.ConnectionError:
                print(f"{RED}  Connection failed — is docker compose up?{RESET}")
            except Exception as e:
                print(f"{RED}  Error: {e}{RESET}")


if __name__ == "__main__":
    main()
