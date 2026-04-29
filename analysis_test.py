"""
Full-pipeline analysis test — simulates an elderly user (George, 74) across
8 distinct scenarios covering every major system feature.

Runs against the orchestrator at :8001 (full pipeline).
"""

import json, time, textwrap
import requests

ORCHESTRATOR = "http://localhost:8001"
ELARA        = "http://localhost:8002"
SPEAKER      = "User_1"

RESET   = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
CYAN    = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
RED     = "\033[31m"; MAGENTA = "\033[35m"; BLUE = "\033[34m"

issues = []   # collect problems for summary

def post(text, speaker=SPEAKER, emotion=None):
    payload = {"text": text, "speaker": speaker}
    if emotion:
        payload["emotion"] = emotion
    try:
        r = requests.post(f"{ORCHESTRATOR}/input", json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "reply": f"[ERROR: {e}]"}

def header(title):
    print(f"\n{CYAN}{BOLD}{'═'*64}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'═'*64}{RESET}")

def scenario(n, desc):
    print(f"\n\n{YELLOW}{BOLD}{'━'*64}{RESET}")
    print(f"{YELLOW}{BOLD}  SCENARIO {n}: {desc}{RESET}")
    print(f"{YELLOW}{BOLD}{'━'*64}{RESET}")

def turn(user_msg, result, note=""):
    action = result.get("debug", {}).get("router_action", "—")
    affect = result.get("affect", "—")
    reply  = result.get("reply", result.get("error", ""))
    caregiver = result.get("caregiver_alert", False)

    action_color = {
        "STORE_MEMORY": MAGENTA, "RETRIEVE_MEMORY": CYAN,
        "STORE_AND_RETRIEVE": YELLOW, "DIRECT_CHAT": GREEN,
        "USE_TOOL": BLUE,
    }.get(action, RESET)
    affect_color = {
        "calm": GREEN, "frustrated": RED, "confused": YELLOW,
        "sad": MAGENTA, "disengaged": RED,
    }.get(affect, RESET)

    print(f"\n{BOLD}[USER]{RESET}  {user_msg}")
    if note:
        print(f"  {DIM}(expected: {note}){RESET}")
    print(f"  {action_color}ROUTER → {action}{RESET}   |   {affect_color}AFFECT: {affect}{RESET}", end="")
    if caregiver:
        print(f"   {RED}{BOLD}⚠ CAREGIVER ALERT{RESET}", end="")
    print()

    # memory details
    if result.get("memory_stored"):
        claims = result.get("debug", {}).get("claims_extracted", "?")
        print(f"  {MAGENTA}MEMORY STORED{RESET}  ({claims} claims)")
        for s in result.get("active_states", []):
            print(f"    • {s['entity']}.{s['attribute']} = {BOLD}{s['value']}{RESET}  {DIM}[{s.get('stability','?')} | imp={s.get('importance','?')}]{RESET}")
    if result.get("memory_used"):
        print(f"  {CYAN}MEMORY RETRIEVED{RESET}  intent={result.get('intent','?')}")
    if result.get("tool_called"):
        print(f"  {BLUE}TOOL → {result['tool_called']}{RESET}")

    reply_lines = textwrap.wrap(reply, 80)
    print(f"\n{GREEN}{BOLD}  ELARA:{RESET}", end="")
    for i, ln in enumerate(reply_lines):
        prefix = "  " if i > 0 else " "
        print(f"{prefix}{ln}")

    # flag issues
    if action == "STORE_MEMORY" and "STORE" not in note.upper() and note:
        issues.append(f"WRONG ROUTE: '{user_msg[:45]}' → {action} (expected {note})")
    if action == "DIRECT_CHAT" and "DIRECT" not in note.upper() and "GREET" not in note.upper() and note:
        issues.append(f"WRONG ROUTE: '{user_msg[:45]}' → {action} (expected {note})")
    if "error" in result:
        issues.append(f"ERROR on '{user_msg[:45]}': {result['error']}")

    return result


# ─────────────────────────────────────────────────────────────────────────────

header("ELARA FULL-PIPELINE ANALYSIS  — George, 74, Kerala")
print(f"\n{DIM}Testing: greetings · memory · tools · emotion · retrieval · distress{RESET}")

# ── SCENARIO 1: Greetings & introductions ────────────────────────────────────
scenario(1, "Greetings & self-introduction")

r = post("Hello there!")
turn("Hello there!", r, "DIRECT_CHAT (greeting)")
time.sleep(1)

r = post("My name is George and I am 74 years old, living in Trivandrum, Kerala")
turn("My name is George and I am 74 years old, living in Trivandrum, Kerala", r, "STORE_MEMORY")
time.sleep(1)

r = post("I used to be a schoolteacher for 35 years, retired now")
turn("I used to be a schoolteacher for 35 years, retired now", r, "STORE_MEMORY")
time.sleep(1)


# ── SCENARIO 2: Family, personal life ────────────────────────────────────────
scenario(2, "Sharing family & personal life")

r = post("I have two sons. David lives in London with his wife and two kids. Steven is here in Kerala, still single, poor lad")
turn("I have two sons. David lives in London with his wife and two kids. Steven is still single.", r, "STORE_MEMORY")
time.sleep(1)

r = post("I absolutely love cupcakes, my wife used to make the most wonderful ones")
turn("I absolutely love cupcakes, my wife used to make the most wonderful ones", r, "STORE_MEMORY")
time.sleep(1)

r = post("I also have a little garden, I grow roses and marigolds mostly")
turn("I also have a little garden, I grow roses and marigolds mostly", r, "STORE_MEMORY")
time.sleep(1)


# ── SCENARIO 3: Memory retrieval ─────────────────────────────────────────────
scenario(3, "Memory retrieval — asking about what was shared")

r = post("Do you remember where my son David lives?")
turn("Do you remember where my son David lives?", r, "RETRIEVE_MEMORY")
time.sleep(1)

r = post("What do you know about me so far?")
turn("What do you know about me so far?", r, "RETRIEVE_MEMORY")
time.sleep(1)


# ── SCENARIO 4: Health concerns ──────────────────────────────────────────────
scenario(4, "Health concerns — shoulder pain, doctor visit")

r = post("My shoulder has been giving me a lot of trouble lately, very painful when I lift things")
turn("My shoulder has been giving me a lot of trouble lately", r, "STORE_MEMORY")
time.sleep(1)

r = post("The doctor thinks it might be arthritis. I have an appointment next week for an x-ray")
turn("The doctor thinks it might be arthritis", r, "STORE_MEMORY")
time.sleep(1)


# ── SCENARIO 5: Grief & emotional weight ─────────────────────────────────────
scenario(5, "Grief — loss of wife, flowers on grave")

r = post("I miss my wife terribly. She passed away 3 years ago. Some days the house feels so empty")
turn("I miss my wife terribly. She passed away 3 years ago", r, "STORE_MEMORY (sad affect)")
time.sleep(1.5)

r = post("Last Sunday I went and put flowers on her grave. Roses. She loved roses.")
turn("Last Sunday I went and put flowers on her grave. She loved roses.", r, "STORE_MEMORY (should NOT ask about garden/flowers - wrong context)")
time.sleep(1.5)

r = post("She was the love of my life for 48 years")
turn("She was the love of my life for 48 years", r, "STORE_MEMORY")
time.sleep(1.5)


# ── SCENARIO 6: Tool use — web search ────────────────────────────────────────
scenario(6, "Tool use — web search & reminders")

r = post("Oh! I am actually staying at a hotel near the city centre this weekend for my grandson's birthday")
turn("Staying at hotel near city centre this weekend", r, "STORE_MEMORY/DIRECT_CHAT")
time.sleep(1)

r = post("Can you search for good bakeries near the city centre in Trivandrum?")
turn("Can you search for good bakeries near city centre Trivandrum?", r, "USE_TOOL (web_search)")
time.sleep(2)

r = post("Remind me to call David tomorrow at 6 in the evening")
turn("Remind me to call David tomorrow at 6 in the evening", r, "USE_TOOL (set_reminder)")
time.sleep(1)

r = post("What reminders do I have?")
turn("What reminders do I have?", r, "USE_TOOL (list_reminders)")
time.sleep(1)


# ── SCENARIO 7: Confusion & frustration ──────────────────────────────────────
scenario(7, "Frustration / confusion — testing escalation rules")

r = post("What? I already told you all of this before! Why are you asking again?")
turn("What? I already told you all of this before!", r, "DIRECT_CHAT (frustrated)")
time.sleep(1)

r = post("I feel like you never remember anything I say. It's very frustrating.")
turn("I feel like you never remember anything I say.", r, "DIRECT_CHAT (frustrated)")
time.sleep(1)

r = post("Sorry, I'm just having a bad day. I didn't mean to be rude.")
turn("Sorry, I'm just having a bad day. I didn't mean to be rude.", r, "DIRECT_CHAT (calm→recovery)")
time.sleep(1)


# ── SCENARIO 8: Casual natural chat ──────────────────────────────────────────
scenario(8, "Casual chat — cricket, weather, small talk")

r = post("It's a lovely evening here, the garden is looking beautiful")
turn("It's a lovely evening here, the garden is looking beautiful", r, "DIRECT_CHAT")
time.sleep(1)

r = post("I was watching the cricket match on the telly earlier, India did very well!")
turn("Watching cricket on the telly earlier, India did very well!", r, "DIRECT_CHAT")
time.sleep(1)

r = post("You know what, you're actually quite good company for an old man like me, haha!")
turn("You're quite good company for an old man like me, haha!", r, "DIRECT_CHAT (positive feedback)")
time.sleep(1)

r = post("Thanks so much for everything today")
turn("Thanks so much for everything today", r, "DIRECT_CHAT (gratitude → You're welcome)")
time.sleep(1)


# ── SCENARIO 9: Elara-directed questions ─────────────────────────────────────
scenario(9, "Questions directed at Elara herself")

r = post("How are you doing today Elara?")
turn("How are you doing today Elara?", r, "DIRECT_CHAT (not STORE_MEMORY)")
time.sleep(1)

r = post("What do you think of gardening?")
turn("What do you think of gardening?", r, "DIRECT_CHAT")
time.sleep(1)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
header("ANALYSIS SUMMARY")

if issues:
    print(f"\n{RED}{BOLD}  Issues detected ({len(issues)}):{RESET}")
    for i, issue in enumerate(issues, 1):
        print(f"  {RED}{i}. {issue}{RESET}")
else:
    print(f"\n{GREEN}{BOLD}  No routing/error issues flagged automatically.{RESET}")

print(f"\n{DIM}  Check the output above manually for:{RESET}")
print(f"  {DIM}• Reply quality & tone appropriateness{RESET}")
print(f"  {DIM}• Memory extraction accuracy (correct attributes/values){RESET}")
print(f"  {DIM}• Affect detection correctness{RESET}")
print(f"  {DIM}• Tool call results{RESET}")
print(f"  {DIM}• Elara's handling of grief context (scenario 5){RESET}")
print(f"  {DIM}• Curiosity/proactive questions appearing in later turns{RESET}")
print()
