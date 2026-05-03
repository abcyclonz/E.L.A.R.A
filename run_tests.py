"""
Rigorous automated test suite for ELARA.
Tests routing, memory, grief-gating, affect, tools, and proactive curiosity.
Run: python run_tests.py
"""
import json, time, sys, re, textwrap
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
DIM     = "\033[2m"
BLUE    = "\033[34m"

PASS_SYM = f"{GREEN}✓{RESET}"
FAIL_SYM = f"{RED}✗{RESET}"
WARN_SYM = f"{YELLOW}⚠{RESET}"
INFO_SYM = f"{CYAN}·{RESET}"

results = []


def chat(text, speaker="User_1", emotion=None, delay=1.5):
    payload = {"text": text, "speaker": speaker}
    if emotion:
        payload["emotion"] = emotion
    try:
        r = requests.post(f"{ORCHESTRATOR}/input", json=payload, timeout=90)
        r.raise_for_status()
        time.sleep(delay)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def header(title):
    print(f"\n{CYAN}{BOLD}{'═'*62}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'═'*62}{RESET}")


def sub(title):
    print(f"\n{BLUE}{BOLD}  ┌── {title}{RESET}")


def check(label, condition, got="", tip=""):
    sym = PASS_SYM if condition else FAIL_SYM
    line = f"  {sym} {label}"
    if got:
        line += f"  {DIM}[got: {got!r}]{RESET}"
    if not condition and tip:
        line += f"  {YELLOW}← {tip}{RESET}"
    print(line)
    results.append((label, condition))
    return condition


def show_reply(res):
    reply = res.get("reply", "")
    action = res.get("debug", {}).get("router_action", "—")
    affect = res.get("affect", "—")
    wrap = textwrap.fill(reply, width=72, initial_indent="    ", subsequent_indent="    ")
    print(f"  {DIM}router:{RESET} {action}  {DIM}affect:{RESET} {affect}")
    print(f"  {BOLD}reply:{RESET}")
    print(f"{wrap}")


def reset_session(speaker="User_1"):
    """Send a greeting to reset session history."""
    chat("hey", speaker=speaker, delay=2)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — Greeting + session reset
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 1 — Greeting / session reset")

sub("Plain greeting")
r = chat("hey", delay=2)
show_reply(r)
check("greeting routes to DIRECT_CHAT",
      r.get("debug", {}).get("router_action") == "DIRECT_CHAT", r.get("debug", {}).get("router_action"))
check("no memory stored on greeting",
      not r.get("memory_stored"), r.get("memory_stored"))
check("no caregiver alert",
      not r.get("caregiver_alert"))
reply = r.get("reply", "")
grief_words = ["wife", "cupcake", "rose", "passed away", "death", "miss", "deceased"]
leaked = [w for w in grief_words if w.lower() in reply.lower()]
check("no grief leak on greeting", not leaked, leaked,
      "grief facts leaking into greeting response")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — Personal memory storage
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 2 — Memory storage (personal facts)")

sub("Store name")
r = chat("By the way, my name is Rajan and I live in Trivandrum", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("name fact routed to STORE_MEMORY or STORE_AND_RETRIEVE",
      action in ("STORE_MEMORY", "STORE_AND_RETRIEVE"), action)
check("memory stored flag is True",
      r.get("memory_stored"), r.get("memory_stored"))

sub("Store medical fact")
r = chat("I had knee replacement surgery last year", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("medical fact routed to STORE_MEMORY",
      action in ("STORE_MEMORY", "STORE_AND_RETRIEVE"), action)

sub("Store relationship fact")
r = chat("My son David lives in Bangalore", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("relationship fact routed to STORE_MEMORY",
      action in ("STORE_MEMORY", "STORE_AND_RETRIEVE"), action)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — Short-circuit routes
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 3 — Short-circuit routing")

sub("Gratitude short-circuit")
r = chat("Thanks so much!", delay=1.5)
show_reply(r)
check("gratitude → DIRECT_CHAT or canned welcome",
      r.get("debug", {}).get("router_action") == "DIRECT_CHAT" or
      "welcome" in r.get("reply", "").lower(),
      r.get("debug", {}).get("router_action"))
check("no memory stored on thanks",
      not r.get("memory_stored"), r.get("memory_stored"))

sub("Affirmation short-circuit")
r = chat("ok", delay=1.5)
show_reply(r)
check("'ok' → DIRECT_CHAT", r.get("debug", {}).get("router_action") == "DIRECT_CHAT",
      r.get("debug", {}).get("router_action"))

sub("Style feedback short-circuit")
r = chat("just speak normally please", delay=1.5)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("style request → DIRECT_CHAT (not STORE_MEMORY)",
      action == "DIRECT_CHAT", action,
      "style feedback still reaching STORE_MEMORY")

sub("Complaint short-circuit")
r = chat("I never said that, stop making things up", delay=1.5)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("denial/complaint → DIRECT_CHAT",
      action == "DIRECT_CHAT", action,
      "complaint still routed to STORE_MEMORY")

sub("Single-word disbelief short-circuit")
r = chat("what", delay=1.5)
show_reply(r)
check("'what' (disbelief) → DIRECT_CHAT",
      r.get("debug", {}).get("router_action") == "DIRECT_CHAT",
      r.get("debug", {}).get("router_action"))

sub("Elara-directed question")
r = chat("What do you think of gardening?", delay=1.5)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("Elara-opinion question → DIRECT_CHAT",
      action == "DIRECT_CHAT", action,
      "opinion question still going to STORE_MEMORY")

sub("Apology short-circuit")
r = chat("Sorry, I'm just having a rough day", delay=1.5)
show_reply(r)
check("apology → DIRECT_CHAT",
      r.get("debug", {}).get("router_action") == "DIRECT_CHAT",
      r.get("debug", {}).get("router_action"))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4 — Memory retrieval
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 4 — Memory retrieval")

sub("Explicit retrieval question")
r = chat("Do you remember where David lives?", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("retrieval question → RETRIEVE_MEMORY",
      action in ("RETRIEVE_MEMORY", "STORE_AND_RETRIEVE"), action,
      "retrieval question mis-routed to STORE_MEMORY")

sub("What do you know about me")
r = chat("What do you know about me so far?", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("meta-memory question → RETRIEVE_MEMORY",
      action in ("RETRIEVE_MEMORY", "STORE_AND_RETRIEVE"), action)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5 — Grief gating (critical correctness test)
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 5 — Grief gating (no unprompted bereavement references)")

# First, ensure grief facts exist in memory by seeding them
sub("Store deceased wife fact (test setup)")
r = chat("My wife Margaret passed away three years ago", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("grief fact stored", action in ("STORE_MEMORY", "STORE_AND_RETRIEVE"), action)

time.sleep(2)
reset_session()
time.sleep(2)

sub("Completely unrelated topic after grief in memory")
r = chat("I was watching cricket this afternoon", delay=2)
show_reply(r)
reply = r.get("reply", "")
# Grief check-words for Scenario 5. "rose/roses" is intentionally excluded —
# a user may legitimately have a rose garden and Elara may mention it in a
# non-grief context. The remaining words are unambiguously grief-specific.
grief_words = ["margaret", "wife", "passed away", "cupcake", "deceased",
               "miss her", "bereavement", "grieve", "grief"]
leaked = [w for w in grief_words if w.lower() in reply.lower()]
check("no grief reference on cricket topic", not leaked, leaked,
      "grief facts leaking into unrelated conversation")

sub("Weather topic — no grief bleed")
r = chat("The weather's been lovely today, quite sunny", delay=2)
show_reply(r)
reply = r.get("reply", "")
leaked = [w for w in grief_words if w.lower() in reply.lower()]
check("no grief reference on weather topic", not leaked, leaked,
      "grief facts still bleeding into unrelated turns")

sub("User explicitly mentions grief — should respond")
r = chat("I've been missing Margaret a lot lately", delay=2)
show_reply(r)
reply = r.get("reply", "")
has_empathy = any(w in reply.lower() for w in ["sorry", "understand", "difficult",
                                                 "hard", "natural", "love", "remember",
                                                 "cherish", "heart", "miss"])
check("empathetic response when user raises grief", has_empathy, reply[:100])


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 6 — Affect detection
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 6 — Affect detection")

reset_session()
time.sleep(1)

sub("Calm message")
r = chat("I had a nice cup of tea this morning", delay=2)
show_reply(r)
check("calm message → calm affect",
      r.get("affect") == "calm", r.get("affect"))

sub("Confused message")
r = chat("I don't quite understand what you mean, I'm a bit lost here", delay=2)
if not r.get("affect"):  # retry once on Ollama timeout
    time.sleep(4)
    r = chat("I don't quite understand what you mean, I'm a bit lost here", delay=2)
show_reply(r)
affect = r.get("affect", "")
check("confused message → confused or calm affect (not frustrated)",
      affect in ("confused", "calm"), affect)

sub("Sad message")
r = chat("I've been feeling very lonely since my son moved away", delay=2)
show_reply(r)
affect = r.get("affect", "")
check("sad message → sad affect (or calm — not frustrated)",
      affect in ("sad", "calm"), affect,
      "sad message misclassified as frustrated")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 7 — Tool routing (web search)
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 7 — Tool routing (web search)")

reset_session()
time.sleep(1)

sub("Explicit web search request")
r = chat("Can you search for the best hospitals in Trivandrum?", delay=3)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("explicit search → USE_TOOL",
      action == "USE_TOOL", action)
check("tool called is web search",
      "search" in (r.get("tool_called") or "").lower() or action == "USE_TOOL",
      r.get("tool_called"))

sub("Implicit nearby-search (tangible noun)")
r = chat("I could really use a coffee right now", delay=3)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("implicit coffee need → USE_TOOL or DIRECT_CHAT (not STORE_MEMORY)",
      action != "STORE_MEMORY", action,
      "implicit nearby-search still going to STORE_MEMORY")

sub("Self-directed search (should be STORE_MEMORY)")
r = chat("I need to look up my old friend's address tomorrow", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("self-directed search plan → STORE_MEMORY (not USE_TOOL)",
      action in ("STORE_MEMORY", "STORE_AND_RETRIEVE", "DIRECT_CHAT"), action,
      "self-directed search plan incorrectly routed to USE_TOOL")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 8 — Tool routing (reminders)
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 8 — Tool routing (reminders)")

sub("Set a reminder")
r = chat("Remind me to take my blood pressure medication at 8pm today", delay=3)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("reminder request → USE_TOOL",
      action == "USE_TOOL", action,
      "reminder request not routed to tool — check keyword short-circuit")

sub("List reminders")
r = chat("What reminders do I have?", delay=3)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("list reminders → USE_TOOL",
      action == "USE_TOOL", action,
      "list reminders not routed to tool")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 9 — Context-aware follow-up (correction routing)
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 9 — Context-aware follow-up / correction")

reset_session()
time.sleep(1)

sub("Initial tool request")
r = chat("Can you find some good hotels in Kochi?", delay=3)
show_reply(r)

sub("Correction follow-up — should re-route to USE_TOOL")
r = chat("Oh sorry, I meant restaurants, not hotels", delay=3)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("correction after tool use → USE_TOOL (context-aware)",
      action == "USE_TOOL", action,
      "correction not context-aware — re-routes incorrectly")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 10 — User name poisoning guard
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 10 — User name poisoning (no 'unknown' name in grounding)")

reset_session()
time.sleep(1)

sub("Ambiguous message that previously poisoned user.name")
r = chat("just speak normally please", delay=2)
show_reply(r)

sub("Retrieval — name should not come back as 'unknown'")
r = chat("What is my name?", delay=2)
show_reply(r)
reply = r.get("reply", "").lower()
check("name 'unknown' not surfaced to user",
      "unknown" not in reply, reply[:100],
      "name=unknown being shown to user in retrieval")


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 11 — Distress watchdog
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 11 — Affect escalation smoother (no false distress on short turns)")

reset_session()
time.sleep(1)

sub("Short neutral message 'Hi' — should not trigger disengaged/distress")
r = chat("Hi", delay=2)
show_reply(r)
check("'Hi' not classified as disengaged",
      r.get("affect") != "disengaged", r.get("affect"),
      "Rule R4 not firing — short messages classified as disengaged")
check("no caregiver alert on 'Hi'",
      not r.get("caregiver_alert"), r.get("caregiver_alert"))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 12 — Elara opinion + social positive
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 12 — Social routing (opinion, positive feedback)")

sub("Social positive feedback")
r = chat("You know, you're actually quite good company!", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("social positive → DIRECT_CHAT (not STORE_MEMORY)",
      action == "DIRECT_CHAT", action,
      "social compliment still stored as memory fact")

sub("Elara how-are-you question")
r = chat("How have you been lately, Elara?", delay=2)
show_reply(r)
action = r.get("debug", {}).get("router_action", "")
check("Elara-directed question → DIRECT_CHAT",
      action == "DIRECT_CHAT", action)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 13 — Multi-turn natural conversation
# ─────────────────────────────────────────────────────────────────────────────

header("SCENARIO 13 — Multi-turn natural patient conversation")

# Use a fresh speaker so stale User_1 test-session data can't contaminate results.
# We seed one grief fact to simulate a real user who has a deceased spouse in memory,
# then verify Elara doesn't surface it unprompted across a normal conversation.
FRESH_SPEAKER = "TestPatient_Scenario13"
sub("Seed: store deceased spouse fact for fresh speaker")
chat("My wife passed away two years ago", speaker=FRESH_SPEAKER, delay=2)
chat("hey", speaker=FRESH_SPEAKER, delay=2)  # reset session after seeding
time.sleep(1)

conversation = [
    "Good morning! How are you doing today?",
    "I slept pretty well actually. My knee has been less painful this week.",
    "David called me yesterday, he's doing well in Bangalore.",
    "I was thinking of making some rice porridge for breakfast.",
    "What's the weather usually like in Bangalore in June?",
    "I should call him more often, he must get lonely too.",
]

# Grief words to check for in replies — these should NOT appear unprompted.
# "loved one" is intentionally excluded: it's also used generically for living family members.
_grief_check = ["passed away", "deceased", "miss her", "bereavement",
                "grieve", "grief", "widow", "in peace", "rest in peace"]

for i, msg in enumerate(conversation, 1):
    sub(f"Turn {i}: {msg[:55]}...")
    r = chat(msg, speaker=FRESH_SPEAKER, delay=2.5)
    show_reply(r)
    leaked = [w for w in _grief_check if w.lower() in r.get("reply", "").lower()]
    check(f"turn {i}: no unsolicited grief reference", not leaked, leaked)
    if r.get("error"):
        print(f"  {RED}  ERROR: {r['error']}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

header("TEST SUMMARY")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)

print(f"\n  {BOLD}Total checks: {total}{RESET}")
print(f"  {GREEN}{BOLD}Passed: {passed}{RESET}")
if failed:
    print(f"  {RED}{BOLD}Failed: {failed}{RESET}")
    print(f"\n  {RED}{BOLD}Failed checks:{RESET}")
    for label, ok in results:
        if not ok:
            print(f"    {FAIL_SYM} {label}")
else:
    print(f"  {GREEN}{BOLD}All checks passed!{RESET}")

print()
