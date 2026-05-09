"""
Rigorous multi-run conversation tester for E.L.A.R.A.
Runs N full conversation scripts, scores each turn, and prints a summary report.
No code changes — pure black-box evaluation.
"""

import requests, json, time, sys
from dataclasses import dataclass, field
from typing import Optional

ORCH  = "http://localhost:8003"
MEM   = "http://localhost:8000"
RUNS  = 3          # how many times to repeat the full suite
DELAY = 0.8        # seconds between turns (let LLM breathe)

# ── ANSI colors ───────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; D = "\033[2m"; N = "\033[0m"

def ok(s):  return f"{G}✓{N} {s}"
def fail(s): return f"{R}✗{N} {s}"
def warn(s): return f"{Y}~{N} {s}"

# ── Test turn definition ──────────────────────────────────────────────────────
@dataclass
class Turn:
    msg: str
    speaker: str = "Abhi"
    expect_route: Optional[str] = None          # e.g. "USE_TOOL", "STORE_MEMORY"
    expect_tool:  Optional[str] = None          # e.g. "calendar", "web_search"
    expect_stored: Optional[bool] = None        # True = must store, False = must NOT store
    min_claims:   int = 0                       # minimum claims extracted
    memory_check: Optional[str] = None         # substring that should appear in active_states values
    no_memory:    Optional[str] = None         # substring that must NOT appear (world facts, etc.)
    reply_must_contain: Optional[str] = None   # substring in Elara's reply
    reply_must_not:     Optional[str] = None   # must NOT appear in reply
    note: str = ""

# ── Conversation scripts ──────────────────────────────────────────────────────

SCRIPT_IDENTITY = [
    Turn("hi there",                              expect_route="DIRECT_CHAT",    reply_must_not="frustrated",   note="greeting"),
    Turn("my name is abhi",                       expect_route="STORE_MEMORY",   min_claims=1, memory_check="Abhi", note="name storage"),
    Turn("i live in kerala",                      expect_route="STORE_MEMORY",   min_claims=1, memory_check="Kerala", note="location storage"),
    Turn("what is my name?",                      expect_route="RETRIEVE_MEMORY", memory_check="Abhi",           note="name retrieval"),
    Turn("where do i live?",                      expect_route="RETRIEVE_MEMORY", memory_check="Kerala",         note="location retrieval"),
    Turn("do you remember me?",                   expect_route="RETRIEVE_MEMORY",                                note="memory recall"),
    Turn("my name is abhi",                       expect_stored=False, min_claims=0,                            note="no duplicate storage"),
]

SCRIPT_PEOPLE = [
    Turn("my name is abhi",                       expect_route="STORE_MEMORY",   min_claims=1),
    Turn("i have a neighbour called ravi",        expect_route="STORE_MEMORY",   min_claims=1, memory_check="Ravi", note="new person"),
    Turn("ravi is a retired teacher",             expect_route="STORE_MEMORY",   min_claims=1, memory_check="teacher", note="person attribute"),
    Turn("he loves gardening",                    expect_route="STORE_MEMORY",   min_claims=1, memory_check="garden",   note="pronoun resolution: he=ravi"),
    Turn("who is ravi?",                          expect_route="RETRIEVE_MEMORY", memory_check="Ravi",           note="person retrieval"),
    Turn("does ravi like cooking?",               expect_route="RETRIEVE_MEMORY",                                note="unknown attribute retrieval"),
    Turn("my daughter's name is priya",           expect_route="STORE_MEMORY",   min_claims=1, memory_check="Priya", note="family member"),
    Turn("she lives in bangalore",                expect_route="STORE_MEMORY",   min_claims=1, memory_check="Bangalore", note="pronoun: she=priya"),
    Turn("tell me about priya",                   expect_route="RETRIEVE_MEMORY", memory_check="Priya",          note="family retrieval"),
]

SCRIPT_TOOLS = [
    Turn("what is the capital of france?",        expect_route="USE_TOOL",  expect_tool="web_search",  note="geography → web"),
    Turn("who won the last cricket world cup?",   expect_route="USE_TOOL",  expect_tool="web_search",  note="sports → web"),
    Turn("what's the weather like in kerala?",    expect_route="USE_TOOL",  expect_tool="web_search",  note="weather → web"),
    Turn("can you schedule a meeting on june 5 at 3pm", expect_route="USE_TOOL", expect_tool="calendar", note="calendar tool"),
    Turn("remind me to take my medicine at 8pm",  expect_route="USE_TOOL",  expect_tool="reminder",    note="reminder tool"),
    Turn("set a reminder for my doctor's appointment tomorrow morning", expect_route="USE_TOOL", expect_tool="reminder", note="reminder 2"),
    Turn("what is 2+2?",                          expect_route="DIRECT_CHAT",                            note="simple math → direct"),
    Turn("tell me a joke",                        expect_route="DIRECT_CHAT",                            note="small talk → direct"),
]

SCRIPT_NO_WORLD_FACTS = [
    Turn("my name is abhi",                       expect_route="STORE_MEMORY", min_claims=1),
    Turn("what is the capital of india?",         expect_stored=False, no_memory="capital",  note="world fact not stored"),
    Turn("who is the prime minister of india?",   expect_stored=False, no_memory="prime",    note="political fact not stored"),
    Turn("how old is the taj mahal?",             expect_stored=False,                        note="history not stored"),
    Turn("i am feeling tired today",              expect_stored=False, no_memory="tired",     note="transient emotion not stored"),
    Turn("i feel a bit sad",                      expect_stored=False, no_memory="sad",       note="mood not stored"),
]

SCRIPT_LONG = [
    Turn("hello",                                 expect_route="DIRECT_CHAT"),
    Turn("i am abhi, i am 68 years old",          expect_route="STORE_MEMORY", min_claims=2, memory_check="Abhi"),
    Turn("i live in trivandrum, kerala",          expect_route="STORE_MEMORY", min_claims=1, memory_check="Trivandrum"),
    Turn("i have a son named arjun",              expect_route="STORE_MEMORY", min_claims=1, memory_check="Arjun"),
    Turn("arjun is a software engineer in mumbai", expect_route="STORE_MEMORY", min_claims=1, memory_check="Mumbai"),
    Turn("my wife's name is meena",               expect_route="STORE_MEMORY", min_claims=1, memory_check="Meena"),
    Turn("i enjoy reading malayalam novels",      expect_route="STORE_MEMORY", min_claims=1, memory_check="novel"),
    Turn("do you know the upcoming ipl schedule?",expect_route="USE_TOOL",   expect_tool="web_search"),
    Turn("who is arjun?",                         expect_route="RETRIEVE_MEMORY", memory_check="Arjun"),
    Turn("tell me about my family",               expect_route="RETRIEVE_MEMORY", memory_check="Meena"),
    Turn("my son arjun got promoted last week",   expect_route="STORE_MEMORY", min_claims=1, memory_check="promot"),
    Turn("that makes me very happy",              expect_stored=False, no_memory="happy",    note="emotion not stored"),
    Turn("can you add a reminder for arjun's birthday on june 15?", expect_route="USE_TOOL", expect_tool="reminder"),
    Turn("what do you know about me?",            expect_route="RETRIEVE_MEMORY", memory_check="Abhi"),
    Turn("i also have a cat named whiskers",      expect_route="STORE_MEMORY", min_claims=1, memory_check="whiskers"),
    Turn("she is very playful",                   expect_route="STORE_MEMORY", memory_check="playful", note="pronoun: she=whiskers"),
    Turn("what is the population of trivandrum?", expect_route="USE_TOOL",   expect_tool="web_search"),
    Turn("ok thanks",                             expect_route="DIRECT_CHAT"),
    Turn("do you remember where arjun works?",   expect_route="RETRIEVE_MEMORY", memory_check="Mumbai"),
    Turn("my son also plays chess",              expect_route="STORE_MEMORY", min_claims=1, memory_check="chess"),
]

ALL_SCRIPTS = [
    ("Identity & Retrieval",   SCRIPT_IDENTITY),
    ("People & Pronouns",      SCRIPT_PEOPLE),
    ("Tool Routing",           SCRIPT_TOOLS),
    ("No World Facts",         SCRIPT_NO_WORLD_FACTS),
    ("Long Conversation",      SCRIPT_LONG),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def clear_memory():
    """Wipe all memory between runs."""
    try:
        import subprocess
        subprocess.run(
            ["sudo", "-u", "postgres", "psql", "-d", "memory_db", "-c",
             "TRUNCATE state_memory, belief_memory, event_memory, episodes, memory_logs, topic_frequency RESTART IDENTITY;"],
            capture_output=True, timeout=10
        )
    except Exception as e:
        print(f"  {Y}[warn] Could not clear memory: {e}{N}")


def send(msg: str, speaker: str) -> dict:
    r = requests.post(f"{ORCH}/input",
        json={"text": msg, "speaker": speaker}, timeout=90)
    r.raise_for_status()
    return r.json()


def active_state_values(d: dict) -> str:
    states = d.get("active_states", [])
    return " | ".join(f"{s['entity']}.{s['attribute']}={s['value']}" for s in states)


@dataclass
class TurnResult:
    turn: Turn
    passed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    reply: str = ""
    route: str = ""
    tool: str = ""
    stored: bool = False
    claims: int = 0
    states_str: str = ""

    def score(self): return len(self.passed), len(self.failed)


def evaluate(turn: Turn, resp: dict) -> TurnResult:
    r = TurnResult(turn=turn)
    debug  = resp.get("debug", {})
    route  = debug.get("router_action", "")
    tool   = resp.get("tool_called") or ""
    stored = resp.get("memory_stored", False)
    claims = debug.get("claims_extracted", 0)
    reply  = resp.get("reply", "")
    states = active_state_values(resp)

    r.route = route; r.tool = tool; r.stored = stored
    r.claims = claims; r.reply = reply; r.states_str = states

    def chk(cond, label):
        if cond: r.passed.append(label)
        else:    r.failed.append(label)

    if turn.expect_route:
        chk(route == turn.expect_route, f"route={turn.expect_route}")
    if turn.expect_tool:
        chk(tool == turn.expect_tool, f"tool={turn.expect_tool}")
    if turn.expect_stored is True:
        chk(stored, "memory_stored=True")
    if turn.expect_stored is False:
        chk(not stored, "memory_stored=False")
    if turn.min_claims > 0:
        chk(claims >= turn.min_claims, f"claims≥{turn.min_claims}(got {claims})")
    if turn.memory_check:
        chk(turn.memory_check.lower() in states.lower(), f"memory has '{turn.memory_check}'")
    if turn.no_memory:
        chk(turn.no_memory.lower() not in states.lower(), f"memory lacks '{turn.no_memory}'")
    if turn.reply_must_contain:
        chk(turn.reply_must_contain.lower() in reply.lower(), f"reply contains '{turn.reply_must_contain}'")
    if turn.reply_must_not:
        chk(turn.reply_must_not.lower() not in reply.lower(), f"reply lacks '{turn.reply_must_not}'")

    return r


# ── Runner ────────────────────────────────────────────────────────────────────

def run_script(name: str, turns: list[Turn]) -> tuple[int, int, list[TurnResult]]:
    results = []
    for t in turns:
        try:
            resp = send(t.msg, t.speaker)
            ev = evaluate(t, resp)
        except Exception as e:
            ev = TurnResult(turn=t)
            ev.failed.append(f"request_error: {e}")
        results.append(ev)
        time.sleep(DELAY)

    total_pass = sum(r.score()[0] for r in results)
    total_fail = sum(r.score()[1] for r in results)
    return total_pass, total_fail, results


def print_script_report(name: str, all_run_results: list):
    """all_run_results = list of (pass, fail, [TurnResult]) per run."""
    print(f"\n{B}{'━'*70}{N}")
    print(f"{B}  Script: {name}{N}")
    print(f"{B}{'━'*70}{N}")

    # Aggregate failures across runs
    fail_counts = {}   # turn index → fail labels → count
    num_runs = len(all_run_results)

    for run_idx, (_, _, results) in enumerate(all_run_results):
        for t_idx, r in enumerate(results):
            if t_idx not in fail_counts:
                fail_counts[t_idx] = {}
            for f in r.failed:
                fail_counts[t_idx][f] = fail_counts[t_idx].get(f, 0) + 1

    # Print per-turn summary
    sample_results = all_run_results[0][2]  # use first run for turn list
    for t_idx, r in enumerate(sample_results):
        t = r.turn
        fc = fail_counts.get(t_idx, {})
        consistent_fails = {k: v for k, v in fc.items() if v >= num_runs * 0.6}  # fails 60%+ of runs

        note_str = f" {D}[{t.note}]{N}" if t.note else ""
        msg_str  = t.msg[:50].ljust(52)

        if not fc:
            print(f"  {G}✓{N} {msg_str}{note_str}")
        elif consistent_fails:
            for label, cnt in consistent_fails.items():
                pct = int(cnt / num_runs * 100)
                print(f"  {R}✗{N} {msg_str} → {label} (fails {pct}%){note_str}")
        else:
            print(f"  {Y}~{N} {msg_str} → flaky {note_str}")

    # Overall score
    all_pass = sum(p for p, _, _ in all_run_results)
    all_fail = sum(f for _, f, _ in all_run_results)
    total    = all_pass + all_fail
    pct      = int(all_pass / total * 100) if total else 0
    color    = G if pct >= 80 else Y if pct >= 60 else R
    print(f"\n  Score: {color}{all_pass}/{total} ({pct}%){N} across {num_runs} runs")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{B}{'═'*70}{N}")
    print(f"{B}  E.L.A.R.A Rigorous Test Suite  —  {RUNS} runs × {len(ALL_SCRIPTS)} scripts{N}")
    print(f"{B}{'═'*70}{N}")

    # Quick health check
    try:
        requests.get(f"{ORCH}/health", timeout=5).raise_for_status()
        requests.get(f"{MEM}/health",  timeout=5).raise_for_status()
    except Exception as e:
        print(f"{R}Services not reachable: {e}{N}"); sys.exit(1)

    grand_pass = grand_fail = 0
    script_summaries = []

    for script_name, script_turns in ALL_SCRIPTS:
        all_run_results = []
        print(f"\n{D}Running: {script_name} ({RUNS} runs)...{N}")

        for run_idx in range(RUNS):
            clear_memory()
            time.sleep(1)
            p, f, results = run_script(script_name, script_turns)
            all_run_results.append((p, f, results))
            sys.stdout.write(f"  run {run_idx+1}/{RUNS}: {G}{p}✓{N} {R}{f}✗{N}\n")
            sys.stdout.flush()

        print_script_report(script_name, all_run_results)
        total_p = sum(p for p, _, _ in all_run_results)
        total_f = sum(f for _, f, _ in all_run_results)
        grand_pass += total_p
        grand_fail += total_f
        pct = int(total_p / (total_p + total_f) * 100) if (total_p + total_f) else 0
        script_summaries.append((script_name, pct))

    # Grand summary
    total   = grand_pass + grand_fail
    pct     = int(grand_pass / total * 100) if total else 0
    color   = G if pct >= 80 else Y if pct >= 60 else R

    print(f"\n{B}{'═'*70}{N}")
    print(f"{B}  GRAND SUMMARY{N}")
    print(f"{B}{'═'*70}{N}")
    for sname, spct in script_summaries:
        sc = G if spct >= 80 else Y if spct >= 60 else R
        print(f"  {sc}{spct:>3}%{N}  {sname}")
    print(f"\n  Overall: {color}{grand_pass}/{total} ({pct}%){N}")
    print(f"{B}{'═'*70}{N}\n")


if __name__ == "__main__":
    main()
