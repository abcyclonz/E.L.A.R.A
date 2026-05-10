"""
Seed two mock users for testing.
Idempotent — safe to run repeatedly; skips users that already exist.

Usage:
    python scripts/seed_test_users.py                        # native (port 8003)
    ORCHESTRATOR_URL=http://localhost:8001 python scripts/seed_test_users.py  # docker

Credentials printed at the end.
"""

import os
import sys
import json
import urllib.request
import urllib.error

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8003").rstrip("/")

USERS = [
    {
        "email":                    "maggie@elara.dev",
        "password":                 "test1234",
        "full_name":                "Margaret Thompson",
        "age":                      "74",
        "preferred_language":       "English",
        "background":               "Retired school teacher, lives alone in Kochi. Has two adult children who visit on weekends.",
        "interests":                ["gardening", "reading", "cooking", "crossword puzzles"],
        "conversation_preferences": ["speak slowly", "remind me of things I forget", "check in on my health"],
        "technology_usage":         "Basic smartphone user. Comfortable with calls and WhatsApp, unfamiliar with apps.",
        "conversation_goals":       ["stay mentally active", "feel less lonely", "get help with daily reminders"],
        "additional_info":          "Has mild arthritis in both hands. Prefers mornings for conversation.",
    },
    {
        "email":                    "bob@elara.dev",
        "password":                 "test1234",
        "full_name":                "Robert Chen",
        "age":                      "68",
        "preferred_language":       "English",
        "background":               "Retired electrical engineer. Lives with his wife Anita in Trivandrum. Enjoys staying sharp.",
        "interests":                ["chess", "morning walks", "reading the news", "cricket"],
        "conversation_preferences": ["technical depth is fine", "I like facts and data", "no need to repeat things"],
        "technology_usage":         "Comfortable with computers and smartphones. Uses email daily.",
        "conversation_goals":       ["mental stimulation", "track health metrics", "discuss current events"],
        "additional_info":          "Has type-2 diabetes, monitors blood sugar daily. Prefers evenings.",
    },
]


def post(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{ORCHESTRATOR_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def seed():
    print(f"\nSeeding test users → {ORCHESTRATOR_URL}\n")
    results = []

    for u in USERS:
        status, data = post("/auth/signup", u)
        if status == 200:
            print(f"  ✓  Created   {u['full_name']} ({u['email']})")
            results.append((u, "created"))
        elif status == 400 and "already" in str(data).lower():
            print(f"  –  Exists    {u['full_name']} ({u['email']})")
            results.append((u, "exists"))
        else:
            print(f"  ✗  Failed    {u['full_name']} ({u['email']})  →  {status}: {data}")
            results.append((u, "failed"))

    print()
    print("━" * 52)
    print("  TEST CREDENTIALS")
    print("━" * 52)
    for u, state in results:
        if state != "failed":
            print(f"  Name     : {u['full_name']}")
            print(f"  Email    : {u['email']}")
            print(f"  Password : {u['password']}")
            print()
    print("━" * 52)

    failed = [u for u, s in results if s == "failed"]
    if failed:
        print(f"\n  {len(failed)} user(s) failed — check orchestrator logs.")
        sys.exit(1)


if __name__ == "__main__":
    seed()
