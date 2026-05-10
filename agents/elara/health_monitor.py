"""
health_monitor.py

Rule-based health threshold checks for smartwatch data.
On breach, returns a warm Elara-style proactive message.
Cooldown (10 min per issue per session) prevents spam.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

HR_HIGH       = 110    # BPM — tachycardia concern
HR_LOW        = 50     # BPM — bradycardia concern
TEMP_HIGH     = 37.5   # °C  — fever threshold
TEMP_LOW      = 35.0   # °C  — hypothermia threshold
BATTERY_LOW   = 15     # %   — watch needs charging

ALERT_COOLDOWN = timedelta(minutes=10)

# issue_key → (check fn, message template)
# Templates use {value} for the measured value.
_ISSUE_REGISTRY: dict[str, dict] = {
    "hr_high": {
        "message": (
            "I just noticed your heart rate is reading quite high — {value} BPM. "
            "Are you feeling okay, dear? Maybe sit down for a moment and take a few slow breaths?"
        ),
        "severity": "health",
    },
    "hr_low": {
        "message": (
            "Your heart rate looks a little low at {value} BPM. "
            "How are you feeling? A gentle walk or a warm drink might help."
        ),
        "severity": "health",
    },
    "temp_high": {
        "message": (
            "Your temperature is reading {value}°C, which seems a bit warm. "
            "Are you feeling unwell? Would you like to rest for a bit?"
        ),
        "severity": "health",
    },
    "temp_low": {
        "message": (
            "I see your temperature is on the lower side — {value}°C. "
            "Are you feeling cold? Please make sure you're keeping warm and comfortable."
        ),
        "severity": "health",
    },
    "battery_low": {
        "message": (
            "Just a small heads-up — your watch battery is running low at {value}%. "
            "You might want to give it a little charge when you get the chance!"
        ),
        "severity": "info",
    },
}

# Per-session cooldown state: {session_id: {issue_key: last_alerted_at}}
_last_alerted: dict[str, dict[str, datetime]] = {}


# ── Core check ────────────────────────────────────────────────────────────────

def check_watch_data(
    session_id: str,
    hr: int,
    battery: int,
    steps: int,
    temp: float,
) -> list[dict]:
    """
    Check watch readings against thresholds.
    Returns a list of triggered issues that are past their cooldown window.
    Each item: {"key": str, "message": str, "severity": str}
    """
    now = datetime.utcnow()
    session_cooldowns = _last_alerted.setdefault(session_id, {})

    candidates: list[tuple[str, str]] = []  # (issue_key, formatted_message)

    if hr > HR_HIGH:
        candidates.append(("hr_high", f"{hr}"))
    elif hr > 0 and hr < HR_LOW:
        candidates.append(("hr_low", f"{hr}"))

    if temp > TEMP_HIGH:
        candidates.append(("temp_high", f"{temp:.1f}"))
    elif temp > 0 and temp < TEMP_LOW:
        candidates.append(("temp_low", f"{temp:.1f}"))

    if 0 < battery < BATTERY_LOW:
        candidates.append(("battery_low", f"{battery}"))

    triggered = []
    for key, value in candidates:
        last = session_cooldowns.get(key)
        if last is None or (now - last) >= ALERT_COOLDOWN:
            session_cooldowns[key] = now
            info = _ISSUE_REGISTRY[key]
            triggered.append({
                "key": key,
                "message": info["message"].format(value=value),
                "severity": info["severity"],
            })

    return triggered


def build_proactive_message(issues: list[dict]) -> str:
    """
    Combine triggered issue messages into a single Elara reply.
    Health issues lead; info issues (low battery) trail.
    """
    health  = [i for i in issues if i["severity"] == "health"]
    info    = [i for i in issues if i["severity"] == "info"]
    ordered = health + info

    if not ordered:
        return ""

    if len(ordered) == 1:
        return ordered[0]["message"]

    # Multiple issues — join naturally
    parts = [i["message"] for i in ordered]
    combined = " Also, ".join(parts)
    return combined
