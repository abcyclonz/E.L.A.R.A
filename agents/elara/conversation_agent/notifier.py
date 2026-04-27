"""
conversation_agent/notifier.py
================================
Caregiver alert dispatcher for the distress watchdog.

Uses Gmail SMTP with an App Password — no third-party service required.

Required env vars (set in .env and passed through docker-compose):
  GMAIL_APP_PASSWORD  — 16-char App Password from Google Account → Security → App Passwords
  ALERT_FROM_EMAIL    — Gmail address you're sending FROM (e.g. yourname@gmail.com)
  ALERT_TO_EMAIL      — Caregiver address(es), comma-separated

If any var is missing the alert degrades gracefully to log.warning.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.text import MIMEText

log = logging.getLogger(__name__)

_GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
_ALERT_FROM = os.getenv("ALERT_FROM_EMAIL", "")
_ALERT_TO_RAW = os.getenv("ALERT_TO_EMAIL", "")

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_caregiver_alert(session_id: str, distress_turns: int) -> None:
    """
    Send a caregiver alert email for the given session.
    Called once when consecutive non-calm turns first hit DISTRESS_TURN_LIMIT.
    """
    if not (_GMAIL_APP_PASSWORD and _ALERT_FROM and _ALERT_TO_RAW):
        log.warning(
            "[CAREGIVER ALERT] Session %s — %d consecutive distress turns. "
            "Set GMAIL_APP_PASSWORD / ALERT_FROM_EMAIL / ALERT_TO_EMAIL to enable email alerts.",
            session_id,
            distress_turns,
        )
        return

    recipients = [addr.strip() for addr in _ALERT_TO_RAW.split(",") if addr.strip()]

    body = (
        f"ELARA has detected a distress episode.\n\n"
        f"Session ID      : {session_id}\n"
        f"Non-calm turns  : {distress_turns} consecutive\n\n"
        f"Please check on the user as soon as possible.\n\n"
        f"— ELARA Watchdog"
    )

    msg = MIMEText(body)
    msg["Subject"] = f"[ELARA] Distress Alert — Session {session_id}"
    msg["From"] = _ALERT_FROM
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(_ALERT_FROM, _GMAIL_APP_PASSWORD)
            smtp.sendmail(_ALERT_FROM, recipients, msg.as_string())
        log.info("[CAREGIVER ALERT] Email sent for session %s.", session_id)
    except Exception as exc:
        log.error("[CAREGIVER ALERT] Failed to send email for session %s: %s", session_id, exc)
