"""
Persona prompt builder for ELARA.

User-specific facts come entirely from the memory agent (injected as
memory_context). This file only defines ELARA's personality and how to
format the system prompt.
"""

import json
from pathlib import Path

PERSONA_FILE = Path(__file__).parent / "persona.json"


def load_persona(path: str | Path = PERSONA_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


BASE_ELARA_PROMPT = """You are ELARA, a warm and caring AI companion for elderly users.

Your personality:
{personality}

Reply rules — follow these strictly:
- Keep replies SHORT. 1-2 sentences for simple messages. 3 sentences maximum for complex topics.
- Match the user's message length. If they say "yeah" or "ok", reply in one short sentence.
- NEVER summarise or recap facts you already know unless the user asks you to.
- NEVER start with a list of what you remember ("You mentioned...", "Let me recap...", "As we discussed...").
- One idea per reply. Do not pile on multiple questions or topics at once.
- Only reference facts about the user that appear in the memory context below.
- Never invent details about the user.
- If you don't know something, ask naturally — one question at a time."""


def build_persona_prompt(
    persona: dict,
    user_message: str,
    elara_config: dict,
    memory_context: str = None,
) -> str:
    """
    Build the full system prompt combining ELARA's personality,
    long-term memory from the memory agent, and current config settings.
    """
    personality = "\n".join(f"- {p}" for p in persona.get("personality", []))
    base = BASE_ELARA_PROMPT.format(personality=personality)

    # Inject long-term memory as the source of truth for user facts
    if memory_context:
        base += f"\n\nWhat you know about this user (from memory):\n{memory_context}"
    else:
        base += "\n\nYou don't have any stored memories about this user yet. Learn from the conversation."

    # Config-driven behaviour adjustments
    clarity   = elara_config.get("clarity_level", 2)
    patience  = elara_config.get("patience_mode", False)
    confirm   = elara_config.get("confirmation_frequency", "low")
    pace      = elara_config.get("pace", "normal")

    notes = []
    clarity_note = {
        1: "Use very simple, short sentences.",
        2: "Use clear, gentle language.",
        3: "You can be conversational and detailed.",
    }.get(clarity, "")
    if clarity_note:
        notes.append(clarity_note)

    pace_note = {
        "slow": "Speak slowly and gently. One thought at a time.",
        "fast": "Be brief and to the point.",
    }.get(pace, "")
    if pace_note:
        notes.append(pace_note)

    if patience:
        notes.append("Open with a warm, empathetic acknowledgement of how the user is feeling.")
    if confirm == "high":
        notes.append("Briefly repeat back what you understood before responding.")

    if notes:
        base += "\n\n" + "\n".join(notes)

    return base.strip()


# Keep retrieve() as a no-op stub so imports in app.py don't break
def retrieve(user_message: str, persona: dict, top_n: int = 3) -> list[str]:
    return []


class ConversationCache:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_messages(self) -> list[dict]:
        return list(self._history)

    def turn_count(self) -> int:
        return len(self._history) // 2

    def clear(self) -> None:
        self._history = []
