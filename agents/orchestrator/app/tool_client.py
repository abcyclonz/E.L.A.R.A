"""
Tool client — calls the web_search and assistant tool servers over plain HTTP.

Each tool server exposes POST /call/<tool_name> and returns {"result": "string"}.
"""
from __future__ import annotations
import requests


def call_mcp_tool(server_url: str, tool_name: str, args: dict) -> str:
    """POST args to <server_url>/call/<tool_name> and return the result string."""
    url = f"{server_url.rstrip('/')}/call/{tool_name}"
    try:
        r = requests.post(url, json=args, timeout=20)
        r.raise_for_status()
        return r.json().get("result", "Tool returned no output.")
    except requests.exceptions.ConnectionError:
        return f"Tool server unreachable ({tool_name}). Is it running?"
    except requests.exceptions.Timeout:
        return f"Tool call timed out ({tool_name})."
    except Exception as e:
        return f"Tool call failed ({tool_name}): {e}"
