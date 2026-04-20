"""
MCP tool client — calls FastMCP SSE servers from the orchestrator.

Each tool server exposes an SSE endpoint at /sse. We open a session,
call the tool, collect the text response, and return it as a plain string.

call_mcp_tool() is synchronous so it works inside FastAPI sync route handlers
(which run in a thread pool, with no existing event loop).
"""
from __future__ import annotations
import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession


async def _call_async(server_url: str, tool_name: str, args: dict) -> str:
    sse_url = server_url.rstrip("/") + "/sse"
    try:
        async with sse_client(url=sse_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                texts = [c.text for c in result.content if hasattr(c, "text")]
                return "\n".join(texts) if texts else "Tool returned no output."
    except Exception as e:
        return f"Tool unavailable ({tool_name}): {e}"


def call_mcp_tool(server_url: str, tool_name: str, args: dict) -> str:
    """Synchronous wrapper — safe to call from any sync context."""
    return asyncio.run(_call_async(server_url, tool_name, args))
