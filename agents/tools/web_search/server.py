"""
Web Search MCP Server — wraps Tavily API.

Exposes one tool: search(query, max_results)
Runs as SSE server on port 8010.
"""
from mcp.server.fastmcp import FastMCP
import httpx
import os
import json

mcp = FastMCP("Web Search")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"


@mcp.tool()
def search(query: str, max_results: int = 5) -> str:
    """Search the web for current information, news, weather, or facts."""
    if not TAVILY_API_KEY:
        return "Web search is not configured. Ask the user to provide a TAVILY_API_KEY."

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        r = httpx.post(TAVILY_URL, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()

        parts = []
        if data.get("answer"):
            parts.append(f"Answer: {data['answer']}")

        for result in data.get("results", [])[:max_results]:
            title = result.get("title", "")
            content = result.get("content", "")[:300]
            parts.append(f"• {title}: {content}")

        return "\n".join(parts) if parts else "No results found."

    except httpx.HTTPStatusError as e:
        return f"Search API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"Search failed: {e}"


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)
