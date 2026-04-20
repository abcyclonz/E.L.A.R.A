"""
Web Search tool server — FastAPI REST interface wrapping Tavily API.

POST /call/search   {"query": "...", "max_results": 5}
GET  /health
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Web Search Tool")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/call/search")
def search(req: SearchRequest):
    if not TAVILY_API_KEY:
        return {"result": "Web search is not configured — TAVILY_API_KEY is missing."}

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": req.query,
        "max_results": req.max_results,
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
        for result in data.get("results", [])[:req.max_results]:
            title = result.get("title", "")
            content = result.get("content", "")[:300]
            parts.append(f"• {title}: {content}")

        return {"result": "\n".join(parts) if parts else "No results found."}

    except httpx.HTTPStatusError as e:
        return {"result": f"Search API error {e.response.status_code}."}
    except Exception as e:
        return {"result": f"Search failed: {e}"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "web_search_tool"}
