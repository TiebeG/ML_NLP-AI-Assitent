# backend/tools_web.py

import os
import requests

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


def web_search(query: str, max_results: int = 5) -> dict:
    """
    Tavily web search (simple + stable).
    Returns:
        {
          "ok": bool,
          "results": [{"title":..., "url":..., "content":...}, ...],
          "error": str|None
        }
    """
    if not TAVILY_API_KEY:
        return {
            "ok": False,
            "results": [],
            "error": "Missing TAVILY_API_KEY environment variable.",
        }

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) or []
        # Normalize fields
        norm = []
        for r in results:
            norm.append(
                {
                    "title": r.get("title", "") or "",
                    "url": r.get("url", "") or "",
                    "content": r.get("content", "") or "",
                }
            )
        return {"ok": True, "results": norm, "error": None}
    except Exception as e:
        return {"ok": False, "results": [], "error": str(e)}
