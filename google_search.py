# google_search.py
# Async wrapper for Google Programmable Search (Custom Search JSON API)

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Any

import aiohttp

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

API_URL = "https://www.googleapis.com/customsearch/v1"

logger = logging.getLogger("google_search")


class GoogleSearchError(RuntimeError):
    pass


def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _format_results(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    for it in items or []:
        results.append({
            "title": _clean(it.get("title")),
            "url": _clean(it.get("link")),
            "display_link": _clean(it.get("displayLink")),
            "snippet": _clean(it.get("snippet")),
            "mime": _clean(it.get("mime")),
        })
    return results


async def google_web_search(
    query: str,
    *,
    num: int = 5,
    start: int = 1,
    safe: str = "off",   # 'active' | 'off'
    gl: Optional[str] = None,  # country code, e.g. 'us'
    lr: Optional[str] = None,  # language restrict, e.g. 'lang_en'
    image: bool = False,       # if True -> image results
    api_key: Optional[str] = None,
    cse_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return { query, total, results: [{title,url,display_link,snippet,mime}, ...] }
    """
    api_key = api_key or GOOGLE_API_KEY
    cse_id = cse_id or GOOGLE_CSE_ID
    if not api_key or not cse_id:
        raise GoogleSearchError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID.")

    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": max(1, min(int(num), 10)),
        "start": max(1, int(start)),
        "safe": safe,
    }
    if gl:
        params["gl"] = gl
    if lr:
        params["lr"] = lr
    if image:
        params["searchType"] = "image"

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        async with session.get(API_URL, params=params) as resp:
            text = await resp.text()
            if resp.status != 200:
                logger.warning("[google_search] %s", text[:500])
                raise GoogleSearchError(f"HTTP {resp.status}: {text[:200]}")
            data = json.loads(text)

    info = data.get("searchInformation", {})
    total = int(info.get("totalResults", "0") or "0")
    items = _format_results(data.get("items", []) or [])

    return {"query": query, "total": total, "results": items}
