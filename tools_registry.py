"""
Tool registry for model tool-calling.
- No imports from openai_utils here (to avoid circular imports).
- Keep tools fast and deterministic; return JSON-serializable objects.
"""

from typing import Dict, Any
import json
import re

from url_utils import fetch_url_content, extract_main_text, reduce_text_length

# Stock helper is optional; fail gracefully if missing
def _safe_get_quote(ticker: str) -> Dict[str, Any]:
    try:
        from stock_utils import get_realtime_quote  # provided by your latest stock_utils
        return get_realtime_quote(ticker)
    except Exception as e:
        return {"ok": False, "error": f"quote_lookup_failed: {e}"}


# ---- Tool specs (OpenAI function tools shape) ----

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather or a short forecast for a place name or address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Place or address, e.g. 'Raleigh NC'."},
                    "range": {
                        "type": "string",
                        "enum": ["current", "24h", "7d"],
                        "description": "current conditions, next 24 hours, or next 7 days.",
                        "default": "current",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Fetch latest stock price and change for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol, e.g. 'AAPL'."},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_url",
            "description": "Fetch a URL, extract the main article content, and return a condensed text block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to summarize."},
                    "max_len": {
                        "type": "integer",
                        "description": "Max characters of condensed text.",
                        "default": 3000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_youtube_transcript",
            "description": "Return the raw transcript text for a YouTube URL if available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full YouTube URL (watch?v=… or youtu.be/…)"},
                },
                "required": ["url"],
            },
        },
    },
]


# ---- Handlers the bot calls when the model selects a tool ----

async def handle_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    # The weather pipeline is implemented elsewhere during normal command flow.
    # Here we just return a stub signalling to route inside discord_bot.
    loc = (args or {}).get("location")
    rng = (args or {}).get("range", "current")
    if not loc:
        return {"ok": False, "error": "missing 'location'"}
    return {"ok": True, "intent": "get_weather", "location": loc, "range": rng}


async def handle_get_stock_quote(args: Dict[str, Any]) -> Dict[str, Any]:
    ticker = (args or {}).get("ticker")
    if not ticker:
        return {"ok": False, "error": "missing 'ticker'"}
    data = _safe_get_quote(ticker.upper())
    if not isinstance(data, dict) or not data:
        return {"ok": False, "error": "quote_unavailable"}
    return {"ok": True, "data": data, "ticker": ticker.upper()}


YOUTUBE_ID_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_\-]{6,})"
)

def _extract_youtube_id(url: str) -> str | None:
    m = YOUTUBE_ID_RE.search(url or "")
    return m.group(1) if m else None


async def handle_get_youtube_transcript(args: Dict[str, Any]) -> Dict[str, Any]:
    url = (args or {}).get("url", "")
    vid = _extract_youtube_id(url)
    if not vid:
        return {"ok": False, "error": "bad_youtube_url"}
    try:
        from youtube_transcript_api import (
            YouTubeTranscriptApi,
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
        )
        transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        text = " ".join(chunk["text"] for chunk in transcript_list if chunk.get("text"))
        return {"ok": True, "video_id": vid, "text": text}
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return {"ok": False, "error": f"transcript_unavailable: {e.__class__.__name__}"}
    except Exception as e:
        return {"ok": False, "error": f"transcript_error: {e}"}


async def handle_summarize_url(args: Dict[str, Any]) -> Dict[str, Any]:
    url = (args or {}).get("url", "")
    max_len = int((args or {}).get("max_len", 3000))
    if not url or not url.startswith("http"):
        return {"ok": False, "error": "bad_url"}
    try:
        html = fetch_url_content(url)
        title, text = extract_main_text(html)
        condensed = reduce_text_length(text, max_chars=max_len)
        return {
            "ok": True,
            "title": title,
            "condensed": condensed,
            "url": url,
        }
    except Exception as e:
        return {"ok": False, "error": f"fetch_or_extract_failed: {e}"}


TOOL_HANDLERS = {
    "get_weather": handle_get_weather,
    "get_stock_quote": handle_get_stock_quote,
    "summarize_url": handle_summarize_url,
    "get_youtube_transcript": handle_get_youtube_transcript,
}
