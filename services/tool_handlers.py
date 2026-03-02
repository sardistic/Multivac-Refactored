from __future__ import annotations

import re
from typing import Any, Dict

from services.tool_specs import TOOL_SPECS
from services.url_utils import extract_main_text, fetch_url_content, reduce_text_length


def _safe_get_quote(ticker: str) -> Dict[str, Any]:
    try:
        from services.stock_utils import get_realtime_quote

        return get_realtime_quote(ticker)
    except Exception as e:
        return {"ok": False, "error": f"quote_lookup_failed: {e}"}


def list_tool_summaries(tool_specs=None) -> Dict[str, Any]:
    specs = tool_specs or TOOL_SPECS
    return {
        "tools": [
            {
                "name": t.get("function", {}).get("name"),
                "description": t.get("function", {}).get("description"),
            }
            for t in specs
        ]
    }


async def handle_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    loc = (args or {}).get("location")
    rng = (args or {}).get("range", "current")
    if not loc:
        return {"ok": False, "error": "missing 'location'"}
    return {"ok": True, "intent": "get_weather", "location": loc, "range": rng}


async def handle_web_search(args: Dict[str, Any]) -> Dict[str, Any] | list:
    try:
        from services.search_utils import web_search
    except Exception as e:
        return {"ok": False, "error": f"search_unavailable: {e}"}

    q = (args or {}).get("q", "")
    if not q:
        return {"ok": False, "error": "missing 'q'"}
    num = int((args or {}).get("num", 5))
    gl = (args or {}).get("gl")
    lr = (args or {}).get("lr")
    safe = (args or {}).get("safe")
    return web_search(q, max_results=num, gl=gl, lr=lr, safe=safe)


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
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
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
        return {"ok": True, "title": title, "condensed": condensed, "url": url}
    except Exception as e:
        return {"ok": False, "error": f"fetch_or_extract_failed: {e}"}


async def handle_git_recent_commits(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_recent_commits

    count = int((args or {}).get("count", 10))
    commits = get_recent_commits(count)
    return {"ok": True, "commits": commits}


async def handle_git_commit_diff(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_commit_diff

    sha = (args or {}).get("sha", "")
    if not sha:
        return {"ok": False, "error": "missing 'sha'"}
    diff = get_commit_diff(sha)
    return {"ok": True, "diff": diff}


async def handle_git_read_file(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_file_content

    path = (args or {}).get("path", "")
    if not path:
        return {"ok": False, "error": "missing 'path'"}
    content = get_file_content(path)
    return {"ok": True, "content": content}


async def handle_git_search_code(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import search_code

    query = (args or {}).get("query", "")
    if not query:
        return {"ok": False, "error": "missing 'query'"}
    results = search_code(query)
    return {"ok": True, "results": results}


async def handle_git_file_list(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_file_list

    return {"ok": True, "files": get_file_list()}


async def handle_git_repo_info(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_repo_info

    return {"ok": True, "info": get_repo_info()}


async def handle_search_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.memory_utils import fetch_matches_recent

    ctx = args.get("_context", {})
    guild_id = ctx.get("guild_id")
    channel_id = ctx.get("channel_id")
    user_id = ctx.get("user_id")
    if not (guild_id and channel_id and user_id):
        return {"ok": False, "error": "missing_context_for_memory"}

    query = args.get("query", "")
    limit = int(args.get("limit", 5))
    results = fetch_matches_recent(
        guild_id=guild_id,
        channel_id=channel_id,
        user_id=user_id,
        query=query,
        size=limit,
    )
    return {
        "ok": True,
        "results": [
            {
                "role": r.get("role"),
                "content": r.get("content"),
                "timestamp": r.get("timestamp"),
            }
            for r in results
        ],
    }


async def handle_update_behavioral_instruction(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.database_utils import set_user_instruction

    ctx = args.get("_context", {})
    user_id = ctx.get("user_id")
    if not user_id:
        return {"ok": False, "error": "missing_user_context"}

    instruction = args.get("instruction", "")
    try:
        set_user_instruction(user_id, instruction)
        return {"ok": True, "status": "updated", "instruction": instruction}
    except Exception as e:
        return {"ok": False, "error": f"db_error: {e}"}


async def handle_generate_sora_video(args: Dict[str, Any]) -> Dict[str, Any]:
    from providers.sora_utils import create_sora_job
    from services.database_utils import check_sora_limit, log_sora_usage

    ctx = args.get("_context", {})
    user_id = ctx.get("user_id")
    if not user_id:
        return {"ok": False, "error": "missing_user_context_for_rate_limit"}

    if not check_sora_limit(user_id, limit=2, window_seconds=3600):
        return {
            "ok": False,
            "error": "rate_limit_exceeded",
            "message": "You have reached the limit of 2 Sora videos per hour. Please try again later.",
        }

    prompt = args.get("prompt", "")
    if not prompt:
        return {"ok": False, "error": "missing_prompt"}

    result = await create_sora_job(prompt)
    if result.get("ok"):
        video_id = ((result.get("data") or {}).get("id"))
        log_sora_usage(user_id, video_id=video_id)
        return {"ok": True, "status": "queued", "video_id": video_id, "data": result.get("data")}
    return result


async def handle_list_available_tools(args: Dict[str, Any]) -> Dict[str, Any]:
    return list_tool_summaries()


TOOL_HANDLERS = {
    "web_search": handle_web_search,
    "get_weather": handle_get_weather,
    "get_stock_quote": handle_get_stock_quote,
    "summarize_url": handle_summarize_url,
    "get_youtube_transcript": handle_get_youtube_transcript,
    "git_recent_commits": handle_git_recent_commits,
    "git_commit_diff": handle_git_commit_diff,
    "git_read_file": handle_git_read_file,
    "git_search_code": handle_git_search_code,
    "git_file_list": handle_git_file_list,
    "git_repo_info": handle_git_repo_info,
    "search_memory": handle_search_memory,
    "update_behavioral_instruction": handle_update_behavioral_instruction,
    "list_available_tools": handle_list_available_tools,
    "generate_sora_video": handle_generate_sora_video,
}
