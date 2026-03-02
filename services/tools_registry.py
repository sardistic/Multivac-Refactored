"""
Tool registry for model tool-calling.
- No imports from openai_utils here (to avoid circular imports).
- Keep tools fast and deterministic; return JSON-serializable objects.
"""

from typing import Dict, Any, Optional
import json
import re

from services.url_utils import fetch_url_content, extract_main_text, reduce_text_length

# Stock helper is optional; fail gracefully if missing
def _safe_get_quote(ticker: str) -> Dict[str, Any]:
    try:
        from services.stock_utils import get_realtime_quote  # provided by your latest stock_utils
        return get_realtime_quote(ticker)
    except Exception as e:
        return {"ok": False, "error": f"quote_lookup_failed: {e}"}


def _list_tool_summaries(tool_specs=None) -> Dict[str, Any]:
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


# ---- Tool specs (OpenAI function tools shape) ----

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for fresh information. Returns top results (title, URL, snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "num": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "safe": {"type": "string", "enum": ["off", "active"], "default": "off"},
                    "gl": {"type": "string", "description": "Country code, e.g., 'us'"},
                    "lr": {"type": "string", "description": "Language restrict, e.g., 'lang_en'"},
                    "image": {"type": "boolean", "description": "Image search", "default": False},
                },
                "required": ["q"],
            },
        },
    },
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
    # ---- Git self-awareness tools ----
    {
        "type": "function",
        "function": {
            "name": "git_recent_commits",
            "description": "Get my recent git commits. Use this to answer questions about what I've changed recently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits to fetch (max 50)", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit_diff",
            "description": "Get the diff/changes for a specific commit by SHA. Use after git_recent_commits to see details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sha": {"type": "string", "description": "Commit SHA (short or full)"},
                },
                "required": ["sha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_read_file",
            "description": "Read content of one of my source files. Use to explain my own code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root, e.g. 'discord_bot.py'"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_search_code",
            "description": "Search my codebase for a pattern. Returns matching lines with file and line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (case-insensitive)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_file_list",
            "description": "List all files in my repository.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_repo_info",
            "description": "Get basic info about my repository: branch, remote, last commit.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search my long-term memory (Elasticsearch) for past conversations or context. Use this to remember things the user told you previously.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (e.g. 'favorite pokemon', 'project ideas')"},
                    "limit": {"type": "integer", "description": "Max results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_behavioral_instruction",
            "description": "Update your long-term behavioral instructions for the current user. Use this when the user asks you to change how you speak, behave, or interact with them permanently (e.g. 'always speak in uwu', 'be sassy', 'call me Captain').",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "The full behavioral instruction to store. e.g. 'Always answer in 1920s slang.' Set to empty string to clear."
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_tools",
            "description": "List all my available tools and what they do. Call this to see what capabilities I have.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sora_video",
            "description": "Generate a video using OpenAI Sora. STRICT LIMIT: 2 videos per user per hour.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the video to generate."
                    }
                },
                "required": ["prompt"],
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


# ---- Git tool handlers ----

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
    files = get_file_list()
    return {"ok": True, "files": files}


async def handle_git_repo_info(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.git_utils import get_repo_info
    info = get_repo_info()
    return {"ok": True, "info": info}


async def handle_search_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    # We need to access memory_utils. We'll import it here.
    # Note: We need the context (guild_id, etc.) which usually comes from the bot state.
    # However, these handlers only take `args`.
    # This is a limitation of the current tool system designed for stateless generic tools.
    #
    # WORKAROUND: The `args` should definitely include context if the model puts it there,
    # but the model doesn't know the IDs.
    #
    # Better approach for context-aware tools:
    # The `args` passed to `handle_...` are purely from the model.
    # We might need to inject context into `args` *before* calling the handler in `_exec_tool`?
    # Or `_exec_tool` needs to support context.
    #
    # Looking at `openai_utils._exec_tool`, it just takes `name` and `args`.
    # And `discord_bot` calls `generate...` which calls `_responses_tool_loop`.
    #
    # For now, `search_memory` requires guild/channel/user context.
    # Since we can't easily change the function signature of all handlers without breaking things,
    # we will fail gracefully if context isn't passed, OR we update `_exec_tool` to accept context.
    #
    # PREFERRED FIX: The bot should probably inject `_context` into the args if possible.
    # But for now, let's see if we can make `search_memory` take arguments that the model MIGHT know?
    # No, model doesn't know guild_id.
    #
    # Solution: We will update `openai_utils.py` to allow passing a `context` dict to `_exec_tool`.
    # But that requires updating the loop.
    #
    # SIMPLEST FIX for this session:
    # We'll rely on a global or contextvar? No, that's messy.
    #
    # Let's look at `openai_utils.py` call site.
    # `discord_bot.py` calls `generate_openai_messages_response_with_tools`.
    # It doesn't pass context to that function either, except via 'messages'.
    #
    # WAIT! `discord_bot.py` has the context variables `message`, `guild_id`, `user_id`.
    # But the tool execution happens deep inside `openai_utils.py`.
    #
    # I will modify `generate_openai_messages_response_with_tools` to accept `tool_context`.
    # And pass that down.
    
    # For this file modification, I'll just write the handler assuming `_context` might be in args
    # (injected by the caller) or we'll fail.
    # Actually, let's just write the handler and then fix the plumbing in `openai_utils`.
    from services.memory_utils import fetch_matches_recent
    
    # Check if context was injected
    ctx =  args.get("_context", {})
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
        size=limit
    )
    
    # Format for the model
    formatted = []
    for r in results:
        formatted.append({
            "role": r.get("role"),
            "content": r.get("content"),
            "timestamp": r.get("timestamp")
        })
        
    return {"ok": True, "results": formatted}


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

    # Rate Check (2 per hour = 3600s)
    if not check_sora_limit(user_id, limit=2, window_seconds=3600):
         return {
             "ok": False, 
             "error": "rate_limit_exceeded", 
             "message": "You have reached the limit of 2 Sora videos per hour. Please try again later."
         }

    prompt = args.get("prompt", "")
    if not prompt:
        return {"ok": False, "error": "missing_prompt"}

    result = await create_sora_job(prompt)
    if result.get("ok"):
        video_id = ((result.get("data") or {}).get("id"))
        log_sora_usage(user_id, video_id=video_id)
        return {
            "ok": True,
            "status": "queued",
            "video_id": video_id,
            "data": result.get("data"),
        }
    return result


async def handle_list_available_tools(args: Dict[str, Any]) -> Dict[str, Any]:
    return _list_tool_summaries()


TOOL_HANDLERS = {
    "web_search": handle_web_search,
    "get_weather": handle_get_weather,
    "get_stock_quote": handle_get_stock_quote,
    "summarize_url": handle_summarize_url,
    "get_youtube_transcript": handle_get_youtube_transcript,
    # Git tools
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


async def execute_tool(name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None, tool_specs=None):
    if name == "list_available_tools":
        return _list_tool_summaries(tool_specs=tool_specs)

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown_tool: {name}"}

    call_args = dict(args or {})
    if context:
        call_args["_context"] = context
    return await handler(call_args)
