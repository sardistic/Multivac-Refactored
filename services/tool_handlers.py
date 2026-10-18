from __future__ import annotations

import re
from typing import Any, Dict

from config import ALLOW_CONTEXT_SEARCH_OTHERS
from services.security_limits import check_rate_limit
from services.security_utils import public_error_detail, sanitize_diagnostic_text
from services.tool_specs import TOOL_SPECS
from services.url_utils import extract_main_text, fetch_url_content, reduce_text_length


def _tool_context(args: Dict[str, Any] | None) -> Dict[str, Any]:
    return dict((args or {}).get("_context") or {})


def _owner_required(args: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not _tool_context(args).get("is_owner"):
        return {"ok": False, "error": "owner_required"}
    return None


def _tool_rate_limit(args: Dict[str, Any] | None, action: str) -> Dict[str, Any] | None:
    ctx = _tool_context(args)
    user_id = ctx.get("user_id")
    guild_id = ctx.get("guild_id")
    if user_id in (None, "") or guild_id in (None, ""):
        return None
    decision = check_rate_limit(action, user_id=user_id, guild_id=guild_id)
    if decision.allowed:
        return None
    return {
        "ok": False,
        "error": "rate_limit_exceeded",
        "retry_after_seconds": decision.retry_after,
    }


def _safe_get_quote(ticker: str) -> Dict[str, Any]:
    try:
        from services.stock_utils import get_realtime_quote

        return get_realtime_quote(ticker)
    except Exception as e:
        return {"ok": False, "error": f"quote_lookup_failed: {public_error_detail(e)}"}


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
    import asyncio

    try:
        from services.search_utils import web_search
    except Exception as e:
        return {"ok": False, "error": f"search_unavailable: {public_error_detail(e)}"}

    q = (args or {}).get("q", "")
    if not q:
        return {"ok": False, "error": "missing 'q'"}
    num = int((args or {}).get("num", 5))
    gl = (args or {}).get("gl")
    lr = (args or {}).get("lr")
    safe = (args or {}).get("safe")
    if bool((args or {}).get("image")):
        try:
            from services.google_search import google_web_search

            return await google_web_search(
                q,
                num=num,
                gl=gl,
                lr=lr,
                safe=safe or "off",
                image=True,
            )
        except Exception as e:
            return {"ok": False, "error": f"keyword_image_search_failed: {e}"}
    return await asyncio.to_thread(
        web_search,
        q,
        max_results=num,
        gl=gl,
        lr=lr,
        safe=safe,
    )


async def handle_reverse_image_search(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.reverse_image_search import reverse_image_search

    ctx = (args or {}).get("_context", {})
    images = list(ctx.get("image_urls") or [])
    source_images = list(ctx.get("source_image_urls") or [])
    try:
        image_index = int((args or {}).get("image_index", 0) or 0)
    except (TypeError, ValueError):
        image_index = 0
    if not images:
        return {
            "ok": False,
            "lookup_type": "reverse_image_search",
            "error": "no_attached_image_in_current_request",
        }
    if image_index < 0 or image_index >= len(images):
        return {
            "ok": False,
            "lookup_type": "reverse_image_search",
            "error": "image_index_out_of_range",
            "available_images": len(images),
        }
    source_image_url = (
        source_images[image_index]
        if image_index < len(source_images)
        and str(source_images[image_index]).startswith(("http://", "https://"))
        else None
    )
    return await reverse_image_search(
        images[image_index],
        public_image_url=source_image_url,
        mode=str((args or {}).get("mode") or "all"),
        max_results=(args or {}).get("max_results", 10),
    )


async def handle_get_agent_run_status(args: Dict[str, Any]) -> Dict[str, Any]:
    import json

    from services.agent_runs import AgentRunStore

    ctx = (args or {}).get("_context", {})
    guild = str(ctx.get("guild_id") or "DM")
    channel = str(ctx.get("channel_id") or ctx.get("conversation_id") or "")
    user_id = str(ctx.get("user_id") or "")
    if not channel or not user_id:
        return {"ok": False, "error": "missing_context_for_agent_status"}
    scope_key = f"{guild}:{channel}:{user_id}"
    requested = str((args or {}).get("run_id") or "").strip()
    try:
        limit = max(1, min(int((args or {}).get("limit", 3)), 10))
    except (TypeError, ValueError):
        limit = 3
    store = AgentRunStore()
    rows = store.recent(scope_key=scope_key, limit=max(limit * 3, 10))
    rows = [row for row in rows if row.get("status") != "running"]
    if requested:
        rows = [row for row in rows if row.get("run_id") == requested]
    rows = rows[:limit]
    runs = []
    for row in rows:
        steps = []
        for step in store.steps(row["run_id"]):
            try:
                result = json.loads(step.get("result_json") or "{}")
            except Exception:
                result = {}
            steps.append(
                {
                    "step": step.get("step_index"),
                    "phase": step.get("phase"),
                    "tool": step.get("tool_name"),
                    "status": step.get("status"),
                    "attempt": step.get("attempt"),
                    "duration_ms": step.get("duration_ms"),
                    "evidence": result,
                    "error": step.get("error"),
                }
            )
        runs.append(
            {
                "run_id": row.get("run_id"),
                "created_at": row.get("created_at"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "intent": row.get("intent"),
                "status": row.get("status"),
                "steps": row.get("step_count"),
                "retries": row.get("retry_count"),
                "approval": row.get("approval_state"),
                "trace": steps,
            }
        )
    return {"ok": True, "scope": "current_conversation_user", "runs": runs}


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
    import asyncio

    from services.youtube_utils import fetch_youtube_transcript

    url = (args or {}).get("url", "")
    vid = _extract_youtube_id(url)
    if not vid:
        return {"ok": False, "error": "bad_youtube_url"}
    limited = _tool_rate_limit(args, "url")
    if limited:
        return limited
    text = await asyncio.to_thread(fetch_youtube_transcript, vid)
    if not text:
        return {"ok": False, "error": "transcript_unavailable"}
    return {"ok": True, "video_id": vid, "text": text[:12000]}


async def handle_summarize_url(args: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio

    url = (args or {}).get("url", "")
    max_len = max(1000, min(int((args or {}).get("max_len", 6000)), 12000))
    if not url or not url.startswith("http"):
        return {"ok": False, "error": "bad_url"}
    limited = _tool_rate_limit(args, "url")
    if limited:
        return limited
    try:
        html = await asyncio.to_thread(fetch_url_content, url)
        title, text = await asyncio.to_thread(extract_main_text, html)
        condensed = reduce_text_length(text, max_chars=max_len)
        if not condensed.strip():
            return {
                "ok": False,
                "error": "no_readable_content",
                "title": title,
                "url": url,
            }
        return {"ok": True, "title": title, "condensed": condensed, "url": url}
    except Exception as e:
        return {"ok": False, "error": f"fetch_or_extract_failed: {public_error_detail(e)}"}


async def handle_git_recent_commits(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import get_recent_commits

    count = int((args or {}).get("count", 10))
    commits = get_recent_commits(count)
    return {"ok": True, "commits": commits}


async def handle_git_commit_diff(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import get_commit_diff

    sha = (args or {}).get("sha", "")
    if not sha:
        return {"ok": False, "error": "missing 'sha'"}
    diff = get_commit_diff(sha)
    return {"ok": True, "diff": diff}


async def handle_git_read_file(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import get_file_content

    path = (args or {}).get("path", "")
    if not path:
        return {"ok": False, "error": "missing 'path'"}
    content = get_file_content(path)
    return {"ok": True, "content": content}


async def handle_git_search_code(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import search_code

    query = (args or {}).get("query", "")
    if not query:
        return {"ok": False, "error": "missing 'query'"}
    results = search_code(query)
    return {"ok": True, "results": results}


async def handle_git_search_history(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import search_history

    query = (args or {}).get("query", "")
    if not query:
        return {"ok": False, "error": "missing 'query'"}
    max_results = int((args or {}).get("max_results", 10))
    results = search_history(query, max_results=max_results)
    return {"ok": True, "results": results}


async def handle_git_file_list(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import get_file_list

    return {"ok": True, "files": get_file_list()}


async def handle_git_repo_info(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import get_repo_info

    return {"ok": True, "info": get_repo_info()}


async def handle_git_find_api_calls(args: Dict[str, Any]) -> Dict[str, Any]:
    denied = _owner_required(args)
    if denied:
        return denied
    from services.git_utils import find_api_calls

    provider = (args or {}).get("provider")
    max_results = int((args or {}).get("max_results", 12))
    return find_api_calls(provider=provider, max_results=max_results)


async def handle_search_memory(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.memory_utils import fetch_matches_recent, search_history_for_context

    ctx = args.get("_context", {})
    guild_id = ctx.get("guild_id")
    channel_id = ctx.get("channel_id")
    user_id = ctx.get("user_id")
    if not (guild_id and channel_id and user_id):
        return {"ok": False, "error": "missing_context_for_memory"}

    query = args.get("query", "")
    limit = max(1, min(int(args.get("limit", 5)), 12))
    requested_target_user = args.get("target_user_id")
    target_user_id = None
    if requested_target_user not in (None, "", user_id, str(user_id)):
        if not (ALLOW_CONTEXT_SEARCH_OTHERS and ctx.get("is_owner")):
            return {"ok": False, "error": "cross_user_memory_search_disabled"}
        target_user_id = str(requested_target_user)

    lowered = query.lower()
    is_temporal_query = bool(
        re.search(
            r"\b("
            r"ago|yesterday|week|month|year|first|earliest|history|"
            r"when did|last time|most recent|recently|previous(?:ly)?"
            r")\b",
            lowered,
        )
    ) or any(k in lowered for k in ["what did i talk", "what did i say", "did i mention", "did you say"])

    if is_temporal_query and query:
        recalled = search_history_for_context(
            guild_id=guild_id,
            channel_id=channel_id,
            user_id=user_id,
            target_user_id=target_user_id,
            query_text=query,
            limit=limit,
            oldest_first=any(k in lowered for k in ["first", "earliest", "start", "beginning"]),
            strict_scope=True,
        )
        if recalled:
            recalled_rows = []
            for line in recalled.splitlines():
                if ": " in line:
                    ts_role, content = line.split(": ", 1)
                    role = "unknown"
                    timestamp = ""
                    if "] " in ts_role:
                        timestamp, role = ts_role.split("] ", 1)
                        timestamp = timestamp.lstrip("[")
                    recalled_rows.append({"role": role, "content": content, "timestamp": timestamp})
            if recalled_rows:
                return {"ok": True, "results": recalled_rows}

    results = fetch_matches_recent(
        guild_id=guild_id,
        channel_id=channel_id,
        user_id=user_id,
        target_user_id=target_user_id,
        query=query,
        size=limit,
        strict_scope=True,
    )
    if not results and query:
        if is_temporal_query:
            recalled = search_history_for_context(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                target_user_id=target_user_id,
                query_text=query,
                limit=limit,
                oldest_first=any(k in lowered for k in ["first", "earliest", "start", "beginning"]),
                strict_scope=True,
            )
            if recalled:
                recalled_rows = []
                for line in recalled.splitlines():
                    if ": " in line:
                        ts_role, content = line.split(": ", 1)
                        role = "unknown"
                        timestamp = ""
                        if "] " in ts_role:
                            timestamp, role = ts_role.split("] ", 1)
                            timestamp = timestamp.lstrip("[")
                        recalled_rows.append({"role": role, "content": content, "timestamp": timestamp})
                if recalled_rows:
                    results = recalled_rows
        # If the user asks a time/recall question and we still have nothing,
        # return recent scoped context so the model can still answer usefully.
        if not results and is_temporal_query:
            results = fetch_matches_recent(
                guild_id=guild_id,
                channel_id=channel_id,
                user_id=user_id,
                target_user_id=target_user_id,
                query="",
                size=limit,
                strict_scope=True,
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
    latest = (ctx.get("latest_user_text") or "").lower()
    explicit_change = bool(
        re.search(
            r"\b(?:from now on|always|permanently|remember to|call me|refer to me|"
            r"change how you|change your (?:tone|style|personality)|reset (?:your )?"
            r"personality)\b",
            latest,
        )
        or re.search(
            r"(?:^|\bplease\s+|\b(?:can|could|would|will) you\s+|\b(?:don't|do not|stop)\s+)"
            r"(?:speak|talk|respond|reply|answer|write|act|be)\b",
            latest,
        )
    )
    clear_request = not (instruction or "").strip() and bool(
        re.search(r"\b(?:reset|clear|remove|forget|stop)\b", latest)
    )
    if not explicit_change and not clear_request:
        return {"ok": False, "error": "latest_message_did_not_authorize_behavior_change"}
    instruction = (instruction or "").strip()[:1000]
    try:
        set_user_instruction(user_id, instruction)
        return {"ok": True, "status": "updated", "instruction": instruction}
    except Exception as e:
        return {"ok": False, "error": f"db_error: {public_error_detail(e)}"}


# Redact anything that looks like a credential before logs reach the model.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|xox[a-z]-[A-Za-z0-9\-]{8,}|AIza[A-Za-z0-9_\-]{10,}|"
    r"(?i:(?:api[_-]?key|token|password|secret)\s*[=:]\s*)\S{6,})"
)


async def handle_read_own_logs(args: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio

    is_owner = bool(_tool_context(args).get("is_owner"))
    lines_cap = 120 if is_owner else 20
    since_cap = 2880 if is_owner else 60
    lines = max(1, min(int(args.get("lines", 40) or 40), lines_cap))
    level = (args.get("level") or "all").lower() if is_owner else "error"
    grep = (args.get("grep") or "").lower()[:100] if is_owner else ""
    since_minutes = max(1, min(int(args.get("since_minutes", 180) or 180), since_cap))

    cmd = [
        "journalctl", "-u", "discordbot", "-n", "600",
        "--since", f"-{since_minutes}min",
        "--no-pager", "-q", "-o", "short",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=15)
    except FileNotFoundError:
        return {"ok": False, "error": "journalctl_not_available"}
    except Exception as e:
        return {"ok": False, "error": f"log_read_failed: {public_error_detail(e)}"}

    if proc.returncode != 0:
        return {"ok": False, "error": "journalctl_failed"}

    rows = out.decode(errors="replace").splitlines()
    if level == "error":
        rows = [r for r in rows if "ERROR" in r or "Traceback" in r or "CRITICAL" in r]
    elif level == "warning":
        rows = [r for r in rows if any(k in r for k in ("ERROR", "WARNING", "Traceback", "CRITICAL"))]
    if grep:
        rows = [r for r in rows if grep in r.lower()]

    if is_owner:
        rows = [
            sanitize_diagnostic_text(_SECRET_RE.sub("[REDACTED]", r), max_chars=500)
            for r in rows[-lines:]
        ]
    else:
        # Public callers get useful failure categories, never raw lines that
        # may contain another user's prompt, identifiers, or provider payload.
        rows = [public_error_detail(r) for r in rows[-lines:]]
    text = "\n".join(rows)
    return {
        "ok": True,
        "scope": "owner_diagnostics" if is_owner else "public_error_summary",
        "lines_returned": len(rows),
        "logs": text[-6000:],
    }


async def handle_remember_fact(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.database_utils import add_user_fact, list_user_facts

    ctx = args.get("_context", {})
    user_id = ctx.get("user_id")
    if not user_id:
        return {"ok": False, "error": "missing_user_context"}

    fact = (args.get("fact") or "").strip()
    if not fact:
        return {"ok": False, "error": "missing_fact"}
    if len(fact) > 300:
        fact = fact[:300]

    # Skip near-duplicates so repeated mentions don't pile up.
    existing = list_user_facts(user_id, limit=100)
    lowered = fact.lower()
    for f in existing:
        if f["fact"].lower() == lowered:
            return {"ok": True, "status": "already_known", "fact": fact}

    fact_id = add_user_fact(user_id, fact, args.get("category"))
    return {"ok": True, "status": "remembered", "id": fact_id, "fact": fact}


async def handle_forget_fact(args: Dict[str, Any]) -> Dict[str, Any]:
    from services.database_utils import delete_user_facts_matching

    ctx = args.get("_context", {})
    user_id = ctx.get("user_id")
    if not user_id:
        return {"ok": False, "error": "missing_user_context"}

    match = (args.get("match") or "").strip()
    if not match:
        return {"ok": False, "error": "missing_match"}
    deleted = delete_user_facts_matching(user_id, match)
    return {"ok": True, "deleted": deleted}


async def handle_list_available_tools(args: Dict[str, Any]) -> Dict[str, Any]:
    return list_tool_summaries()


TOOL_HANDLERS = {
    "web_search": handle_web_search,
    "reverse_image_search": handle_reverse_image_search,
    "get_agent_run_status": handle_get_agent_run_status,
    "get_weather": handle_get_weather,
    "get_stock_quote": handle_get_stock_quote,
    "summarize_url": handle_summarize_url,
    "get_youtube_transcript": handle_get_youtube_transcript,
    "git_recent_commits": handle_git_recent_commits,
    "git_commit_diff": handle_git_commit_diff,
    "git_read_file": handle_git_read_file,
    "git_search_code": handle_git_search_code,
    "git_search_history": handle_git_search_history,
    "git_file_list": handle_git_file_list,
    "git_find_api_calls": handle_git_find_api_calls,
    "git_repo_info": handle_git_repo_info,
    "search_memory": handle_search_memory,
    "update_behavioral_instruction": handle_update_behavioral_instruction,
    "remember_fact": handle_remember_fact,
    "forget_fact": handle_forget_fact,
    "read_own_logs": handle_read_own_logs,
    "list_available_tools": handle_list_available_tools,
}
