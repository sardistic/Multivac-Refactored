"""Shared bounded tool execution for OpenAI and Anthropic loops."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Awaitable, Callable

from services.agent_runs import AgentRunRecorder
from services.tools_registry import ToolSnapshot, execute_tool


_READ_ONLY_RETRYABLE = {
    "web_search",
    "reverse_image_search",
    "summarize_url",
    "get_weather",
    "get_stock_quote",
    "get_youtube_transcript",
    "git_recent_commits",
    "git_commit_diff",
    "git_read_file",
    "git_search_code",
    "git_search_history",
    "git_file_list",
    "git_find_api_calls",
    "git_repo_info",
    "search_memory",
    "read_own_logs",
    "list_available_tools",
    "get_agent_run_status",
}
_MUTATING_TOOLS = {
    "update_behavioral_instruction": re.compile(r"\b(?:from now on|remember to|always|stop|don't|do not)\b", re.I),
    "remember_fact": re.compile(r"\b(?:remember|save|store|keep)\b", re.I),
    "forget_fact": re.compile(r"\b(?:forget|delete|remove|erase)\b", re.I),
}
_TRANSIENT_RE = re.compile(
    r"(?:timeout|temporar|rate.?limit|429|\b5\d\d\b|connection|unavailable|try again)",
    re.I,
)


def _approval_error(name: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "approval_required",
        "tool": name,
        "message": (
            "This state-changing or billable tool requires an explicit request in the "
            "current user message. Ask the user to confirm the exact action."
        ),
    }


def _is_error(result: Any) -> tuple[bool, str]:
    if isinstance(result, dict) and result.get("ok") is False:
        return True, str(result.get("error") or "tool_failed")
    if isinstance(result, str) and result.lower().startswith(("tool_error", "error:", "❌")):
        return True, result
    return False, ""


async def execute_agent_tool(
    name: str,
    args: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None,
    snapshot: ToolSnapshot,
    recorder: AgentRunRecorder | None,
    trace: list[dict[str, Any]] | None,
    step_index: int,
    max_attempts: int = 2,
    executor: Callable[..., Awaitable[Any]] | None = None,
) -> Any:
    """Execute one tool with approval policy and read-only transient retry."""
    call_args = dict(args or {})
    ctx = dict(context or {})
    approval_pattern = _MUTATING_TOOLS.get(name)
    if approval_pattern is not None:
        request_text = str(ctx.get("request_text") or "")
        if not approval_pattern.search(request_text):
            result = _approval_error(name)
            if recorder:
                recorder.approval("pending_explicit_user_confirmation")
                recorder.step(
                    step_index=step_index,
                    phase="approval",
                    status="blocked",
                    tool_name=name,
                    args=call_args,
                    result=result,
                    error="approval_required",
                )
            if trace is not None:
                trace.append({"name": name, "args": call_args, "status": "approval_required"})
            return result
        if recorder:
            recorder.approval("explicit_current_request")

    attempts = max(1, min(int(max_attempts), 3)) if name in _READ_ONLY_RETRYABLE else 1
    last_result: Any = None
    for attempt in range(1, attempts + 1):
        trace_item = {"name": name, "args": call_args, "status": "running", "attempt": attempt}
        if trace is not None:
            trace.append(trace_item)
        started = time.monotonic()
        try:
            active_executor = executor or execute_tool
            last_result = await active_executor(
                name,
                call_args,
                context=ctx,
                snapshot=snapshot,
            )
            failed, error = _is_error(last_result)
        except Exception as exc:
            last_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            failed, error = True, str(last_result["error"])
        duration_ms = int((time.monotonic() - started) * 1000)
        trace_item.update(
            {
                "status": "failed" if failed else "completed",
                "duration_ms": duration_ms,
            }
        )
        if recorder:
            recorder.step(
                step_index=step_index,
                phase="tool",
                status=trace_item["status"],
                tool_name=name,
                attempt=attempt,
                duration_ms=duration_ms,
                args=call_args,
                result=last_result,
                error=error if failed else None,
            )
        should_retry = failed and attempt < attempts and bool(_TRANSIENT_RE.search(error))
        if not should_retry:
            return last_result
        if recorder:
            recorder.retry()
        await asyncio.sleep(min(0.25 * attempt, 0.5))
    return last_result


def tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def consume_tool_call_budget(
    name: str,
    limits: dict[str, int] | None,
    counts: dict[str, int],
) -> dict[str, Any] | None:
    """Consume one provider-loop call slot or return a structured refusal."""
    if not limits or name not in limits:
        return None
    try:
        limit = max(0, int(limits[name]))
    except (TypeError, ValueError):
        return None
    used = max(0, int(counts.get(name, 0)))
    if used >= limit:
        return {
            "ok": False,
            "error": "tool_call_limit_reached",
            "tool": name,
            "limit": limit,
        }
    counts[name] = used + 1
    return None


__all__ = ["consume_tool_call_budget", "execute_agent_tool", "tool_result_text"]
