import logging
import os
import re
import time

import anthropic

from services.security_utils import public_error_detail
from typing import List, Dict, Any, Optional

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from services.agent_execution import consume_tool_call_budget, execute_agent_tool, tool_result_text
from services.agent_runs import AgentRunRecorder
from services.tools_registry import ToolSnapshot, get_tool_snapshot

logger = logging.getLogger("discord_bot")

CLAUDE_MODEL = ANTHROPIC_MODEL

_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,([A-Za-z0-9+/=\s]+)$")


def _supports_temperature(model: str) -> bool:
    """Fable/Mythos 5 use always-on adaptive thinking and reject temperature."""
    return not model.startswith(("claude-fable-5", "claude-mythos-5"))


def image_input_to_block(src: str) -> Optional[Dict[str, Any]]:
    """Turn one bot-internal image input (data URL or plain http URL — the two
    shapes collect_image_inputs produces) into an Anthropic image block."""
    src = (src or "").strip()
    m = _DATA_URL_RE.match(src)
    if m:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": m.group(1),
                "data": re.sub(r"\s+", "", m.group(2)),
            },
        }
    if src.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": src}}
    return None


def _content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalize a message's content (str or block list) to a block list."""
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": text}] if text else []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _anthropic_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted = []
    for item in tools or []:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if name:
            converted.append(
                {
                    "name": name,
                    "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    return converted


def _block_dict(block: Any) -> Dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    return {"type": "text", "text": str(block)}


def _record_claude_usage(response: Any, selected_model: str) -> None:
    try:
        from services import usage_costs

        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
        }
        usage_costs.record(
            selected_model,
            usage,
            usage_costs.estimate_cost(selected_model, usage),
            label="claude_chat",
        )
    except Exception:
        logger.warning("claude usage recording failed", exc_info=True)


async def generate_claude_response(
    messages: List[Dict[str, Any]],
    model: str | None = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_tools: bool = False,
    allowed_tool_names: Optional[set[str]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
    forced_tool: str | None = None,
    forced_tool_args: Optional[Dict[str, Any]] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
    max_tool_rounds: int | None = None,
    max_tool_seconds: float | None = None,
    tool_call_limits: Optional[Dict[str, int]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Generate a Claude response, optionally using Multivac's shared tools."""
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is missing.")
        return "❌ Error: `ANTHROPIC_API_KEY` is not set in the environment."

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    selected_model = model or CLAUDE_MODEL
    tool_snapshot: ToolSnapshot | None = get_tool_snapshot() if (tools or enable_tools) else None
    active_tools = tool_snapshot.tool_specs() if enable_tools and tools is None else (tools or [])
    if allowed_tool_names is not None:
        active_tools = [
            tool
            for tool in active_tools
            if tool.get("function", {}).get("name") in allowed_tool_names
        ]
    anthropic_tools = _anthropic_tools(active_tools)
    available_tool_names = {tool.get("name") for tool in anthropic_tools}
    bounded_rounds = max(
        1,
        min(int(max_tool_rounds or os.getenv("AGENT_MAX_TOOL_ROUNDS", "4")), 8),
    )
    bounded_seconds = max(
        5.0,
        min(float(max_tool_seconds or os.getenv("AGENT_MAX_TOOL_SECONDS", "45")), 120.0),
    )
    bounded_steps = max(1, min(int(os.getenv("AGENT_MAX_TOOL_STEPS", "8")), 16))
    recorder = (
        AgentRunRecorder(
            provider="anthropic",
            model=selected_model,
            context=tool_context,
            max_steps=bounded_steps,
            metadata={
                "tool_generation": tool_snapshot.generation if tool_snapshot else None,
                "tool_runner": "multivac_shared",
            },
        )
        if anthropic_tools
        else None
    )

    system_prompt_parts = []
    raw_messages = []
    for msg in messages:
        if msg["role"] == "system":
            if msg.get("content"):
                system_prompt_parts.append(msg["content"])
        else:
            raw_messages.append(msg)
    system_prompt = "\n\n".join(system_prompt_parts).strip()
    if anthropic_tools:
        tool_instruction = (
            "You have access to Multivac's existing tools. Actually call an appropriate tool "
            "when the request depends on current web information, repository/private data, or "
            "an attached image's source; do not merely claim you searched. Search-result "
            "snippets are leads, so open a useful page when the answer depends on it. For "
            "reverse-image requests use reverse_image_search, name the provider that ran, and "
            "distinguish exact/partial matches from visual similarity. Never invent a match. "
            "State-changing and billable tools require an explicit request in the current user "
            "message. Finish with a direct answer grounded in the returned evidence."
        )
        if "reverse_image_search" in available_tool_names and tool_call_limits:
            tool_instruction += (
                " For this reverse-image request, use the combined reverse lookup once, use no "
                "more than two targeted web searches afterward, and open a strong candidate page "
                "when possible instead of repeating unsupported keyword guesses."
            )
        system_prompt = f"{system_prompt}\n\n{tool_instruction}" if system_prompt else tool_instruction

    sanitized_messages = []
    for msg in raw_messages:
        role = msg.get("role")
        blocks = _content_blocks(msg.get("content"))
        if not blocks:
            continue
        if not sanitized_messages:
            if role == "user":
                sanitized_messages.append({"role": "user", "content": blocks})
        else:
            prev = sanitized_messages[-1]
            if role == prev["role"]:
                prev["content"].extend(blocks)
            else:
                sanitized_messages.append({"role": role, "content": blocks})

    if not sanitized_messages:
        sanitized_messages.append({"role": "user", "content": [{"type": "text", "text": "Hello."}]})

    try:
        kwargs = {
            "model": selected_model,
            "max_tokens": max_tokens,
            "messages": sanitized_messages,
        }
        if _supports_temperature(selected_model):
            kwargs["temperature"] = temperature
        if system_prompt:
            kwargs["system"] = system_prompt
        original_system = kwargs.get("system")
        prompted_forced_tool = False
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            if forced_tool in available_tool_names:
                if selected_model.startswith(("claude-fable-5-1", "claude-sonnet-5-5", "claude-opus-5-5")):
                    prompted_forced_tool = True
                    kwargs["system"] = (
                        (kwargs.get("system", "") + "\n\n")
                        + f"In your first response, call the {forced_tool} tool before answering."
                    ).strip()
                else:
                    kwargs["tool_choice"] = {"type": "tool", "name": forced_tool}

        response = await client.messages.create(**kwargs)
        if prompted_forced_tool:
            if original_system:
                kwargs["system"] = original_system
            else:
                kwargs.pop("system", None)
        _record_claude_usage(response, selected_model)
        started = time.monotonic()
        step_index = 0
        tool_call_counts: Dict[str, int] = {}
        for round_index in range(bounded_rounds if anthropic_tools else 0):
            uses = [
                block
                for block in (getattr(response, "content", None) or [])
                if getattr(block, "type", None) == "tool_use"
            ]
            if not uses:
                break
            if time.monotonic() - started >= bounded_seconds:
                if tool_trace is not None:
                    tool_trace.append({"name": "agent_limit", "status": "time_limit_reached"})
                break
            sanitized_messages.append(
                {"role": "assistant", "content": [_block_dict(block) for block in response.content]}
            )
            tool_results = []
            for use in uses:
                name = str(getattr(use, "name", "") or "")
                args = dict(getattr(use, "input", None) or {})
                if step_index >= bounded_steps:
                    if tool_trace is not None:
                        tool_trace.append({"name": name, "status": "step_limit_reached"})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": getattr(use, "id", ""),
                            "content": '{"ok":false,"error":"agent_step_limit_reached"}',
                        }
                    )
                    continue
                step_index += 1
                if round_index == 0 and name == forced_tool and forced_tool_args:
                    args = {**args, **forced_tool_args}
                limit_result = consume_tool_call_budget(name, tool_call_limits, tool_call_counts)
                if limit_result is not None:
                    result = limit_result
                    if tool_trace is not None:
                        tool_trace.append({"name": name, "status": "call_limit_reached"})
                    if recorder:
                        recorder.step(
                            step_index=step_index,
                            phase="stop",
                            status="call_limit_reached",
                            tool_name=name,
                            args=args,
                            result=limit_result,
                            error="tool_call_limit_reached",
                        )
                else:
                    result = await execute_agent_tool(
                        name,
                        args,
                        context=tool_context,
                        snapshot=tool_snapshot,
                        recorder=recorder,
                        trace=tool_trace,
                        step_index=step_index,
                    )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(use, "id", ""),
                        "content": tool_result_text(result),
                    }
                )
            sanitized_messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = sanitized_messages
            kwargs.pop("tool_choice", None)
            response = await client.messages.create(**kwargs)
            _record_claude_usage(response, selected_model)

        remaining_uses = [
            block
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "tool_use"
        ]
        if remaining_uses:
            sanitized_messages.append(
                {"role": "assistant", "content": [_block_dict(block) for block in response.content]}
            )
            sanitized_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "The tool limit is reached. Do not call more tools. Return the "
                                "best direct answer supported by evidence already gathered."
                            ),
                        }
                    ],
                }
            )
            final_kwargs = dict(kwargs)
            final_kwargs["messages"] = sanitized_messages
            final_kwargs.pop("tools", None)
            final_kwargs.pop("tool_choice", None)
            response = await client.messages.create(**final_kwargs)
            _record_claude_usage(response, selected_model)

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            suffix = f" ({category})" if category else ""
            if recorder:
                recorder.finish("refusal")
            return f"❌ Claude Fable declined this request{suffix}."
        text_parts = [
            block.text
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        if text_parts:
            if recorder:
                recorder.finish("completed")
            return "\n".join(text_parts)
        if recorder:
            recorder.finish("empty")
        return "❌ Claude returned an empty response."
    except anthropic.APIStatusError as e:
        if recorder:
            recorder.finish("failed")
        logger.error(f"Claude API Error: {e}")
        return f"❌ Claude API error: {public_error_detail(e)}"
    except Exception as e:
        if recorder:
            recorder.finish("failed")
        logger.exception("Unexpected error calling Claude")
        return f"❌ Claude internal error: {public_error_detail(e)}"
