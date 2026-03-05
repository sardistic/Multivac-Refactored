from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from providers.openai_client import OPENAI_CHAT_MODEL, USE_RESPONSES, get_openai_client
from providers.openai_images import (
    build_user_content_chat,
    build_user_content_responses,
    normalize_image_inputs,
)
from services.tools_registry import TOOL_SPECS, execute_tool

REFUSAL_PATTERNS = [
    r"I cannot help you with that",
    r"I can't help you with that",
    r"I cannot provide that information",
    r"I can't provide that information",
    r"I am unable to provide",
    r"I cannot fulfill this request",
    r"I can't fulfill this request",
    r"I'm sorry, I can't help",
    r"I cannot discuss this topic",
]


class OpenAIModerationError(Exception):
    def __init__(self, message):
        super().__init__(message)


def _check_soft_refusal(text: str):
    if not text or len(text) > 400:
        return
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            logging.warning("Detected soft refusal in OpenAI text: %s", text)
            raise OpenAIModerationError(f"Model refused: {text}")


TOOLS_DEF = TOOL_SPECS


def _normalize_tools(tools: list | None) -> list:
    src = tools if tools is not None else TOOLS_DEF
    if not USE_RESPONSES:
        return src
    flat = []
    for t in src:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            flat.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                }
            )
        else:
            flat.append(t)
    return flat


def _extract_responses_text(resp: Any) -> str:
    try:
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        output = getattr(resp, "output", None) or getattr(resp, "outputs", None)
        if not output:
            return ""
        collected: List[str] = []
        for item in output:
            contents = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else [])
            for c in contents or []:
                ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)
                if ctype in ("output_text", "text", "summary_text"):
                    val = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "")
                    if isinstance(val, str) and val.strip():
                        collected.append(val.strip())
        return "\n".join(collected).strip()
    except Exception:
        return ""


def _normalize_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            norm.append({"role": "system", "content": content})
            continue
        parts = content if not isinstance(content, str) else [{"type": "text", "text": content}]
        out_parts = []
        if role == "user":
            for p in parts or []:
                ptype = p.get("type")
                if ptype in ("text", "input_text"):
                    out_parts.append({"type": "input_text", "text": p.get("text", "")})
                elif ptype in ("image_url", "input_image"):
                    url = p.get("image_url", {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
                    out_parts.append({"type": "input_image", "image_url": url})
                else:
                    out_parts.append({"type": "input_text", "text": p.get("text", str(p))})
            norm.append({"role": "user", "content": out_parts})
        elif role == "assistant":
            for p in parts or []:
                out_parts.append({"type": "output_text", "text": p.get("text", str(p))})
            norm.append({"role": "assistant", "content": out_parts})
        else:
            norm.append({"role": "user", "content": [{"type": "input_text", "text": str(content)}]})
    return norm


async def _exec_tool(name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    logging.debug("[openai.tools] Executing %s with args=%s", name, list(args.keys()))
    try:
        result = await execute_tool(name, args, context=context, tool_specs=TOOLS_DEF)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"tool_error: {name}: {e}"


def _collect_tool_uses(r) -> List[tuple[str, str, Dict[str, Any]]]:
    out: List[tuple[str, str, Dict[str, Any]]] = []
    outputs = getattr(r, "output", None) or getattr(r, "outputs", None) or []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for item in outputs:
        itype = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if itype in ("function_call", "function", "tool_call"):
            cid = getattr(item, "call_id", None) or getattr(item, "id", None) or (item.get("call_id") if isinstance(item, dict) else None)
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
            args = getattr(item, "arguments", None) or (item.get("arguments") if isinstance(item, dict) else None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if cid and name:
                out.append((cid, name, args))
                continue
        calls = getattr(item, "tool_calls", None)
        if calls is None and isinstance(item, dict):
            calls = item.get("tool_calls")
        for c in calls or []:
            cid = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
            name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
            args = getattr(c, "arguments", None) or (c.get("arguments") if isinstance(c, dict) else {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if cid and name:
                out.append((cid, name, args))
        contents = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else [])
        for part in contents or []:
            ptype = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
            if ptype in ("tool_use", "tool_call"):
                cid = getattr(part, "id", None) or (part.get("id") if isinstance(part, dict) else None)
                name = getattr(part, "name", None) or (part.get("name") if isinstance(part, dict) else None)
                args = getattr(part, "input", None) or (part.get("input") if isinstance(part, dict) else {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if cid and name:
                    out.append((cid, name, args))
    return out


async def _responses_tool_loop(
    first_resp,
    messages: List[Any],
    *,
    model: str = OPENAI_CHAT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 700,
    max_rounds: int = 3,
    tool_context: Optional[Dict[str, Any]] = None,
):
    resp = first_resp
    current_input = list(messages)
    for _ in range(max_rounds):
        uses = _collect_tool_uses(resp)
        if not uses:
            break
        raw_output = getattr(resp, "output", []) or getattr(resp, "outputs", [])
        if isinstance(raw_output, list):
            current_input.extend(raw_output)
        elif raw_output:
            current_input.append(raw_output)
        for cid, name, args in uses:
            output_text = await _exec_tool(name, args, context=tool_context)
            current_input.append({"type": "function_call_output", "call_id": cid, "output": str(output_text)})
        resp = await get_openai_client().responses.create(
            model=model,
            input=current_input,
            tools=_normalize_tools(None),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    return resp, current_input


async def generate_openai_response(
    prompt: str,
    conversation_id: str,
    user_id: int | str,
    *,
    image_urls: Optional[List[str]] = None,
    context: Optional[str] = None,
    model: str = OPENAI_CHAT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 800,
) -> str:
    try:
        msgs = [{
            "role": "system",
            "content": "You are a helpful, efficient assistant inside Discord. Keep replies concise. If the output is long, summarize tightly.",
        }]
        if context and context.strip():
            msgs.append({"role": "system", "content": f"CONTEXT (trimmed):\n{context[:3800]}"})

        img_norm = normalize_image_inputs(image_urls)
        if USE_RESPONSES:
            msgs.append({"role": "user", "content": build_user_content_responses(prompt, img_norm)})
            resp = await get_openai_client().responses.create(
                model=model,
                input=msgs,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return _extract_responses_text(resp) or "I’m not sure yet—could you clarify what you need?"

        msgs.append({"role": "user", "content": build_user_content_chat(prompt, img_norm)})
        resp = await get_openai_client().chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=msgs,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise OpenAIModerationError("Response blocked by OpenAI content filter.")
        text = (choice.message.content or "").strip()
        _check_soft_refusal(text)
        return text
    except Exception as e:
        if isinstance(e, OpenAIModerationError):
            raise
        logging.exception("[openai.chat] error")
        return f"⚠️ OpenAI error: {str(e)[:200]}"


async def generate_openai_messages_response(
    messages: List[Dict[str, Any]],
    *,
    model: str = OPENAI_CHAT_MODEL,
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> str:
    try:
        if USE_RESPONSES:
            norm = _normalize_messages_for_responses(messages)
            resp = await get_openai_client().responses.create(
                model=model,
                input=norm,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            return _extract_responses_text(resp) or "I’m not sure yet—could you clarify what you need?"

        resp = await get_openai_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise OpenAIModerationError("Response blocked by OpenAI content filter.")
        text = (choice.message.content or "").strip()
        _check_soft_refusal(text)
        return text
    except Exception as e:
        if isinstance(e, OpenAIModerationError):
            raise
        logging.exception("[openai.messages] error")
        return f"⚠️ OpenAI error: {str(e)[:200]}"


async def generate_openai_messages_response_with_tools(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[list] = None,
    tool_context: Optional[Dict[str, Any]] = None,
    model: str = OPENAI_CHAT_MODEL,
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> str:
    try:
        if USE_RESPONSES:
            tool_instruction = {
                "role": "system",
                "content": (
                "You have access to tools. When the user asks about your code, commits, files, "
                "weather, stocks, or other data you can fetch, USE the appropriate tool to get "
                "real information. Do not say you 'would use' a tool - actually call it."
                "If the user asks about past conversation/history/timeframes (e.g., 'what did I say last month', "
                "'2 weeks ago', 'yesterday'), call `search_memory` before answering."
                "For recall questions, prefer semantic/temporal intent over literal keyword matching."
            ),
        }
            messages_with_instruction = [tool_instruction] + messages
            norm = _normalize_messages_for_responses(messages_with_instruction)
            resp = await get_openai_client().responses.create(
                model=model,
                input=norm,
                tools=_normalize_tools(tools),
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            resp, current_input = await _responses_tool_loop(
                resp,
                messages=norm,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_rounds=3,
                tool_context=tool_context,
            )
            text = _extract_responses_text(resp)
            if text:
                return text
            final_resp = await get_openai_client().responses.create(
                model=model,
                input=current_input
                + [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Using the tool results above, answer the user's request directly in plain text. Do not call more tools.",
                            }
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            text = _extract_responses_text(final_resp)
            if text:
                return text
            return "I tried to use my tools but couldn't get a response. Could you rephrase?"

        resp = await get_openai_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or TOOLS_DEF,
            tool_choice="auto",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise OpenAIModerationError("Response blocked by OpenAI content filter.")

        msg = choice.message
        current_msgs = list(messages)
        for _ in range(3):
            if not msg.tool_calls:
                break
            current_msgs.append(msg)
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    output = await _exec_tool(tc.function.name, args, context=tool_context)
                except Exception as e:
                    output = f"Error: {e}"
                current_msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(output)})
            resp = await get_openai_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="auto",
                tools=tools or TOOLS_DEF,
                messages=current_msgs,
            )
            msg = resp.choices[0].message
            if resp.choices[0].finish_reason == "content_filter":
                raise OpenAIModerationError("Response blocked by OpenAI content filter.")
        text = (msg.content or "").strip()
        _check_soft_refusal(text)
        return text
    except Exception as e:
        if isinstance(e, OpenAIModerationError):
            raise
        logging.exception("[openai.tools] error")
        return f"⚠️ OpenAI tools error: {str(e)[:200]}"


async def generate_openai_response_tools(
    prompt: str,
    conversation_id: str,
    user_id: int | str,
    *,
    image_url: Optional[str] = None,
    max_tool_rounds: int = 3,
    context: Optional[str] = None,
    temperature: float = 0.6,
    max_tokens: int = 700,
) -> str:
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful Discord bot. Keep responses succinct but clear. Use tools only if strictly necessary.",
        }
    ]
    if context and context.strip():
        messages.append({"role": "system", "content": f"CONTEXT (trimmed):\n{context[:3800]}"})
    image_list = normalize_image_inputs([image_url] if image_url else None)
    messages.append({"role": "user", "content": build_user_content_chat(prompt, image_list)})
    return await generate_openai_messages_response_with_tools(
        messages,
        tools=TOOLS_DEF,
        tool_context={"conversation_id": conversation_id, "user_id": str(user_id)},
        model=OPENAI_CHAT_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )
