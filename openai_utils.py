# openai_utils.py
# OpenAI helpers used by discord_bot.py
# - classify_intent(text)
# - image_url_to_base64(url)
# - generate_openai_response(...)
# - generate_openai_response_tools(...)
# - generate_openai_messages_response(messages)
# - generate_openai_messages_response_with_tools(messages, tools=None)
#
# Features:
# - USE_RESPONSES env toggle (OPENAI_USE_RESPONSES=1) to flip to /v1/responses
# - Tool shape normalization for Responses API
# - Robust output parsing for Responses API
# - Safe image normalization (http(s) -> base64 data URL; bare base64 -> data URL)
# - Proper Responses tool loop (exec + submit_tool_outputs on latest resp.id)

from __future__ import annotations

import os
import re
import json
import base64
import logging
from typing import List, Optional, Dict, Any

import aiohttp
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

# Optional tool backends
try:
    from search_utils import web_search as _tool_web_search
except Exception:
    _tool_web_search = None

try:
    from url_utils import fetch_url_content, extract_main_text, reduce_text_length
except Exception:
    fetch_url_content = extract_main_text = reduce_text_length = None

try:
    from weather_utils import get_weather_data as _tool_weather
except Exception:
    _tool_weather = None

try:
    from stock_utils import simple_stock_quote as _tool_stock_quote  # optional
except Exception:
    _tool_stock_quote = None

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Global toggle: flip to the Responses API by setting OPENAI_USE_RESPONSES=1
# -----------------------------------------------------------------------------
USE_RESPONSES = os.getenv("OPENAI_USE_RESPONSES", "").lower() in {"1", "true", "yes", "y", "on"}

# -----------------------------------------------------------------------------
# Intent Classification
# -----------------------------------------------------------------------------

_INTENT_SYSTEM = (
    "You are a fast, lightweight intent classifier.\n"
    "Classify a user's message into one of:\n"
    "- 'edit_image'\n"
    "- 'generate_image'\n"
    "- 'summarize_url'\n"
    "- 'describe_image'\n"
    "- 'get_weather'\n"
    "- 'get_stock'\n"
    "- 'gemini_chat'\n"
    "- 'claude_chat'\n"
    "- 'chat'\n\n"
    "Rules:\n"
    "- If message starts with \"imagine\", \"generate\", \"draw\", \"create\", \"paint\" → 'generate_image'.\n"
    "- If user says \"transparent background\" → 'generate_image'.\n"
    "- If replying to an image and mentions \"change\", \"edit\", \"make transparent\", \"fix\" → 'edit_image'.\n"
    "- Weather words (forecast, rain, snow, temperature) → 'get_weather'.\n"
    "- If a URL is present and they want a summary → 'summarize_url'.\n"
    "- If they ask to describe an image → 'describe_image'.\n"
    "- Stock words or 'stock <TICKER>' → 'get_stock'.\n"
    "- If message starts with 'gemini' and is NOT image generation/editing → 'gemini_chat'.\n"
    "- If message starts with 'claude' or user explicitly asks for 'Claude' → 'claude_chat'.\n"
    "- Else → 'chat'.\n\n"
    "IMPORTANT: Output ONLY ONE label."
)

async def classify_intent(text: str, has_images: bool = False) -> str:
    """Return one of: edit_image | generate_image | summarize_url | describe_image | get_weather | get_stock | gemini_chat | chat."""
    try:
        if not (text or "").strip():
            return "chat"
        
        # If images are present with vague prompts, assume image analysis
        if has_images:
            lower_text = text.lower().strip()
            vague_triggers = ["analyze", "what", "explain", "describe", "tell me about", "look at"]
            if any(trigger in lower_text for trigger in vague_triggers) and len(text.split()) < 10:
                return "describe_image"
        
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": text.strip()},
            ],
        )
        label = (resp.choices[0].message.content or "").strip().lower()
        label = re.sub(r"[^a-z_]", "", label)
        return label if label in {
            "edit_image", "generate_image", "summarize_url",
            "describe_image", "get_weather", "get_stock", "gemini_chat", "claude_chat", "chat"
        } else "chat"
    except Exception as e:
        logging.warning(f"[intent] fallback to chat due to: {e}")
        return "chat"

# -----------------------------------------------------------------------------
# Helper: image URL → base64 (for vision input/editing)
# -----------------------------------------------------------------------------

def _guess_mime_from_bytes(first_bytes: bytes) -> str:
    if first_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if first_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if first_bytes.startswith(b"GIF8"):
        return "image/gif"
    if first_bytes[0:4] in (b"RIFF", b"WEBP"):
        return "image/webp"
    return "image/png"

def _ensure_data_url(s: str, fallback_mime: str = "image/png") -> str:
    st = (s or "").strip()
    if not st:
        return st
    if st.startswith("http://") or st.startswith("https://"):
        return st
    if st.startswith("data:image/"):
        return st
    return f"data:{fallback_mime};base64,{st}"

async def image_url_to_base64(url: str, timeout: int = 15) -> Optional[str]:
    if not url:
        return None
    if url.startswith("data:image/"):
        return url
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", url) and len(url) > 200:
        return _ensure_data_url(url)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers={"User-Agent": "DiscordBot/1.0"}) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type")
                raw = await r.read()
        mime = ctype if ctype and ctype.startswith("image/") else _guess_mime_from_bytes(raw[:16])
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logging.warning(f"[image_url_to_base64] {e}")
        return None

# -----------------------------------------------------------------------------
# Shared tools definition (Chat Completions shape)
# -----------------------------------------------------------------------------

TOOLS_DEF = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for fresh information. Returns top results (title, URL, snippet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "q":   {"type": "string", "description": "Search query"},
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
            "name": "summarize_url",
            "description": "Fetch a URL, extract the main article content, and return condensed text for further summarization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_len": {"type": "integer", "default": 3000}
                },
                "required": ["url"],
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
                    "location": {"type": "string", "description": "Place or address, e.g., 'Raleigh NC'."},
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
                "properties": {"ticker": {"type": "string", "description": "Ticker symbol, e.g., 'AAPL'."}},
                "required": ["ticker"],
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
                "properties": {"url": {"type": "string", "description": "YouTube watch URL or youtu.be short link."}},
                "required": ["url"],
            },
        },
    },
]

# -----------------------------------------------------------------------------
# Tool normalization for Responses API
# -----------------------------------------------------------------------------

def _normalize_tools(tools: list | None) -> list:
    src = tools if tools is not None else TOOLS_DEF
    if not USE_RESPONSES:
        return src
    flat = []
    for t in src:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            flat.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            })
        else:
            flat.append(t)
    return flat

# -----------------------------------------------------------------------------
# Responses API output parsing helpers
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Message normalization for Responses API
# -----------------------------------------------------------------------------

def _normalize_messages_for_responses(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert simple {'role': ..., 'content': <str|parts>} into Responses-compliant shapes:
      - user parts -> 'input_text' / 'input_image'
      - assistant parts -> 'output_text'
      - system may remain as a plain string
    """
    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            # string is fine for system
            norm.append({"role": "system", "content": content})
            continue

        # unify to list of parts
        parts = content
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]

        out_parts = []
        if role == "user":
            for p in parts or []:
                ptype = p.get("type")
                if ptype in ("text", "input_text"):
                    out_parts.append({"type": "input_text", "text": p.get("text", "")})
                elif ptype in ("image_url", "input_image"):
                    # Extract URL from Chat Completions format: {"type": "image_url", "image_url": {"url": "..."}}
                    if ptype == "image_url":
                        # Chat format: nested {"url": ...}
                        url = p.get("image_url", {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
                    else:
                        # Already in input_image format
                        url = p.get("image_url", {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
                    
                    # Responses API expects: {"type": "input_image", "image_url": "direct-string-url"}
                    # NOT {"type": "input_image", "image_url": {"url": "..."}}
                    out_parts.append({"type": "input_image", "image_url": url})
                else:
                    # fallback treat as text
                    out_parts.append({"type": "input_text", "text": p.get("text", str(p))})
            norm.append({"role": "user", "content": out_parts})
        elif role == "assistant":
            for p in parts or []:
                ptype = p.get("type")
                if ptype in ("text", "output_text"):
                    out_parts.append({"type": "output_text", "text": p.get("text", "")})
                else:
                    # any other assistant content becomes output_text
                    out_parts.append({"type": "output_text", "text": p.get("text", str(p))})
            norm.append({"role": "assistant", "content": out_parts})
        else:
            # any other role -> pass through as string user text
            out_parts = [{"type": "input_text", "text": str(content)}]
            norm.append({"role": "user", "content": out_parts})
    return norm

# -----------------------------------------------------------------------------
# Tool execution router
# -----------------------------------------------------------------------------

async def _exec_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "web_search":
            if _tool_web_search is None:
                return "tool_unavailable: web_search backend not configured"
            q = args.get("q", "")
            num = int(args.get("num") or 5)
            rows = _tool_web_search(q, max_results=num)
            return json.dumps(rows, ensure_ascii=False)

        if name == "summarize_url":
            if not (fetch_url_content and extract_main_text and reduce_text_length):
                return "tool_unavailable: summarize_url backend not configured"
            url = args.get("url", "")
            max_len = int(args.get("max_len") or 3000)
            html = fetch_url_content(url)
            title, text = extract_main_text(html)
            condensed = reduce_text_length(text, max_chars=max_len)
            return json.dumps({"title": title, "text": condensed}, ensure_ascii=False)

        if name == "get_weather":
            if _tool_weather is None:
                return "tool_unavailable: weather backend not configured"
            loc = args.get("location", "")
            rng = args.get("range", "current")
            data = _tool_weather(loc, range=rng)  # type: ignore
            return json.dumps(data, ensure_ascii=False)

        if name == "get_stock_quote":
            if _tool_stock_quote is None:
                return "tool_unavailable: stock backend not configured"
            ticker = args.get("ticker", "")
            data = _tool_stock_quote(ticker)  # type: ignore
            return json.dumps(data, ensure_ascii=False)

        if name == "get_youtube_transcript":
            return "tool_unavailable: youtube transcript not configured"

        return f"tool_error: unknown tool '{name}'"
    except Exception as e:
        return f"tool_error: {name}: {e}"

# -----------------------------------------------------------------------------
# Responses tool-call plumbing
# -----------------------------------------------------------------------------

def _collect_tool_uses(r) -> List[tuple[str, str, Dict[str, Any]]]:
    """
    Return [(tool_call_id, name, arguments_dict), ...] from a Responses result.
    Works with SDK objects and plain dicts.
    """
    out: List[tuple[str, str, Dict[str, Any]]] = []

    outputs = getattr(r, "output", None) or getattr(r, "outputs", None) or []
    if not isinstance(outputs, list):
        outputs = [outputs]

    for item in outputs:
        # A) top-level .tool_calls
        calls = getattr(item, "tool_calls", None)
        if calls is None and isinstance(item, dict):
            calls = item.get("tool_calls")
        if calls:
            for c in calls:
                cid  = getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None)
                name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                args = getattr(c, "arguments", None) or (c.get("arguments") if isinstance(c, dict) else {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if cid and name:
                    out.append((cid, name, args))

        # B) content-level parts of type tool_use/tool_call
        contents = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else [])
        for part in contents or []:
            ptype = getattr(part, "type", None) or (part.get("type") if isinstance(part, dict) else None)
            if ptype in ("tool_use", "tool_call"):
                cid  = getattr(part, "id", None)   or (part.get("id")   if isinstance(part, dict) else None)
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

async def _responses_tool_loop(first_resp, *, max_rounds: int = 3):
    """
    Execute tool calls and iteratively submit outputs via responses.submit_tool_outputs
    until the model stops asking for tools or we hit max_rounds.
    Always submit against the *latest* resp.id.
    """
    resp = first_resp
    for _ in range(max_rounds):
        uses = _collect_tool_uses(resp)
        if not uses:
            break

        tool_outputs = []
        for tool_call_id, name, args in uses:
            try:
                output_text = await _exec_tool(name, args)
            except Exception as e:
                output_text = f"tool_error: {name}: {e}"
            tool_outputs.append({"tool_call_id": tool_call_id, "output": str(output_text)})

        # Submit on the current response id
        resp = await openai_client.responses.submit_tool_outputs(
            id=getattr(resp, "id"),
            tool_outputs=tool_outputs,
        )

    return resp

# -----------------------------------------------------------------------------
# Content builders
# -----------------------------------------------------------------------------

def _build_user_content_chat(prompt: str, image_urls: Optional[List[str]] = None) -> Any:
    if image_urls:
        parts = [{"type": "text", "text": prompt}]
        for u in image_urls:
            parts.append({"type": "image_url", "image_url": {"url": u}})
        return parts
    return prompt

def _build_user_content_responses(prompt: str, image_urls: Optional[List[str]] = None) -> Any:
    if image_urls:
        parts = [{"type": "input_text", "text": prompt}]
        for u in image_urls:
            parts.append({"type": "input_image", "image_url": {"url": u}})
        return parts
    return prompt

def _normalize_image_inputs(image_urls: Optional[List[str]]) -> Optional[List[str]]:
    if not image_urls:
        return None
    normed: List[str] = []
    for s in image_urls:
        if not s:
            continue
        if (not s.startswith("http")) and (not s.startswith("data:image/")):
            s = _ensure_data_url(s)
        normed.append(s)
    return normed or None

# -----------------------------------------------------------------------------
# Core: single-turn answer (optionally with vision + context)
# -----------------------------------------------------------------------------

async def generate_openai_response(
    prompt: str,
    conversation_id: str,
    user_id: int | str,
    *,
    image_urls: Optional[List[str]] = None,
    context: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.6,
    max_tokens: int = 800,
) -> str:
    try:
        sys = {
            "role": "system",
            "content": (
                "You are a helpful, efficient assistant inside Discord. Keep replies concise. "
                "If the output is long, summarize tightly."
            ),
        }
        msgs = [sys]
        if context and context.strip():
            msgs.append({"role": "system", "content": f"CONTEXT (trimmed):\n{context[:3800]}"})

        img_norm = _normalize_image_inputs(image_urls)

        if USE_RESPONSES:
            msgs.append({"role": "user", "content": _build_user_content_responses(prompt, img_norm)})
            resp = await openai_client.responses.create(
                model=model,
                input=msgs,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = _extract_responses_text(resp)
            return text or "I’m not sure yet—could you clarify what you need?"
        else:
            msgs.append({"role": "user", "content": _build_user_content_chat(prompt, img_norm)})
            resp = await openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=msgs,
            )
            return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("[openai.chat] error")
        return f"⚠️ OpenAI error: {str(e)[:200]}"

# -----------------------------------------------------------------------------
# Tools path (auto)
# -----------------------------------------------------------------------------

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
    try:
        sys = {
            "role": "system",
            "content": "You are a helpful Discord bot. Keep responses succinct but clear. Use tools only if strictly necessary.",
        }
        msgs: List[Dict[str, Any]] = [sys]
        if context and context.strip():
            msgs.append({"role": "system", "content": f"CONTEXT (trimmed):\n{context[:3800]}"})

        img_list = _normalize_image_inputs([image_url] if image_url else None)

        if USE_RESPONSES:
            if img_list:
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": {"url": img_list[0]}},
                    ],
                })
            else:
                msgs.append({"role": "user", "content": [{"type": "input_text", "text": prompt}]})

            resp = await openai_client.responses.create(
                model="gpt-4o",
                input=msgs,
                tools=_normalize_tools(None),
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            # Execute tool loop
            resp = await _responses_tool_loop(resp, max_rounds=max_tool_rounds)
            text = _extract_responses_text(resp)
            if text:
                return text
            return "I would use tools for this, but I can proceed directly if you share more specifics."
        else:
            if img_list:
                msgs.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_list[0]}},
                    ],
                })
            else:
                msgs.append({"role": "user", "content": prompt})

            resp = await openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="auto",
                tools=TOOLS_DEF,
                messages=msgs,
            )
            msg = resp.choices[0].message
            if msg.content:
                return msg.content.strip()
            if getattr(msg, "tool_calls", None):
                return "I would use tools for this, but I can proceed directly if you share more specifics."
            return "I’m not sure yet—could you clarify what you need?"
    except Exception as e:
        logging.exception("[openai.tools] error")
        return f"⚠️ OpenAI tools error: {str(e)[:200]}"

# -----------------------------------------------------------------------------
# NEW: messages[] APIs (feed ES history directly)
# -----------------------------------------------------------------------------

async def generate_openai_messages_response(
    messages: List[Dict[str, Any]],
    *,
    model: str = "gpt-4o",
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> str:
    try:
        if USE_RESPONSES:
            norm = _normalize_messages_for_responses(messages)
            resp = await openai_client.responses.create(
                model=model,
                input=norm,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            text = _extract_responses_text(resp)
            return text or "I’m not sure yet—could you clarify what you need?"
        else:
            resp = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logging.exception("[openai.messages] error")
        return f"⚠️ OpenAI error: {str(e)[:200]}"

async def generate_openai_messages_response_with_tools(
    messages: List[Dict[str, Any]],
    *,
    tools: Optional[list] = None,
    model: str = "gpt-4o",
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> str:
    try:
        if USE_RESPONSES:
            norm = _normalize_messages_for_responses(messages)
            resp = await openai_client.responses.create(
                model=model,
                input=norm,
                tools=_normalize_tools(tools),
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            # run tool loop (critical)
            resp = await _responses_tool_loop(resp, max_rounds=3)
            text = _extract_responses_text(resp)
            if text:
                return text
            return "I would use tools for this, but I can proceed directly if you share more specifics."
        else:
            resp = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or TOOLS_DEF,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            if msg.content:
                return msg.content.strip()
            if getattr(msg, "tool_calls", None):
                return "I would use tools for this, but I can proceed directly if you share more specifics."
            return "I’m not sure yet—could you clarify what you need?"
    except Exception as e:
        logging.exception("[openai.messages+tools] error")
        return f"⚠️ OpenAI tools error: {str(e)[:200]}"
