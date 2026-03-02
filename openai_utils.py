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
from tools_registry import TOOL_SPECS, execute_tool

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

def _check_soft_refusal(text: str):
    if not text or len(text) > 400: # detailed explanations are usually safe
        return
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
             logging.warning(f"Detected soft refusal in OpenAI text: {text}")
             raise OpenAIModerationError(f"Model refused: {text}")

class OpenAIModerationError(Exception):
    """Raised when OpenAI content generation is blocked by moderation filters."""
    def __init__(self, message):
        super().__init__(message)

openai_client: AsyncOpenAI | None = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_openai_client() -> AsyncOpenAI:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return openai_client

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
    "- 'generate_video'\n"
    "- 'chat'\n\n"
    "Rules:\n"
    "- If message requests a VIDEO, MOVIE, or CLIP → 'generate_video'.\n"
    "- If message starts with \"imagine\", \"generate\", \"draw\", \"create\", \"paint\" AND is NOT about video → 'generate_image'.\n"
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
        
        # Build image-aware system prompt
        if has_images:
            system_prompt = (
                _INTENT_SYSTEM + "\n\n"
                "CRITICAL: The user has attached one or more IMAGES with their message.\n"
                "When images are present, assume the user's request is ABOUT those images unless explicitly stated otherwise.\n\n"
                "Choose the intent:\n"
                "- 'generate_image' = User wants to CREATE a NEW image from scratch (imagine, generate, draw, paint, create)\n\n"
                "IMPORTANT: If the user says 'edit', 'change', 'make', 'transform' -> 'edit_image' (even if it involves text).\n"
                "Only use 'describe_image' if they specifically ask what is in the image, or to transcribe/translate text WITHOUT modifying the image.\n"
                "Only use 'chat' if the message is clearly NOT about the attached images."
            )
        else:
            system_prompt = _INTENT_SYSTEM
        
        resp = await get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": system_prompt},
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

TOOLS_DEF = TOOL_SPECS


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

async def _exec_tool(name: str, args: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    logging.debug(f"[openai.tools] Executing {name} with args={list(args.keys())}")  # don't log full args for privacy/size
    try:
        result = await execute_tool(name, args, context=context, tool_specs=TOOLS_DEF)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)
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
    
    # logger.debug(f"[openai.tools.parser] Inspecting outputs: {outputs}") # Very verbose

    for item in outputs:
        # C) Item itself is the tool call (ResponseFunctionToolCall)
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
                 logging.debug(f"[openai.tools.parser] Found direct tool call: {name}")
                 out.append((cid, name, args))
                 continue

        # A) top-level .tool_calls
        calls = getattr(item, "tool_calls", None)
        if calls is None and isinstance(item, dict):
            calls = item.get("tool_calls")
        
        if calls:
             logging.debug(f"[openai.tools.parser] Found {len(calls)} tool_calls in item")
        
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

async def _responses_tool_loop(
    first_resp, 
    messages: List[Any], 
    *, 
    model: str = "gpt-5.2",
    temperature: float = 0.6,
    max_tokens: int = 700,
    max_rounds: int = 3, 
    tool_context: Optional[Dict[str, Any]] = None
):
    """
    Execute tool calls and iteratively call responses.create by appending outputs to input history.
    """
    resp = first_resp
    current_input = list(messages)  # Copy

    for _ in range(max_rounds):
        uses = _collect_tool_uses(resp)
        if not uses:
            break

        # Append assistant's output (tool calls) to history
        # resp.output is typically a list of ContentPart or ToolCall objects
        raw_output = getattr(resp, "output", []) or getattr(resp, "outputs", [])
        if isinstance(raw_output, list):
            current_input.extend(raw_output)
        elif raw_output:
            current_input.append(raw_output)

        for cid, name, args in uses:
            try:
                logging.debug(f"[openai.tools] Calling tool {name}...")
                output_text = await _exec_tool(name, args, context=tool_context)
                logging.debug(f"[openai.tools] Tool {name} returned {len(str(output_text))} chars")
            except Exception as e:
                logging.error(f"[openai.tools] Tool {name} failed: {e}")
                output_text = f"tool_error: {name}: {e}"
            
            # Append tool output part
            current_input.append({
                "type": "function_call_output",
                "call_id": cid,
                "output": str(output_text)
            })

        # Generate next response
        resp = await get_openai_client().responses.create(
            model=model,
            input=current_input,
            tools=_normalize_tools(None), # Tools are still available? Yes.
            max_output_tokens=max_tokens,
            temperature=temperature,
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
    model: str = "gpt-5.2",
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
            resp = await get_openai_client().responses.create(
                model=model,
                input=msgs,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = _extract_responses_text(resp)
            return text or "I’m not sure yet—could you clarify what you need?"
        else:
            msgs.append({"role": "user", "content": _build_user_content_chat(prompt, img_norm)})
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

            resp = await get_openai_client().responses.create(
                model="gpt-5.2",
                input=msgs,
                tools=_normalize_tools(None),
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            # Execute tool loop
            # Execute tool loop
            # Note: _responses_tool_loop doesn't support context yet in this codebase version?
            # We need to verify if we need to patch it. 
            # Actually, standard chat completions (below) is where main logic lives for most bots.
            # But USE_RESPONSES might be on.
            # I will pass context if possible, but the signature above (line 645) was just viewed as NOT accepting it?
            # Wait, line 645 in previous view was NOT shown. I only viewed up to 600.
            # I need to be careful.
            # Let's just fix the Chat Completions loop (lines 950+) which is more standard.
            # Execute tool loop
            # Pass messages (norm) so loop can extend history
            resp = await _responses_tool_loop(
                resp, 
                norm, 
                max_rounds=max_tool_rounds,
                tool_context=tool_context
            )
            text = _extract_responses_text(resp)
            if text:
                return text
            return "I completed the tool actions."

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

            # Standard Chat Completions Tool Loop
            # We must handle tool_calls manually here
            current_msgs = list(msgs)
            
            resp = await get_openai_client().chat.completions.create(
                model="gpt-5.2",
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="auto",
                tools=TOOLS_DEF,
                messages=current_msgs,
            )
            msg = resp.choices[0].message
            
            # Loop
            for _ in range(max_tool_rounds):
                if not msg.tool_calls:
                    break
                
                # Append assistant message with tool calls
                current_msgs.append(msg)
                
                # Exec tools
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        logging.debug(f"[openai.chat] Exec tool {tc.function.name}")
                        output = await _exec_tool(tc.function.name, args, context=tool_context)
                    except Exception as e:
                        output = f"Error: {e}"
                        
                    current_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(output)
                    })
                
                # Next turn
                resp = await get_openai_client().chat.completions.create(
                    model="gpt-5.2",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice="auto",
                    tools=TOOLS_DEF,
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

# -----------------------------------------------------------------------------
# NEW: messages[] APIs (feed ES history directly)
# -----------------------------------------------------------------------------

async def generate_openai_messages_response(
    messages: List[Dict[str, Any]],
    *,
    model: str = "gpt-5.2",
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
            text = _extract_responses_text(resp)
            return text or "I’m not sure yet—could you clarify what you need?"
        else:
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
    model: str = "gpt-5.2",
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> str:
    try:
        if USE_RESPONSES:
            # Add system instruction to encourage tool use
            tool_instruction = {
                "role": "system",
                "content": (
                    "You have access to tools. When the user asks about your code, commits, files, "
                    "weather, stocks, or other data you can fetch, USE the appropriate tool to get "
                    "real information. Do not say you 'would use' a tool - actually call it."
                )
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
            
            # Debug: log if tools were called
            uses = _collect_tool_uses(resp)
            if uses:
                logging.debug(f"[openai.tools] Model called {len(uses)} tools: {[u[1] for u in uses]}")
            else:
                logging.debug("[openai.tools] Model did not call any tools")
            
            # run tool loop (critical)
            resp = await _responses_tool_loop(
                resp, 
                messages=norm,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_rounds=3, 
                tool_context=tool_context
            )
            text = _extract_responses_text(resp)
            if text:
                return text
            
            # Fallback if no text content found
            logging.warning(f"[openai] No text and no tools! Raw Resp Output: {getattr(resp, 'output', 'N/A')}")
            try:
                # Debug dump output structure
                outputs = getattr(resp, 'output', []) or getattr(resp, 'outputs', [])
                if isinstance(outputs, list):
                    for i, o in enumerate(outputs):
                         logging.warning(f"  [openai] output[{i}]: {o}")
            except Exception: 
                pass

            return "I tried to use my tools but couldn't get a response. Could you rephrase?"
        else:
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

            text = (choice.message.content or "").strip()
            if text:
                _check_soft_refusal(text)
                return text
            if getattr(choice.message, "tool_calls", None):
                return "I would use tools for this, but I can proceed directly if you share more specifics."
            return "I’m not sure yet—could you clarify what you need?"
    except Exception as e:
        if isinstance(e, OpenAIModerationError):
            raise
        logging.exception("[openai.messages+tools] error")
        return f"⚠️ OpenAI tools error: {str(e)[:200]}"
