# discord_bot.py
# Full Discord bot with:
# - Mentions/replies trigger
# - Intent classification
# - URL summarize / weather / stock routed to existing utilities
# - Image generation & editing
# - Chat uses ES-backed messages[] window (per-user continuous context)
# - Live progress bar + expand/collapse UI
# - Timeline-aware system prompt
# - "memory fetch more" now fetches RECENT messages only (after last-seen id)
# - NEW: Image-describe injection, reply-reference, tool-nudging
# - FIX: Google CSE keys pulled via config (metadata → env), search fast-path is gated

from __future__ import annotations

import os
import re
import sys
import json
import logging
import mimetypes
import asyncio
import contextlib
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime, timezone

import discord
from discord.ext import commands

# IMPORTANT: importing config mirrors GCE metadata → os.environ for all keys
# Importing GOOGLE_* here ensures that side-effect runs even if this file
# doesn't directly use the variables. (We also log a redacted presence check.)
from config import DISCORD_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID

# Memory / Elasticsearch helpers
from memory_utils import (
    index_message,                  # (message_id, guild_id, channel_id, user_id, role, content, timestamp?, reply_to_id?)
    build_message_window,           # (guild_id, channel_id, user_id, limit_msgs=24) -> List[{role, content}]
    build_timeline_prompt_block,    # timeline block string (newest→oldest)
)

# OpenAI helpers
from openai_utils import (
    classify_intent,
    image_url_to_base64,
    generate_openai_messages_response_with_tools,
    TOOLS_DEF,
)

# Optional features (your existing utilities)
from stability_utils import handle_image_generation, edit_image_with_prompt
from gemini_utils import generate_gemini_image
from weather_utils import get_location_details, get_weather_data, handle_weather_request, format_weather_response
from url_utils import fetch_url_content, extract_main_text, reduce_text_length
from progress import start_progress_bar
from database_utils import save_message_expansion, get_message_expansion, set_message_expanded

# NEW: direct search fast-path (kept, but now properly gated)
try:
    from search_utils import web_search
except Exception:
    web_search = None

# Streaming niceties (optional)
try:
    from stream_utils import ThrottledEditor
    STREAM_OK = True
except Exception:
    STREAM_OK = False

# ---- Logging ----
logger = logging.getLogger("discord_bot")
if "--verbose" in sys.argv:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

def _redact(s: Optional[str], keep: int = 6) -> str:
    if not s:
        return "(missing)"
    if len(s) <= keep:
        return "*" * (len(s) - 1) + s[-1:]
    return ("*" * (len(s) - keep)) + s[-keep:]

# Log key presence (redacted) once so we can see config->metadata->env worked
logger.info(
    "Google CSE keys: GOOGLE_API_KEY=%s, GOOGLE_CSE_ID=%s",
    _redact(GOOGLE_API_KEY), _redact(GOOGLE_CSE_ID)
)

# ---- Discord ----
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ---- Expand/Collapse UI ----
LINE_TRUNCATE_AT = 2
EXPAND_EMOJI = "🧾"
COLLAPSE_EMOJI = "🔼"

# ---- Local state for backfill high-water marks (per guild:channel) ----
STATE_FILE = "/mnt/data/memory_state.json"
os.makedirs("/mnt/data", exist_ok=True)
if not os.path.exists(STATE_FILE):
    with open(STATE_FILE, "w") as f:
        json.dump({}, f)

def _load_state() -> Dict[str, Any]:
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(data: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

def _state_key(guild_id: int | None, channel_id: int) -> str:
    g = str(guild_id) if guild_id else "DM"
    return f"{g}:{channel_id}"

# --------------------------
# Helpers
# --------------------------

def is_probably_image(url: str) -> bool:
    path = urlparse(url).path
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith("image/"))

def make_preview(full_text: str, max_lines: int = LINE_TRUNCATE_AT):
    lines = full_text.splitlines()
    if len(lines) > max_lines:
        preview = "\n".join(lines[:max_lines]).rstrip()
        
        # Check for unclosed code blocks
        # Count occurrences of triple backticks in the preview
        code_fence_count = preview.count("```")
        if code_fence_count % 2 != 0:
            # Odd number means one is open. Close it.
            preview += "\n```"
            
        return preview + "…", True
    return full_text, False

async def send_or_edit_with_truncation(
    full_text: str, *, channel: Optional[discord.abc.Messageable] = None,
    target_msg: Optional[discord.Message] = None, reply_to: Optional[discord.Message] = None
):
    """Send or edit a message with 2-line preview and reactions to expand/collapse."""
    if not isinstance(full_text, str):
        full_text = str(full_text)

    preview, did_trunc = make_preview(full_text, LINE_TRUNCATE_AT)

    if did_trunc:
        content = f"{preview}\n\n(react {EXPAND_EMOJI} to expand)"
        if target_msg:
            await target_msg.edit(content=content)
            save_message_expansion(target_msg.id, full_text, expanded=False)
            with contextlib.suppress(Exception):
                await target_msg.clear_reactions()
            with contextlib.suppress(Exception):
                await target_msg.add_reaction(EXPAND_EMOJI)
            return target_msg
        else:
            sent = await channel.send(content, reference=reply_to)
            save_message_expansion(sent.id, full_text, expanded=False)
            with contextlib.suppress(Exception):
                await sent.add_reaction(EXPAND_EMOJI)
            return sent
    else:
        if target_msg:
            await target_msg.edit(content=full_text)
            with contextlib.suppress(Exception):
                await target_msg.clear_reactions()
            save_message_expansion(target_msg.id, full_text, expanded=True)
            return target_msg
        else:
            return await channel.send(full_text, reference=reply_to)

async def live_status_with_progress(
    message: discord.Message, *, action_label: str, emoji: str, coro, duration_estimate: int, summarizer=None
):
    """Post a status line, run a progress bar alongside the task, optionally live-summarize."""
    status_msg = await message.reply(f"[{emoji} {action_label} ░░░░░░░░░░]")

    loop = asyncio.get_event_loop()
    task = loop.create_task(coro)
    progress_task = loop.create_task(
        start_progress_bar(status_msg, task, action_label=action_label, emoji=emoji, duration_estimate=duration_estimate)
    )

    stop_summary = asyncio.Event()
    summary_task = None

    async def _summary_loop():
        if not STREAM_OK or summarizer is None:
            return
        editor = ThrottledEditor(status_msg, min_interval_s=1.5, max_len=1300)
        while not task.done():
            try:
                s = summarizer()
                if s:
                    await editor.update(f"[{emoji} {action_label} ░░░░░░░░░░]\n{s}")
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop_summary.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                continue

    if summarizer:
        summary_task = loop.create_task(_summary_loop())

    try:
        result = await task
    finally:
        if summary_task:
            stop_summary.set()
            with contextlib.suppress(Exception):
                await summary_task
        with contextlib.suppress(Exception):
            await progress_task

    return status_msg, result

# --------------------------
# RECENT backfill (new behavior)
# --------------------------

async def backfill_recent_channel_history_to_es(
    guild_id: int | None,
    channel_id: int,
    chunk: int = 200,
) -> int:
    """
    Fetch ONLY recent messages:
      - Uses a channel-scoped 'last_seen_id' (stored locally) as a high-water mark.
      - If first run (no last_seen), grab the latest <chunk> messages.
      - On subsequent runs, fetch messages strictly AFTER last_seen (i.e., newer).
    Indexes each message by snowflake id; duplicates are naturally ignored by ES.
    Returns the number of messages we attempted to index.
    """
    # Resolve channel
    guild = bot.get_guild(int(guild_id)) if guild_id else None
    channel = None
    if guild:
        channel = guild.get_channel(int(channel_id))
    if channel is None:
        channel = bot.get_channel(int(channel_id))
    if channel is None:
        raise RuntimeError(f"Channel {channel_id} not found")

    # Load & read the high-water mark
    state = _load_state()
    key = _state_key(guild_id, channel_id)
    last_seen_id = state.get("last_seen_by_channel", {}).get(key)

    # Build history kwargs to fetch RECENT, not last
    kwargs: Dict[str, Any] = dict(limit=int(chunk), oldest_first=False)
    if last_seen_id:
        kwargs["after"] = discord.Object(id=int(last_seen_id))

    indexed = 0
    max_id_seen = int(last_seen_id) if last_seen_id else 0

    async for msg in channel.history(**kwargs):
        content = (msg.content or "").strip()
        if not content and not msg.reference:
            continue

        role = "assistant" if bot.user and msg.author.id == bot.user.id else "user"
        reply_to = str(msg.reference.message_id) if msg.reference else None

        try:
            index_message(
                message_id=str(msg.id),
                guild_id=str(msg.guild.id) if msg.guild else "DM",
                channel_id=str(channel_id),
                user_id=str(msg.author.id),
                role=role,
                content=content,
                timestamp=msg.created_at.isoformat(),
                reply_to_id=reply_to,
            )
            indexed += 1
            if msg.id > max_id_seen:
                max_id_seen = msg.id
        except Exception:
            logging.exception("Recent backfill index error", exc_info=False)
            continue

    if max_id_seen and max_id_seen != (int(last_seen_id) if last_seen_id else 0):
        state.setdefault("last_seen_by_channel", {})[key] = str(max_id_seen)
        _save_state(state)

    return indexed

# --------------------------
# Events
# --------------------------

@bot.event
async def on_ready():
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="Graphs Go BRRR 📈")
    )
    logger.info("Bot is online and ready!")

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    msg = reaction.message
    if msg.author.id != bot.user.id:
        return
    rec = get_message_expansion(msg.id)
    if not rec:
        return

    emoji = str(reaction.emoji)

    if emoji == EXPAND_EMOJI and not rec["expanded"]:
        full = rec["full_text"]
        footer = f"\n\n(react {COLLAPSE_EMOJI} to collapse)"
        
        if len(full) + len(footer) > 2000:
            # Too big for one message -> Send as file
            import io
            try:
                f = io.BytesIO(full.encode("utf-8"))
                await msg.reply(
                    "⚠️ Response too long to expand inline. Sending as file.",
                    file=discord.File(f, filename="response.md")
                )
                # We do NOT mark as expanded because we didn't actually expand the original message.
                # Just remove the user's reaction so they can try again if they really want the file again?
                # Or maybe we leave it.
                await msg.remove_reaction(emoji, user)
            except Exception as e:
                logger.error(f"Failed to send long response file: {e}")
        else:
            with contextlib.suppress(Exception):
                await msg.edit(content=f"{full}{footer}")
            set_message_expanded(msg.id, True)
            with contextlib.suppress(Exception):
                await msg.clear_reaction(EXPAND_EMOJI)
            with contextlib.suppress(Exception):
                await msg.add_reaction(COLLAPSE_EMOJI)

    elif emoji == COLLAPSE_EMOJI and rec["expanded"]:
        full = rec["full_text"]
        preview, _ = make_preview(full, LINE_TRUNCATE_AT)
        footer = f"\n\n(react {EXPAND_EMOJI} to expand)"
        with contextlib.suppress(Exception):
            await msg.edit(content=f"{preview}{footer}")
        set_message_expanded(msg.id, False)
        with contextlib.suppress(Exception):
            await msg.clear_reaction(COLLAPSE_EMOJI)
        with contextlib.suppress(Exception):
            await msg.add_reaction(EXPAND_EMOJI)

# --------------------------
# Commands
# --------------------------

@bot.command(name="ping")
async def ping(ctx):
    await ctx.reply("pong")

@bot.command(name="memory_fetch_more")
@commands.has_permissions(manage_messages=True)
async def memory_fetch_more(ctx, chunk: int = 200):
    """
    Fetch RECENT messages for this channel only (not old history).
    Uses a per-channel 'last_seen_id' high-water mark; first run grabs latest <chunk>.
    """
    try:
        count = await backfill_recent_channel_history_to_es(
            ctx.guild.id if ctx.guild else None, ctx.channel.id, chunk=chunk
        )
        await ctx.reply(f"Indexed ~{count} recent message(s) for this channel.")
    except Exception as e:
        await ctx.reply(f"❌ {e}")

# --------------------------
# Main message handler
# --------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _strip_mention_and_trigger(raw: str) -> str:
    s = raw
    if bot.user:
        s = re.sub(f"<@!?{bot.user.id}>", "", s).strip()
    return s

def _looks_like_search(s: str) -> bool:
    s = s.lower().strip()
    return (
        s.startswith("search ") or
        s.startswith("look up ") or
        s.startswith("lookup ") or
        s.startswith("news ") or
        " search " in f" {s} " or
        " news "   in f" {s} " or
        s in {"search", "news"}
    )

def _extract_search_query(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(search|lookup|look up|news)\s*[:,-]*\s*", "", s, flags=re.I)
    return s or "latest"

def _has_google_search() -> bool:
    # Check both imported vars (triggered metadata → env mirror) and env directly
    ga = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    gc = GOOGLE_CSE_ID or os.environ.get("GOOGLE_CSE_ID")
    ok = bool(ga and gc)
    if not ok:
        logger.debug("Google search disabled: GOOGLE_API_KEY=%s, GOOGLE_CSE_ID=%s", bool(ga), bool(gc))
    return ok

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Pass-through slash/! commands
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Trigger: direct mention or reply to bot
    raw_prompt = _strip_mention_and_trigger(message.content)
    prompt = raw_prompt
    user_id = message.author.id

    is_direct_mention = bot.user.mentioned_in(message)
    is_reply_to_bot = False
    ref_msg = None
    if message.reference and isinstance(message.reference, discord.MessageReference):
        try:
            ref_msg = message.reference.resolved or await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None
        if ref_msg and ref_msg.author.id == bot.user.id:
            is_reply_to_bot = True

    # Allow the plain text trigger too
    if not (is_direct_mention or is_reply_to_bot) or message.mention_everyone:
        return

    # Pre-log the user's message (live indexing)
    try:
        index_message(
            message_id=str(message.id),
            guild_id=str(message.guild.id) if message.guild else "DM",
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            role="user",
            content=message.content or "",
            timestamp=message.created_at.isoformat(),
            reply_to_id=(str(message.reference.message_id) if message.reference else None),
        )
    except Exception:
        pass

    # ---- Search: fast-path ONLY if fully configured; else fall through to tools ----
    if _looks_like_search(prompt) and web_search is not None and _has_google_search():
        q = _extract_search_query(prompt)
        try:
            results = web_search(q, max_results=5)
        except Exception:
            results = []
            logger.exception("web_search failed")
        if results:
            lines = ["**Top results:**"]
            for r in results:
                title = r.get("title") or "(untitled)"
                url = r.get("url") or ""
                snippet = (r.get("snippet") or "").strip()
                if snippet:
                    snippet = snippet[:300]
                lines.append(f"- [{title}]({url}) — {snippet}")
            await send_or_edit_with_truncation("\n".join(lines), channel=message.channel, reply_to=message)
            return
        # If configured but the query returned nothing, say so (this path is “real”)
        await send_or_edit_with_truncation("No results found.", channel=message.channel, reply_to=message)
        return
    # If not configured, we do NOT send “No results found” — we let the model’s web_search tool handle it.

    # Intent
    # Fix for "gemini imagine" being grabbed by chat intent
    if raw_prompt.lower().strip().startswith("gemini imagine"):
        intent = "generate_image"
        # We don't strip "gemini" here because stability_utils expects it? 
        # Actually stability_utils checks if content.startswith("gemini").
    else:
        intent = await classify_intent(raw_prompt)
    logger.debug(f"Intent classified as: {intent}")

    # Collect image inputs (replied image FIRST, then attachments, then URLs)
    image_urls: List[str] = []
    
    # 1. Replied Image (Priority 0 - The Base)
    if ref_msg and ref_msg.attachments:
        for a in ref_msg.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                b64 = await image_url_to_base64(a.url)
                if b64:
                    image_urls.append(b64)
    # Also check replied-to embeds? (Optional but good for completeness)
    if ref_msg and ref_msg.embeds:
        for e in ref_msg.embeds:
            if e.image and e.image.url:
                b64 = await image_url_to_base64(e.image.url)
                if b64:
                    image_urls.append(b64)

    # 2. Attachments in current message
    if message.attachments:
        for a in message.attachments:
            if a.content_type and a.content_type.startswith("image/"):
                b64 = await image_url_to_base64(a.url)
                if b64:
                    image_urls.append(b64)

    # 3. URLs in current message
    matches = re.findall(
        r"https?://[^\s]+(?:\.(?:png|jpg|jpeg|webp|gif)|cdn\.discordapp\.com/attachments/[^\s]+)",
        message.content,
    )
    for raw_url in matches:
        if "cdn.discordapp.com" in raw_url:
            b64 = await image_url_to_base64(raw_url)
            if b64:
                image_urls.append(b64)
        else:
            image_urls.append(raw_url)
            
    # Remove duplicates while preserving order
    # (Simple logic: if b64 strings are identical, dedup. URLs dedup by string)
    seen = set()
    unique_urls = []
    for u in image_urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    image_urls = unique_urls

    # Any URL (for summarize)
    general_url_match = re.search(r"https?://[^\s]+", message.content)

    try:
        # Quick ETAs for progress bar polish
        duration_estimate = {
            "generate_image": 40,
            "edit_image": 40,
            "summarize_url": 10,
            "describe_image": 8,
            "chat": 6,
            "get_weather": 5,
            "get_stock": 5,
        }.get(intent, 12)

        if intent == "get_weather":
            response = await handle_weather_request(
                message, bot.user.id, get_location_details, get_weather_data,  # util fns
                None, f"{message.guild.id}-{message.channel.id}", message.author.id
            )
            if response:
                await send_or_edit_with_truncation(response, channel=message.channel, reply_to=message)
            return

        # GEMINI CHAT
        if intent == "gemini_chat":
            from gemini_utils import generate_gemini_text
            # Strip 'gemini' prefix if present to clean up prompt
            clean_prompt = re.sub(r"^gemini\s*", "", prompt, flags=re.IGNORECASE).strip()
            
            # Fetch Memory used by Gemini
            context_msgs = build_message_window(
                guild_id=message.guild.id if message.guild else "DM",
                channel_id=message.channel.id,
                user_id=message.author.id,
                limit_msgs=20 
            )
            
            # Filter out "gemini imagine" prompts from context to prevent confusion/hallucination
            # (The model sometimes answers the previous imagine prompt instead of the current one)
            context_msgs = [
                m for m in context_msgs 
                if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
            ]

            # Collect Image Bytes
            gemini_image_inputs = []
            if image_urls:
                import base64
                for url in image_urls:
                    # image_urls contains data URIs (data:image/png;base64,...) from earlier logic
                    try:
                        if url.startswith("data:image/"):
                            header, encoded = url.split(",", 1)
                            gemini_image_inputs.append(base64.b64decode(encoded))
                    except Exception as e:
                        logger.error(f"Failed to decode image data URI for Gemini: {e}")

            # Explicit Code Execution Trigger
            enable_code_execution = False
            # Check if user said "gemini code <prompt>"
            # 'clean_prompt' currently has "code <prompt>" if original was "gemini code <prompt>"
            # because we stripped "gemini " earlier.
            if clean_prompt.lower().startswith("code "):
                enable_code_execution = True
                clean_prompt = clean_prompt[5:].strip() # Remove "code "
            
            # Status Tracking for Live Code Execution
            status_tracker = {"text": ""}
            
            def _live_code_summarizer():
                return status_tracker["text"] or "Using Gemini 3..."

            status_msg, response = await live_status_with_progress(
                message,
                action_label="Thinking (Gemini)",
                emoji="✨",
                coro=asyncio.to_thread(generate_gemini_text, clean_prompt, context=context_msgs, images=gemini_image_inputs, status_tracker=status_tracker, enable_code_execution=enable_code_execution), 
                duration_estimate=6,
                summarizer=_live_code_summarizer,
            )
            
            if response:
                # Unpack tuple from gemini_utils (text, artifacts)
                if isinstance(response, tuple):
                    text_resp, artifacts = response
                else:
                    text_resp, artifacts = response, []

                if text_resp:
                    await send_or_edit_with_truncation(text_resp, target_msg=status_msg)
                
                # Send artifacts (images/plots)
                if artifacts:
                    import io
                    import mimetypes
                    files = []
                    for i, (data, mime) in enumerate(artifacts):
                        ext = mimetypes.guess_extension(mime) or ".png"
                        f = io.BytesIO(data)
                        files.append(discord.File(f, filename=f"artifact_{i}{ext}"))
                    
                    if files:
                        try:
                            # Send as reply to status_msg
                            await status_msg.reply(files=files)
                        except Exception as e:
                            logger.error(f"Failed to send artifacts: {e}")
                            await status_msg.reply("⚠️ Failed to upload generated artifacts.")
                
                # For indexing, use text_resp
                response = text_resp or "" # normalize for index_message below
                
                # Index the Gemini response with model tag
                try:
                    index_message(
                        message_id=str(status_msg.id) if status_msg else str(message.id) + "-gemini",
                        guild_id=str(message.guild.id) if message.guild else "DM",
                        channel_id=str(message.channel.id),
                        user_id=str(message.author.id),
                        role="assistant",
                        content=response,
                        timestamp=_now_iso(),
                        reply_to_id=str(message.id),
                        model="gemini-3-flash-preview"
                    )
                except Exception:
                    pass
            else:
                await status_msg.edit(content="❌ Gemini returned no response.")
            return

        # IMAGE GENERATION
        if intent == "generate_image":
            # ---------------------------------------------------------
            # SPECIAL: Weather Widget Generator
            # "imagine weather <location>"
            # ---------------------------------------------------------
            weather_match = re.search(r"imagine\s+weather\s+(.*)", prompt, flags=re.IGNORECASE)
            if weather_match:
                loc_query = weather_match.group(1).strip()
                if not loc_query:
                    await status_msg.edit(content="❌ Please specify a location, e.g. `imagine weather Tokyo`.")
                    return

                async def _generate_weather_widget():
                    # 1. Fetch Real-Time Data through weather_utils
                    try:
                        loc = await get_location_details(loc_query)
                        # Default to metric for international flavor, or guess
                        units = "imperial" if "US" in loc.get("name", "") else "metric"
                        data = await get_weather_data(loc["lat"], loc["lon"], units=units)
                        
                        # Format a nice string for the prompt
                        # We use format_weather_response to get the narrative, then extract key bits
                        # Actually simpler: just pull from data directly
                        current = data.get("current", {})
                        temp = current.get("main", {}).get("temp", "?")
                        cond = (current.get("weather") or [{}])[0].get("description", "unknown")
                        
                        # Construct prompt - REFINED for data prominence and WIDESCREEN
                        widget_prompt = (
                            f"A professional, high-end 3D weather widget layout in a WIDESCREEN 16:9 cinematic format. "
                            f"THE PRIMARY FOCUS is the weather data: '{round(float(temp))}°' and '{loc['name']}' in HUGE, BOLD, HIGH-CONTRAST typography. "
                            f"Condition: '{cond.capitalize()}' with a large, vibrant weather icon. "
                            f"The data is prominently displayed on the left or center with extreme clarity. "
                            f"The background is a cinematic, expansive widescreen shot of {loc['name']} "
                            f"reflecting current {cond} skies. "
                            f"Premium Apple/Glassmorphism UI design, clean wide layout, super crisp 8k text rendering."
                        )
                        logger.info(f"Generating widescreen weather widget for {loc['name']}: {widget_prompt}")
                        return await asyncio.to_thread(generate_gemini_image, widget_prompt, 1600, 900)
                    except Exception as e:
                        logger.error(f"Weather widget failed: {e}")
                        return None

                status_msg, image_data = await live_status_with_progress(
                    message,
                    action_label="Building Widget",
                    emoji="🌦️",
                    coro=_generate_weather_widget(),
                    duration_estimate=15,
                    summarizer=(lambda: "Fetching live data... Rendering widget...") if STREAM_OK else None,
                )
                
                if image_data:
                    # image_data is already a BytesIO object from generate_gemini_image
                    image_data.seek(0)
                    await status_msg.reply(file=discord.File(image_data, filename="weather_widget.png"))
                    await status_msg.edit(content=f"✅ Weather Widget for **{loc_query}**")
                else:
                    await status_msg.edit(content="❌ Failed to generate weather widget.")
                return
            
            # Standard Image Gen
            status_msg, image_data = await live_status_with_progress(
                message,
                action_label="Generating",
                emoji="🎨",
                coro=handle_image_generation(message, prompt),
                duration_estimate=duration_estimate,
                summarizer=(lambda: "Rendering image… adding details…") if STREAM_OK else None,
            )
            if image_data:
                await status_msg.edit(content="✅ Image generated")
                await message.channel.send(file=discord.File(image_data, "generated_image.png"))
            else:
                await status_msg.edit(content="❌ Image generation failed.")
            return

        # IMAGE EDIT
        if intent == "edit_image" and image_urls:
            status_msg, image_data = await live_status_with_progress(
                message,
                action_label="Editing",
                emoji="🔧",
                coro=edit_image_with_prompt(image_urls, prompt),
                duration_estimate=duration_estimate,
                summarizer=(lambda: "Applying edits… refining…") if STREAM_OK else None,
            )
            if image_data:
                await status_msg.edit(content="✅ Image edited")
                await message.channel.send(file=discord.File(image_data, "edited_image.png"))
            else:
                await status_msg.edit(content="❌ Edit failed or blocked.")
            return

        # SUMMARIZE URL
        if intent == "summarize_url" and general_url_match:
            url = general_url_match.group(0)

            async def _do_summarize():
                html = fetch_url_content(url)
                title, text = extract_main_text(html)
                condensed = reduce_text_length(text, max_chars=3000)
                msgs = [
                    {"role": "system", "content": "Summarize crisply (bullets ok) and extract key facts/figures."},
                    {"role": "user", "content": f"Title: {title or ''}\n\n{condensed}"},
                ]
                summary = await generate_openai_messages_response_with_tools(msgs, tools=[])
                return f"**{title or 'Summary'}**\n{summary}"

            status_msg, summary = await live_status_with_progress(
                message,
                action_label="Summarizing",
                emoji="📰",
                coro=_do_summarize(),
                duration_estimate=duration_estimate,
                summarizer=(lambda: f"Fetching page…\nURL: {url}\nExtracting main content…") if STREAM_OK else None,
            )
            if summary:
                await send_or_edit_with_truncation(summary, target_msg=status_msg)
            else:
                await status_msg.edit(content="❌ Summary failed.")
            return

        # DESCRIBE IMAGE (with injection)
        if intent == "describe_image" and image_urls:
            # Build an injection tailored to humor/visual explanation
            describe_injection = (
                "When asked to describe an image:\n"
                "- Identify the setting, subjects, and any visible text (transcribe briefly).\n"
                "- If humor/irony/meme is implied, explain *why* it's funny or incongruous.\n"
                "- Point to 2–3 specific visual cues that support your explanation.\n"
                "- Keep it concise and concrete."
            )
            if is_reply_to_bot and ref_msg and (ref_msg.content or "").strip():
                reply_context = f"You are responding to your previous message:\n---\n{ref_msg.content.strip()}\n---"
            else:
                reply_context = ""

            async def _describe():
                msgs = [
                    # DO NOT modify base system; add a second system block
                    {"role": "system", "content": "Describe these images concisely. If text exists, transcribe it."},
                    {"role": "system", "content": describe_injection},
                ]
                if reply_context:
                    msgs.append({"role": "system", "content": reply_context})
                msgs.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                               + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls],
                })
                return await generate_openai_messages_response_with_tools(msgs, tools=[])

            status_msg, response = await live_status_with_progress(
                message,
                action_label="Describing",
                emoji="🖼️",
                coro=_describe(),
                duration_estimate=duration_estimate,
                summarizer=(lambda: "Looking at visual elements… noting layout/text…") if STREAM_OK else None,
            )
            
            if response:
                # Unpack tuple from gemini_utils (text, artifacts)
                # If old version or error returned just None, handle gracefully
                if isinstance(response, tuple):
                    text_resp, artifacts = response
                else:
                    text_resp, artifacts = response, []

                if text_resp and text_resp.strip():
                    await send_or_edit_with_truncation(text_resp, target_msg=status_msg)
                
                # Send artifacts (images/plots)
                if artifacts:
                    import io
                    import mimetypes
                    files = []
                    for i, (data, mime) in enumerate(artifacts):
                        ext = mimetypes.guess_extension(mime) or ".png"
                        f = io.BytesIO(data)
                        files.append(discord.File(f, filename=f"artifact_{i}{ext}"))
                    
                    if files:
                        try:
                            # Send as reply to status_msg
                            await status_msg.reply(files=files)
                        except Exception as e:
                            logger.error(f"Failed to send artifacts: {e}")
                            await status_msg.reply("⚠️ Failed to upload generated artifacts.")
            else:
                await status_msg.edit(content="❌ Generation failed.")
            return

        # STOCK
        if intent == "get_stock" and prompt.lower().startswith("stock"):
            async with message.channel.typing():
                await handle_stock_command(message, prompt)
            return

        # ----- CHAT (ES-backed messages[] window) -----
        def build_msgs_for_chat() -> List[Dict[str, Any]]:
            msgs: List[Dict[str, Any]] = []
            # 1) Base system prompt
            msgs.append({"role": "system", "content": "You are a helpful Discord bot. Keep responses succinct but clear."})
            # 1b) Tool-nudging block
            msgs.append({"role": "system", "content":
                "If the user explicitly says 'search', 'look up', or 'news', prefer using the web_search tool with their query."})
            # 2) Temporal awareness block (recent timeline newest→oldest)
            timeline_block = build_timeline_prompt_block(
                guild_id=message.guild.id, channel_id=message.channel.id, user_id=user_id, max_items=12
            )
            msgs.append({"role": "system", "content": timeline_block})
            # 2b) If replying to the bot, include the replied-to assistant message
            if is_reply_to_bot and ref_msg and (ref_msg.content or "").strip():
                msgs.append({"role": "system", "content":
                    f"You are replying to your earlier assistant message:\n---\n{ref_msg.content.strip()}\n---"})
            # 3) ES conversation window (oldest→newest)
            history_msgs = build_message_window(
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                user_id=user_id,
                limit_msgs=24,
            )
            msgs.extend(history_msgs)
            # 4) Current user message last
            msgs.append({"role": "user", "content": raw_prompt})
            return msgs

        async def _chat_with_es_window():
            msgs = build_msgs_for_chat()
            return await generate_openai_messages_response_with_tools(msgs, tools=TOOLS_DEF)

        # Light live summary outline
        def _summarizer():
            return "• Using recent timeline + ES history…\n• Drafting answer…"

        status_msg, response = await live_status_with_progress(
            message,
            action_label="Responding",
            emoji="💬",
            coro=_chat_with_es_window(),
            duration_estimate=duration_estimate,
            summarizer=_summarizer if STREAM_OK else None,
        )

        if response and response.strip():
            await send_or_edit_with_truncation(response, target_msg=status_msg)
        else:
            await status_msg.edit(content="🤖 INSUFFICIENT DATA FOR MEANINGFUL ANSWER")

        # Store assistant reply (live indexing)
        try:
            index_message(
                message_id=str(status_msg.id) if status_msg else str(message.id) + "-bot",
                guild_id=str(message.guild.id) if message.guild else "DM",
                channel_id=str(message.channel.id),
                user_id=str(message.author.id),  # conversation_key is anchored to the human user
                role="assistant",
                content=response or "",
                timestamp=_now_iso(),
                reply_to_id=str(message.id),
            )
        except Exception:
            pass

    except Exception as e:
        logger.exception("Critical error in on_message dispatch")
        with contextlib.suppress(Exception):
            await message.reply(f"❌ Critical failure: {str(e)[:150]}...")

    # let other cogs/commands run too
    await bot.process_commands(message)

# ---- Entrypoint (used by main.py) ----
def run_bot():
    bot.run(DISCORD_TOKEN)
