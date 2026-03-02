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
import collections
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
from services.memory_utils import (
    index_message,                  # (message_id, guild_id, channel_id, user_id, role, content, timestamp?, reply_to_id?)
)

# OpenAI helpers
from providers.openai_utils import (
    classify_intent,
    image_url_to_base64,
)


# Optional features (your existing utilities)
from services.weather_utils import get_location_details, get_weather_data
from services.database_utils import get_message_expansion
from providers.claude_utils import ANTHROPIC_API_KEY
from bot.intent_dispatcher import dispatch_intent
from bot.message_inputs import (
    collect_gemini_parts,
    collect_image_inputs,
    extract_search_query,
    has_google_search,
    looks_like_search,
    resolve_reference_message,
    strip_mention_and_trigger,
)
from bot.moderation_view import ModerationFallbackView
from bot.ui_messages import (
    EXPAND_EMOJI,
    COLLAPSE_EMOJI,
    handle_expansion_reaction,
    send_or_edit_with_truncation as ui_send_or_edit_with_truncation,
    live_status_with_progress as ui_live_status_with_progress,
)

# NEW: direct search fast-path (kept, but now properly gated)
try:
    from services.search_utils import web_search
except Exception:
    web_search = None

# Streaming niceties (optional)
try:
    from services.stream_utils import ThrottledEditor
    STREAM_OK = True
except Exception:
    STREAM_OK = False

# ---- Logging ----
logger = logging.getLogger("discord_bot")
if "--verbose" in sys.argv:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# Suppress noisy libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

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
logger.info(f"Anthropic Key: ANTHROPIC_API_KEY={_redact(ANTHROPIC_API_KEY)}")

# ---- Discord ----
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="/", intents=intents)

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

# ---- Multi-Image Selection ----
# Track messages recently processed to prevent gateway replays / feedback loops
_processed_msg_ids = collections.deque(maxlen=100)

# Track users currently being prompted (to prevent on_message from double-processing)
_pending_image_selection: set[int] = set()  # user IDs awaiting reply

# Track message IDs currently being expanded to prevent race conditions
_expansion_locks: set[int] = set()

async def prompt_for_image_selection(message, image_count: int, timeout: float = 30.0):
    """
    Ask user which image to process when multiple are present.
    Returns: int (0-based index), "all", or 0 on timeout/invalid.
    """
    user_id = message.author.id
    _pending_image_selection.add(user_id)
    
    try:
        prompt_msg = await message.reply(
            f"📷 I see **{image_count} images**. Which one should I edit?\n"
            "Reply with a number (1, 2, ...) or **all**."
        )
        
        def check(m):
            return m.author.id == user_id and m.channel == message.channel
        
        try:
            reply = await bot.wait_for("message", check=check, timeout=timeout)
            text = reply.content.strip().lower()
            if text == "all":
                return "all"
            if text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < image_count:
                    return idx
            await message.channel.send("⚠️ Invalid selection. Using the first image.")
            return 0
        except asyncio.TimeoutError:
            await prompt_msg.edit(content="⏰ Timed out. Using the first image.")
            return 0
    finally:
        _pending_image_selection.discard(user_id)

# --------------------------
# Helpers
# --------------------------

def is_probably_image(url: str) -> bool:
    path = urlparse(url).path
    mime, _ = mimetypes.guess_type(path)
    return bool(mime and mime.startswith("image/"))

async def _auto_index_bot_message(sent_message, full_text: str, *, original_message=None, reply_to=None, model: Optional[str] = None):
    try:
        src_msg = original_message or reply_to
        if src_msg:
            index_message(
                message_id=str(sent_message.id),
                guild_id=str(src_msg.guild.id) if src_msg.guild else "DM",
                channel_id=str(src_msg.channel.id),
                user_id=str(src_msg.author.id),
                role="assistant",
                content=full_text,
                timestamp=_now_iso(),
                reply_to_id=str(src_msg.id),
                model=model or "unknown",
            )
    except Exception as e:
        logger.warning(f"Failed to auto-index bot message: {e}")


async def send_or_edit_with_truncation(*args, **kwargs):
    kwargs.setdefault("index_callback", _auto_index_bot_message)
    return await ui_send_or_edit_with_truncation(*args, **kwargs)


async def live_status_with_progress(*args, **kwargs):
    kwargs.setdefault("stream_ok", STREAM_OK)
    if STREAM_OK:
        kwargs.setdefault("editor_factory", lambda status_msg: ThrottledEditor(status_msg, min_interval_s=1.5, max_len=1300))
    return await ui_live_status_with_progress(*args, **kwargs)

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
    logger.info("Startup: Gemini Tools Patch Loaded (Scipy+Artifacts) v3")

@bot.event
async def on_raw_reaction_add(payload):
    # Ignore bot's own reactions
    if payload.user_id == bot.user.id:
        return

    # Ignore ANY bot reaction (if member is available, e.g. in guild)
    if payload.member and payload.member.bot:
        return
    
    # If member not in payload (DM? or uncached), try to fetch user
    if not payload.member:
        try:
            u = bot.get_user(payload.user_id) or await bot.fetch_user(payload.user_id)
            if u.bot:
                return
        except Exception:
            pass  # If we can't fetch, assume user? Or safe to ignore? Let's process.

    # Check if this message is a truncatable message (fast DB check)
    rec = get_message_expansion(payload.message_id)
    if not rec:
        return

    # Get channel and message
    channel = bot.get_channel(payload.channel_id)
    if not channel:
        return
    try:
        msg = await channel.fetch_message(payload.message_id)
    except discord.NotFound:
        return
    except Exception as e:
        logger.error(f"Failed to fetch message for expansion: {e}")
        return

    # Ensure we only care about reactions to the bot's messages
    if msg.author.id != bot.user.id:
        return

    emoji = str(payload.emoji)

    user = bot.get_user(payload.user_id) 
    # If user is None (not in cache), we can't easily pass 'user' to remove_reaction 
    # unless we fetch member. But remove_reaction accepts Member or User.
    # We can use payload.member if in guild.
    member = payload.member or user

    # Expansion lock check
    if payload.message_id in _expansion_locks:
        return
    _expansion_locks.add(payload.message_id)

    try:
        await handle_expansion_reaction(msg, emoji, rec, member=member)
    finally:
        _expansion_locks.discard(payload.message_id)

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

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    if message.id in _processed_msg_ids:
        return
    _processed_msg_ids.append(message.id)

    # Pass-through slash/! commands
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Trigger: direct mention or reply to bot
    raw_prompt = strip_mention_and_trigger(message.content, bot.user.id if bot.user else None)
    prompt = raw_prompt
    user_id = message.author.id

    is_direct_mention = bot.user.mentioned_in(message)
    ref_msg, is_reply_to_bot = await resolve_reference_message(message, bot.user)

    # Allow the plain text trigger too
    if not (is_direct_mention or is_reply_to_bot) or message.mention_everyone:
        return
    
    # Skip if user is currently responding to an image selection prompt
    if message.author.id in _pending_image_selection:
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
    if looks_like_search(prompt) and web_search is not None and has_google_search(GOOGLE_API_KEY, GOOGLE_CSE_ID, os.environ):
        q = extract_search_query(prompt)
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

    # Quick check for images BEFORE intent classification
    has_attachments = bool(message.attachments or (ref_msg and ref_msg.attachments))
    
    # Intent
    # 1. Determine Intent
    # Fix for "gemini imagine" being grabbed by chat intent
    if raw_prompt.lower().strip().startswith("gemini imagine"):
        intent = "generate_image"
        # We don't strip "gemini" here because stability_utils expects it? 
        # Actually stability_utils checks if content.startswith("gemini").
    else:
        # Quick override for video
        lower_prompt = prompt.lower()
        if "generate" in lower_prompt and ("video" in lower_prompt or "movie" in lower_prompt or "clip" in lower_prompt or "sora" in lower_prompt):
             intent = "generate_video"
        else:
             intent = await classify_intent(prompt, has_images=has_attachments)
        
    logger.info(f"Intent identified as: {intent} (has_attachments={has_attachments})")

    image_urls = await collect_image_inputs(message, ref_msg, image_url_to_base64)
    gemini_parts = await collect_gemini_parts(message, ref_msg, image_urls)

    # Any URL (for summarize)
    general_url_match = re.search(r"https?://[^\s]+", message.content)

    try:
        await dispatch_intent(
            intent=intent,
            message=message,
            prompt=prompt,
            raw_prompt=raw_prompt,
            user_id=user_id,
            ref_msg=ref_msg,
            is_reply_to_bot=is_reply_to_bot,
            image_urls=image_urls,
            gemini_parts=gemini_parts,
            general_url_match=general_url_match,
            stream_ok=STREAM_OK,
            bot_user=bot.user,
            get_location_details=get_location_details,
            get_weather_data=get_weather_data,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
            prompt_for_image_selection=prompt_for_image_selection,
            moderation_view_factory=ModerationFallbackView,
        )

    except Exception as e:
        logger.exception("Critical error in on_message dispatch")
        with contextlib.suppress(Exception):
            await message.reply(f"❌ Critical failure: {str(e)[:150]}...")

    # let other cogs/commands run too
    await bot.process_commands(message)

# ---- Entrypoint (used by main.py) ----
def run_bot():
    bot.run(DISCORD_TOKEN)
