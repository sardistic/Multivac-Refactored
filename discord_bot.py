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
import io
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
    build_message_window,           # (guild_id, channel_id, user_id, limit_msgs=24) -> List[{role, content}]
)

# OpenAI helpers
from providers.openai_utils import (
    classify_intent,
    image_url_to_base64,
    generate_openai_messages_response_with_tools,
    TOOLS_DEF,
    OpenAIModerationError,
)


# Optional features (your existing utilities)
from providers.gemini_utils import generate_gemini_text, GeminiModerationError
from google.genai import types
from services.weather_utils import get_location_details, get_weather_data, handle_weather_request, format_weather_response
from services.url_utils import fetch_url_content, extract_main_text, reduce_text_length
from services.database_utils import get_message_expansion
from providers.claude_utils import generate_claude_response, ANTHROPIC_API_KEY
from bot.chat_handler import handle_chat_intent
from bot.image_handler import (
    handle_describe_image_intent,
    handle_edit_image_intent,
    handle_generate_image_intent,
    send_debug_context,
)
from bot.video_handler import handle_generate_video_intent
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
# UI Classes
# --------------------------
class ModerationFallbackView(discord.ui.View):
    """
    Dropdown view to select an alternative model when moderation blocks a response.
    """
    def __init__(self, author_id, retry_callback):
        super().__init__(timeout=120)
        self.author_id = author_id
        self.retry_callback = retry_callback
        
        # Define model options with fallback choices
        options = [
            discord.SelectOption(
                label="Gemini 1.5 Pro (Smarter)", 
                value="gemini-1.5-pro", 
                description="Higher reasoning, might be less strict.",
                emoji="🧠"
            ),
            discord.SelectOption(
                label="Gemini 1.5 Flash (Fast)", 
                value="gemini-1.5-flash", 
                description="Fast and efficient.",
                emoji="⚡"
            ),
             discord.SelectOption(
                label="Gemini 1.5 Pro 002", 
                value="gemini-1.5-pro-002", 
                description="Updated Pro model.",
                emoji="🆕"
            ),
            discord.SelectOption(
                label="GPT-5.2 (OpenAI)", 
                value="gpt-5.2", 
                description="Switch provider to OpenAI.",
                emoji="🟢"
            ),
        ]
        
        select = discord.ui.Select(
            placeholder="Select an alternative model...",
            min_values=1,
            max_values=1,
            options=options,
            custom_id="moderation_model_select"
        )
        select.callback = self.select_callback
        self.add_item(select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Not your request! make your own.", ephemeral=True)
            return False
        return True

    async def select_callback(self, interaction: discord.Interaction):
        # Determine selected model
        model = interaction.data["values"][0]
        
        # Defer and edit to clean up UI
        await interaction.response.edit_message(content=f"🔄 **Retrying with {model}...**", view=None)
        self.stop()
        
        # Trigger retry
        await self.retry_callback(model_name=model)


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
    
    # 4. Gemini Parts (Images and Text/Docs)
    gemini_parts = []
    
    # 4.1 Collect from REPLIED message
    if ref_msg:
        if ref_msg.attachments:
            for a in ref_msg.attachments:
                try:
                    data = await a.read()
                    mime = a.content_type or mimetypes.guess_type(a.filename)[0] or "application/octet-stream"
                    
                    # Handle Images
                    if mime.startswith("image/"):
                        gemini_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                        logger.info(f"Added replied image part: {a.filename}")
                    # Handle Text/Docs
                    else:
                        text_exts = (".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".c", ".cpp", ".h", ".java", ".go", ".rs", ".sql", ".yaml", ".yml", ".html", ".css")
                        if mime.startswith("text/") or "/json" in mime or a.filename.lower().endswith(text_exts):
                            # For text files, we can just pass the text string for better grounding
                            try:
                                content = data.decode("utf-8")
                            except UnicodeDecodeError:
                                content = data.decode("latin-1")
                            
                            if len(content) > 150_000:
                                content = content[:150_000] + "\n... [TRUNCATED] ..."
                            
                            full_text = f"--- REPLIED FILE: {a.filename} ---\n{content}\n"
                            gemini_parts.append(types.Part(text=full_text))
                            logger.info(f"Added replied text part: {a.filename}")
                except Exception as e:
                    logger.error(f"Failed to process replied attachment: {e}")

    # 4.2 Collect from CURRENT message
    if message.attachments:
        for a in message.attachments:
            try:
                data = await a.read()
                mime = a.content_type or mimetypes.guess_type(a.filename)[0] or "application/octet-stream"
                
                # Handle Images
                if mime.startswith("image/"):
                    # Only add if not already present in image_urls list?
                    # Actually, we rely on gemini_parts now for Gemini.
                    gemini_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                    logger.info(f"Added attachment image part: {a.filename}")
                # Handle Text/Docs
                else:
                    text_exts = (".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".c", ".cpp", ".h", ".java", ".go", ".rs", ".sql", ".yaml", ".yml", ".html", ".css")
                    if mime.startswith("text/") or "/json" in mime or a.filename.lower().endswith(text_exts):
                        try:
                            content = data.decode("utf-8")
                        except UnicodeDecodeError:
                            content = data.decode("latin-1")
                        
                        if len(content) > 150_000:
                            content = content[:150_000] + "\n... [TRUNCATED] ..."
                        
                        full_text = f"--- FILE: {a.filename} ---\n{content}\n"
                        gemini_parts.append(types.Part(text=full_text))
                        logger.info(f"Added attachment text part: {a.filename}")
            except Exception as e:
                logger.error(f"Failed to process attachment: {e}")

    # 4.3 Collect image_urls (URLs or base64 data URIs)
    # Convert any unique_urls to Parts
    for url in unique_urls:
        if url.startswith("data:image/"):
            try:
                import base64
                header, encoded = url.split(",", 1)
                mime = header.split(":", 1)[1].split(";", 1)[0]
                gemini_parts.append(types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime))
            except Exception as e:
                logger.error(f"Failed to decode data URI: {e}")
        else:
            # It's a raw URL. Gemini GenAI SDK doesn't always handle raw URLs in Parts easily without uploading.
            # We already handled base64-ing them in image_urls if they were attachments.
            # If it's a web URL, we might need a Part(uri=url) but only if it's GCS.
            # For now, we'll ignore raw web URLs or let the model's search tool handle them.
            pass

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

        # CLAUDE CHAT
        if intent == "claude_chat":
            # Strip 'claude' prefix
            clean_prompt = re.sub(r"^(claude|hey claude)\s*", "", prompt, flags=re.IGNORECASE).strip()
            
            # Fetch Context (same logic as Gemini)
            context_msgs = build_message_window(
                guild_id=message.guild.id if message.guild else "DM",
                channel_id=message.channel.id,
                user_id=message.author.id,
                limit_msgs=20 
            )
            # Filter out "imagine" noise
            context_msgs = [
                m for m in context_msgs 
                if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
            ]
            
            # Format context for Claude (list of dicts)
            # We need to ensure alternating user/assistant roles if possible, or Claude might complain?
            # Our `build_message_window` returns list of dicts {role, content}.
            # But we must append the current user prompt at the end.
            
            claude_messages = []
            # Add system prompt as a fake system message (wrapper handles extraction)
            claude_messages.append({"role": "system", "content": "You are Claude, a helpful AI assistant."})
            
            # Add history
            claude_messages.extend(context_msgs)
            
            # Add current prompt
            claude_messages.append({"role": "user", "content": clean_prompt})
            
            status_msg, response = await live_status_with_progress(
                message,
                action_label="Thinking (Claude)",
                emoji="🧠",
                coro=generate_claude_response(claude_messages),
                duration_estimate=5,
                summarizer=(lambda: "Queries Anthropic API...") if STREAM_OK else None,
            )
            
            if response:
                await send_or_edit_with_truncation(
                    response, 
                    target_msg=status_msg, 
                    original_message=message,
                    model="claude-sonnet-4"
                )
                
                # Manual indexing removed - auto-indexing now handles this
            else:
                 await status_msg.edit(content="❌ Claude returned no response.")
            return

        # GEMINI CHAT
        if intent == "gemini_chat":
            from providers.gemini_utils import generate_gemini_text
            # Strip 'gemini' prefix if present to clean up prompt
            clean_prompt = re.sub(r"^gemini\s*", "", prompt, flags=re.IGNORECASE).strip()
            
            # Special Test Mode: "gemini test" -> Force code block output, bypass pagination UI
            is_test_mode = False
            # Check if it starts with "test " or is exactly "test"
            if clean_prompt.lower() == "test" or clean_prompt.lower().startswith("test "):
                is_test_mode = True
                # Remove "test" from the actual prompt sent to LLM so it answers the query
                clean_prompt = re.sub(r"^test\s*", "", clean_prompt, flags=re.IGNORECASE).strip()
            
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

            # Explicit Code Execution Trigger
            enable_code_execution = False
            if clean_prompt.lower().startswith("code "):
                enable_code_execution = True
                clean_prompt = clean_prompt[5:].strip() # Remove "code "
            elif is_test_mode:
                enable_code_execution = True # Always enable code for debug/test mode so we can generate large lists
            
            # Status Tracking for Live Code Execution
            status_tracker = {"text": ""}
            
            def _live_code_summarizer():
                return status_tracker["text"] or "Using Gemini 1.5 Flash..."
            
            # Prepare search_ids for RAG
            search_ids = {
                "guild_id": str(message.guild.id) if message.guild else "DM",
                "channel_id": str(message.channel.id),
                "user_id": str(message.author.id)
            }

            # Wrapper for generation to support retries on moderation block
            async def _do_gemini_generation(model_name=None):
                selected_model = model_name or "gemini-2.0-flash"
                
                async def _run_gen():
                    if "gpt" in selected_model.lower():
                         ctx = {
                           "guild_id": message.guild.id if message.guild else "DM",
                           "channel_id": message.channel.id,
                           "user_id": str(message.author.id)
                         }
                         msgs = list(context_msgs) 
                         msgs.append({"role": "user", "content": clean_prompt})
                         
                         txt = await generate_openai_messages_response_with_tools(
                             msgs, 
                             tools=TOOLS_DEF, 
                             tool_context=ctx, 
                             model=selected_model
                         )
                         return txt, []
                    else:
                         return await asyncio.to_thread(
                            generate_gemini_text, 
                            clean_prompt, 
                            context=context_msgs, 
                            extra_parts=gemini_parts, 
                            status_tracker=status_tracker, 
                            enable_code_execution=enable_code_execution,
                            search_ids=search_ids,
                            model_name=selected_model
                         )

                try:
                    status_msg, response = await live_status_with_progress(
                        message,
                        action_label=f"Thinking ({selected_model})",
                        emoji="✨",
                        coro=_run_gen(), 
                        duration_estimate=6,
                        summarizer=_live_code_summarizer,
                    )
                    
                    if response:
                        # Unpack tuple from gemini_utils (text, artifacts)
                        if isinstance(response, tuple):
                            text_resp, artifacts = response
                        else:
                            text_resp, artifacts = response, []

                        # Prepare artifacts (images/plots/audio)
                        files_to_send = []
                        if artifacts:
                            import io
                            import mimetypes
                            for i, (data, mime) in enumerate(artifacts):
                                # Detect extension (wav/png/etc)
                                ext = mimetypes.guess_extension(mime) or ".bin"
                                # Force .wav for audio/wav to be safe
                                if "wav" in mime: ext = ".wav"
                                
                                f = io.BytesIO(data)
                                files_to_send.append(discord.File(f, filename=f"artifact_{i}{ext}"))

                        if text_resp:
                            if is_test_mode:
                                # Check size for file embedding logic (similar to main bot handling)
                                if len(text_resp) > 1900:
                                     import io
                                     try:
                                         f_text = io.BytesIO(text_resp.encode("utf-8"))
                                         text_file = discord.File(f_text, filename="response.md")
                                         
                                         # Edit status to show we are done but sent a file
                                         await status_msg.edit(content="⚠️ **Test Result Too Long** -> Sent as file `response.md`")
                                         
                                         # Setup files list - append text file to any other artifacts
                                         all_files = [text_file]
                                         if files_to_send:
                                             all_files.extend(files_to_send)
                                         
                                         await status_msg.reply(files=all_files)
                                     except Exception as e:
                                         await status_msg.edit(content=f"❌ Test mode file send failed: {e}")
                                else:
                                    # Force code block for testing, bypass expand/collapse
                                    # We truncate to 1990 chars to roughly fit in 2000 limit with fences
                                    code_content = f"```\n{text_resp[:1990]}\n```"
                                    try:
                                        await status_msg.edit(content=code_content)
                                        if files_to_send:
                                            await status_msg.reply(files=files_to_send)
                                    except Exception as e:
                                         await status_msg.edit(content=f"❌ Test mode failed: {e}")
                            else:
                                # Pass text AND files to the helper so they stay attached even if text becomes a file
                                await send_or_edit_with_truncation(text_resp, target_msg=status_msg, extra_files=files_to_send)
                        elif files_to_send:
                            # If no text but we have files, send them
                            await status_msg.reply(files=files_to_send)
                        else:
                            # response is tuple (text, artifacts) but both are empty?
                            # This happens if Gemini returns empty string and no files.
                            await status_msg.edit(content="❌ Gemini returned no text or files.")
                        
                    else:
                        await status_msg.edit(content="❌ Gemini returned no response.")

                except GeminiModerationError as e:
                    logger.warning(f"Gemini moderation hit: {e}")
                    # Show Fallback View
                    user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
                    view = ModerationFallbackView(author_id=message.author.id, retry_callback=_do_gemini_generation)
                    await message.reply(user_msg, view=view)
                except OpenAIModerationError as e:
                    logger.warning(f"OpenAI moderation hit in Gemini fallback: {e}")
                    user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
                    view = ModerationFallbackView(author_id=message.author.id, retry_callback=_do_gemini_generation)
                    await message.reply(user_msg, view=view)
                except Exception as e:
                     logger.exception("Gemini generation error")
                     await message.reply(f"❌ Gemini Error: {e}")

            # Start initial generation
            await _do_gemini_generation()
            return

        # IMAGE GENERATION
        if intent == "generate_image":
            if message.content.startswith("/debug_context"):
                await send_debug_context(message, bot.user)
                return
            await handle_generate_image_intent(
                message=message,
                prompt=prompt,
                duration_estimate=duration_estimate,
                stream_ok=STREAM_OK,
                live_status_with_progress=live_status_with_progress,
            )
            return

        # IMAGE EDIT using Responses API
        if intent == "edit_image" and image_urls:
            await handle_edit_image_intent(
                message=message,
                prompt=prompt,
                image_urls=image_urls,
                prompt_for_image_selection=prompt_for_image_selection,
                live_status_with_progress=live_status_with_progress,
            )
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
            await handle_describe_image_intent(
                message=message,
                prompt=prompt,
                image_urls=image_urls,
                ref_msg=ref_msg,
                is_reply_to_bot=is_reply_to_bot,
                duration_estimate=duration_estimate,
                stream_ok=STREAM_OK,
                live_status_with_progress=live_status_with_progress,
                send_or_edit_with_truncation=send_or_edit_with_truncation,
            )
            return

        # GENERATE VIDEO (Sora)
        if intent == "generate_video":
            await handle_generate_video_intent(
                message=message,
                prompt=prompt,
                user_id=user_id,
                live_status_with_progress=live_status_with_progress,
                stream_ok=STREAM_OK,
            )
            return

        # STOCK
        if intent == "get_stock" and prompt.lower().startswith("stock"):
            async with message.channel.typing():
                await handle_stock_command(message, prompt)
            return

        # ----- CHAT (ES-backed messages[] window) -----
        await handle_chat_intent(
            message=message,
            prompt=prompt,
            raw_prompt=raw_prompt,
            user_id=user_id,
            ref_msg=ref_msg,
            is_reply_to_bot=is_reply_to_bot,
            duration_estimate=duration_estimate,
            stream_ok=STREAM_OK,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
            moderation_view_factory=ModerationFallbackView,
        )

        # Manual indexing removed - auto-indexing now handles this

    except Exception as e:
        logger.exception("Critical error in on_message dispatch")
        with contextlib.suppress(Exception):
            await message.reply(f"❌ Critical failure: {str(e)[:150]}...")

    # let other cogs/commands run too
    await bot.process_commands(message)

# ---- Entrypoint (used by main.py) ----
def run_bot():
    bot.run(DISCORD_TOKEN)
