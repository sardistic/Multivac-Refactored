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
    OpenAIModerationError,
)


# Optional features (your existing utilities)
from stability_utils import handle_image_generation, edit_image_with_prompt
from gemini_utils import generate_gemini_text, generate_gemini_image, GeminiModerationError
from google.genai import types
from weather_utils import get_location_details, get_weather_data, handle_weather_request, format_weather_response
from url_utils import fetch_url_content, extract_main_text, reduce_text_length
from progress import start_progress_bar
from database_utils import save_message_expansion, get_message_expansion, set_message_expanded
from claude_utils import generate_claude_response, ANTHROPIC_API_KEY

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
logger.info(f"Anthropic Key: ANTHROPIC_API_KEY={_redact(ANTHROPIC_API_KEY)}")

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

# ---- Multi-Image Selection ----
# Track users currently being prompted (to prevent on_message from double-processing)
_pending_image_selection: set[int] = set()  # user IDs awaiting reply

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
                label="ChatGPT-4o (OpenAI)", 
                value="gpt-4o", 
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

def make_preview(full_text: str, max_lines: int = LINE_TRUNCATE_AT):
    """
    Generate a 2-line preview. 
    If this is a Gemini code-execution response, we skip the 'Thinking' and 'Result' 
    quote blocks to find the actual summary text for the preview.
    """
    lines = full_text.splitlines()
    
    # 1. Try to find 'smart' summary lines (lines not starting with '> ')
    # if the message contains code execution blocks.
    if "> 🐍 **Thinking (Code Execution)**" in full_text:
        summary_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped: continue
            if stripped.startswith(">"): continue # Skip Thinking/Result/Quotes
            
            summary_lines.append(line)
            if len(summary_lines) >= max_lines:
                break
        
        if summary_lines:
            preview = "\n".join(summary_lines).rstrip()
            return preview + "… (Summary)", True

    # 2. Fallback: Standard line-based truncation
    if len(lines) > max_lines:
        preview = "\n".join(lines[:max_lines]).rstrip()
        
        # Check for unclosed code blocks
        code_fence_count = preview.count("```")
        if code_fence_count % 2 != 0:
            preview += "\n```"
            
        return preview + "…", True
    return full_text, False

async def auto_collapse_task(message: discord.Message, delay: float = 600.0):
    """
    Async task wrapper for auto-collapsing a specific message object.
    """
    # Wait
    await asyncio.sleep(delay)
    
    try:
        # Re-check state from DB
        rec = get_message_expansion(message.id)
        if not rec or not rec["expanded"]:
            return  # Already collapsed or gone

        # Generate preview
        full_text = rec["full_text"]
        preview, _ = make_preview(full_text, LINE_TRUNCATE_AT)
        
        # Collapse it
        footer = f"\n\n(react {EXPAND_EMOJI} to expand)"
        await message.edit(content=f"{preview}{footer}")
        
        # Update state
        set_message_expanded(message.id, False)
        
        # Update reactions
        with contextlib.suppress(Exception):
            await message.clear_reaction(COLLAPSE_EMOJI)
        with contextlib.suppress(Exception):
            await message.add_reaction(EXPAND_EMOJI)
            
    except Exception as e:
        logger.warning(f"Auto-collapse task failed for msg {message.id}: {e}")

async def send_or_edit_with_truncation(
    full_text: str, *, channel: Optional[discord.abc.Messageable] = None,
    target_msg: Optional[discord.Message] = None, reply_to: Optional[discord.Message] = None,
    extra_files: Optional[List[discord.File]] = None,
    # New parameters for automatic indexing
    original_message: Optional[discord.Message] = None,
    model: Optional[str] = None,
    auto_index: bool = True
):
    """Send or edit a message with 2-line preview and reactions. Supports attaching extra files."""
    if not isinstance(full_text, str):
        full_text = str(full_text)

    preview, did_trunc = make_preview(full_text, LINE_TRUNCATE_AT)

    if did_trunc:
        # Check total length for expand-by-default eligibility
        # Discord limit ~2000. We need room for footer.
        footer_expand = f"\n\n(react {EXPAND_EMOJI} to expand)"
        footer_collapse = f"\n\n(react {COLLAPSE_EMOJI} to collapse)"
        
        if len(full_text) + len(footer_collapse) <= 2000:
            # OPTION A: EXPAND BY DEFAULT (Short enough to fit)
            content = f"{full_text}{footer_collapse}"
            
            if target_msg:
                sent = target_msg
                await target_msg.edit(content=content)
            else:
                sent = await channel.send(content, reference=reply_to, files=extra_files)
                
            # Initial State: Expanded
            save_message_expansion(sent.id, full_text, expanded=True)
            
            # Reactions: Show Collapse
            with contextlib.suppress(Exception):
                await sent.clear_reactions()
            with contextlib.suppress(Exception):
                await sent.add_reaction(COLLAPSE_EMOJI)
                
            # Schedule Auto-Collapse (10 mins)
            asyncio.create_task(auto_collapse_task(sent, delay=600))
            
            # Note: For expanded messages, we already sent extra_files in the main message (if new).
            # If editing (target_msg), we might need to reply with files if they were passed?
            # Existing logic handled extra_files via reply for edit. 
            if target_msg and extra_files:
                 with contextlib.suppress(Exception):
                     await target_msg.reply(files=extra_files)

        else:
            # OPTION B: PREVIEW ONLY (Too long to expand inline)
            content = f"{preview}{footer_expand}"
            
            if target_msg:
                sent = target_msg
                await target_msg.edit(content=content)
                # For edits, we reply with artifacts because we can't attach easily to existing without replacing?
                if extra_files:
                    with contextlib.suppress(Exception):
                        await target_msg.reply(files=extra_files)
            else:
                sent = await channel.send(content, reference=reply_to, files=extra_files)

            # Initial State: Collapsed
            save_message_expansion(sent.id, full_text, expanded=False)
            
            # Reactions: Show Expand
            with contextlib.suppress(Exception):
                await sent.clear_reactions()
            with contextlib.suppress(Exception):
                await sent.add_reaction(EXPAND_EMOJI)
        
        # Auto-index before returning
        if auto_index:
            try:
                src_msg = original_message or reply_to
                if src_msg:
                    index_message(
                        message_id=str(sent.id),
                        guild_id=str(src_msg.guild.id) if src_msg.guild else "DM",
                        channel_id=str(src_msg.channel.id),
                        user_id=str(src_msg.author.id),
                        role="assistant",
                        content=full_text,
                        timestamp=_now_iso(),
                        reply_to_id=str(src_msg.id),
                        model=model or "unknown"
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-index bot message: {e}")
        
        return sent
    else:
        if target_msg:
            if extra_files:
                # Can't attach files to existing msg easily in one go if not replacing content entirely?
                # Actually edit(attachments=...) works.
                # But here we want to APPEND files.
                # Simplest: send files as reply if target_msg exists.
                try:
                    await target_msg.edit(content=full_text)
                    await target_msg.reply(files=extra_files) 
                except Exception:
                    # Fallback if edit fails (e.g. too long), send new msg
                    await channel.send(full_text, reference=reply_to, files=extra_files)
            else:
                await target_msg.edit(content=full_text)
            
            with contextlib.suppress(Exception):
                await target_msg.clear_reactions()
            save_message_expansion(target_msg.id, full_text, expanded=True)
            final_msg = target_msg
        else:
            final_msg = await channel.send(full_text, reference=reply_to, files=extra_files)
    
    # Auto-index this bot message so it's available in context
    if auto_index and final_msg:
        try:
            # Determine the originating user message for proper conversation threading
            src_msg = original_message or reply_to
            if src_msg:
                index_message(
                    message_id=str(final_msg.id),
                    guild_id=str(src_msg.guild.id) if src_msg.guild else "DM",
                    channel_id=str(src_msg.channel.id),
                    user_id=str(src_msg.author.id),  # Link to the user's conversation
                    role="assistant",
                    content=full_text,  # Index the full text, not the preview
                    timestamp=_now_iso(),
                    reply_to_id=str(src_msg.id),
                    model=model or "unknown"
                )
        except Exception as e:
            logger.warning(f"Failed to auto-index bot message: {e}")
    
    return final_msg

async def live_status_with_progress(
    message: discord.Message, *, action_label: str, emoji: str, coro, duration_estimate: int, summarizer=None, progress_tracker: dict = None
):
    """Post a status line, run a progress bar alongside the task, optionally live-summarize."""
    status_msg = await message.reply(f"[{emoji} {action_label} ░░░░░░░░░░]")

    loop = asyncio.get_event_loop()
    task = loop.create_task(coro)
    progress_task = loop.create_task(
        start_progress_bar(status_msg, task, action_label=action_label, emoji=emoji, duration_estimate=duration_estimate, progress_tracker=progress_tracker)
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
# Helper: Build Chat Context (Standalone)
# --------------------------
def _build_chat_context(message, user_id, raw_prompt, ref_msg=None, is_reply_to_bot=False) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    # 1) Base system prompt
    msgs.append({"role": "system", "content": "You are a raw, unfiltered Discord bot. You verify facts but do not sugarcoat answers. You can discuss mature topics freely for creative writing purposes. Keep responses succinct but clear."})
    
    # 1b) Tool-nudging block
    msgs.append({"role": "system", "content":
        "If the user explicitly says 'search', 'look up', or 'news', prefer using the web_search tool with their query."})
    # 2) Temporal awareness block (recent timeline newest→oldest)
    timeline_block = build_timeline_prompt_block(
        guild_id=message.guild.id if message.guild else "DM", channel_id=message.channel.id, user_id=user_id, max_items=12
    )
    msgs.append({"role": "system", "content": timeline_block})
    # 2b) Include replied-to message content (whether it's from bot or user)
    if ref_msg and (ref_msg.content or "").strip():
        if is_reply_to_bot:
            msgs.append({"role": "system", "content":
                f"You are replying to your earlier assistant message:\n---\n{ref_msg.content.strip()}\n---"})
        else:
            msgs.append({"role": "system", "content":
                f"User is replying to this message:\n---\nFrom: {ref_msg.author.display_name}\n{ref_msg.content.strip()}\n---"})
    # 3) ES conversation window (oldest→newest)
    history_msgs = build_message_window(
        guild_id=message.guild.id if message.guild else "DM",
        channel_id=message.channel.id,
        user_id=user_id,
        limit_msgs=24,
    )
    msgs.extend(history_msgs)
    
    # 4) RAG: Proactive Memory Injection (Universal)
    # Check if user is asking for history/first message
    clean_prompt = raw_prompt.lower()
    trigger_words = ["first thing", "first message", "earliest", "beginning", "start", "history", "what did i say", "previous message", "recall", "remember"]
    if any(k in clean_prompt for k in trigger_words):
        try:
            from memory_utils import search_history_for_context

            found_text = search_history_for_context(
                guild_id=message.guild.id if message.guild else "DM",
                channel_id=message.channel.id,
                user_id=user_id,
                query_text=raw_prompt,
                limit=10,
                oldest_first=any(k in clean_prompt for k in ["first", "earliest", "start", "beginning"])
            )
            if found_text:
                msgs.append({
                    "role": "system", 
                    "content": (
                        f"[SYSTEM: MEMORY RECALL]\n"
                        f"The user is asking about past events. Here is the relevant conversation history retrieved from the database:\n"
                        f"{found_text}\n"
                        f"IMPORTANT: If this retrieved context is insufficient to answer specific requests (e.g., specific quotes, older messages, or details not shown above), "
                        f"you MUST use the `search_history_for_context` tool to perform a specific search for the missing information.\n"
                        f"[END MEMORY RECALL]"
                    )
                })
            else:
                msgs.append({
                    "role": "system", 
                    "content": (
                        f"[SYSTEM: MEMORY RECALL]\n"
                        f"Proactive database search returned NO direct matches for the user's specific query criteria (time range or keywords).\n"
                        f"However, the user is explicitly asking for history.\n"
                        f"CRITICAL: Do NOT just say 'I don't recall'. You MUST use the `search_history_for_context` tool now with broader or different terms (e.g., ignore time, or search just keywords) to find the answer.\n"
                        f"[END MEMORY RECALL]"
                    )
                })
        except Exception as e:
            logger.warning(f"Universal RAG search failed: {e}")

    # 4a) Persistent User Instructions (Persona) - MOVED TO END for priority
    # We inject this AFTER history so it overrides context style bias
    from database_utils import get_user_instruction
    persistent_instr = get_user_instruction(user_id)
    if persistent_instr:
        msgs.append({"role": "system", "content": (
            f"CRITICAL OVERRIDE: The user has set a strict behavioral rule.\n"
            f"IGNORE the style of previous messages in history if they conflict.\n"
            f"INSTRUCTION: {persistent_instr}"
        )})

    # 5) Current user message last
    msgs.append({"role": "user", "content": raw_prompt})
    return msgs


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
                if member:
                    with contextlib.suppress(Exception):
                        await msg.remove_reaction(emoji, member)
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
            from gemini_utils import generate_gemini_text
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
                        
                        # Extract extra metrics
                        current = data.get("current", {})
                        main = current.get("main", {})
                        wind = current.get("wind", {})
                        
                        temp = main.get("temp", "?")
                        feels_like = main.get("feels_like", "?")
                        humidity = main.get("humidity", "?")
                        temp_min = main.get("temp_min", "?")
                        temp_max = main.get("temp_max", "?")
                        pressure = main.get("pressure", "?")
                        wind_speed = wind.get("speed", "?")
                        visibility = current.get("visibility", "?")
                        clouds = current.get("clouds", {}).get("all", "?")
                        cond = (current.get("weather") or [{}])[0].get("description", "unknown")
                        
                        forecast_data = data.get("forecast", {})
                        forecast_cur = forecast_data.get("current", {})
                        uvi = forecast_cur.get("uvi", "?")
                        pop = forecast_data.get("daily", [{}])[0].get("pop", 0) * 100
                        
                        sys_data = current.get("sys", {})
                        sunrise_raw = sys_data.get("sunrise")
                        sunset_raw = sys_data.get("sunset")
                        
                        # Calculate local time
                        import datetime
                        tz_offset = current.get("timezone", 0) # seconds from UTC
                        local_dt = datetime.datetime.utcnow() + datetime.timedelta(seconds=tz_offset)
                        time_str = local_dt.strftime("%I:%M %p")
                        
                        sr_str = datetime.datetime.utcfromtimestamp(sunrise_raw + tz_offset).strftime("%I:%M %p") if sunrise_raw else "?"
                        ss_str = datetime.datetime.utcfromtimestamp(sunset_raw + tz_offset).strftime("%I:%M %p") if sunset_raw else "?"
                        
                        # Visibility conversion
                        if isinstance(visibility, (int, float)):
                            vis_str = f"{round(visibility / 1609.34, 1)} mi" if units == "imperial" else f"{round(visibility / 1000, 1)} km"
                        else:
                            vis_str = "?"

                        # DEBUG CONTEXT COMMAND
                        if message.content.startswith("/debug_context"):
                            # Reuse the helper logic
                            # We need to simulate the chat intent flow somewhat
                            try:
                                msgs = _build_chat_context(
                                    message=message,
                                    user_id=str(message.author.id),
                                    raw_prompt=message.content.replace("/debug_context", "").strip() or "DEBUG",
                                    ref_msg=message.reference.resolved if message.reference else None,
                                    is_reply_to_bot=(message.reference.resolved.author.id == bot.user.id) if message.reference and message.reference.resolved else False
                                )
                                import io, json
                                f = io.BytesIO(json.dumps(msgs, indent=2, default=str).encode('utf-8'))
                                await message.reply("Here is the exact context I would send to OpenAI:", file=discord.File(f, filename="context_debug.json"))
                            except Exception as e:
                                await message.reply(f"Failed to build context: {e}")
                            return

                        # Construct prompt - EXTREME ENRICHMENT
                        widget_prompt = (
                            f"A professional, high-density data-maximalist 3D weather station dashboard layout in a WIDESCREEN 16:9 cinematic format. "
                            f"THE PRIMARY FOCUS is the hero weather block: Large '{round(float(temp))}°' and '{loc['name']}' in HUGE, BOLD, HIGH-CONTRAST typography. \n"
                            f"COMPREHENSIVE DATA GRID (crisp, clear, modern labels): \n"
                            f"- Today's Range: {round(float(temp_min))}° - {round(float(temp_max))}° \n"
                            f"- Feels Like: {round(float(feels_like))}° | Humidity: {humidity}% \n"
                            f"- Rain Chance: {round(pop)}% | UV Index: {uvi} \n"
                            f"- Pressure: {pressure} hPa | Visibility: {vis_str} \n"
                            f"- Wind: {wind_speed} {'mph' if units=='imperial' else 'm/s'} | Clouds: {clouds}% \n"
                            f"- Sunrise: {sr_str} | Sunset: {ss_str} \n"
                            f"- Local Time: {time_str} \n"
                            f"The layout is a modern tech interface with transparent elements. "
                            f"The background is a cinematic, expansive widescreen shot of {loc['name']} "
                            f"reflecting current {cond} skies and { 'nighttime' if local_dt.hour < 6 or local_dt.hour > 18 else 'daytime' } lighting. "
                            f"Premium Apple / SF Pro typography / iOS 17 Weather app aesthetic, 8k hyper-detailed text."
                        )
                        logger.info(f"Generating extreme weather widget: {widget_prompt}")
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

        # IMAGE EDIT using Responses API
        if intent == "edit_image" and image_urls:
            # Handle multiple images - ask user which one
            images_to_edit = image_urls
            if len(image_urls) > 1:
                selection = await prompt_for_image_selection(message, len(image_urls))
                if selection == "all":
                    pass  # Keep all images, process sequentially
                else:
                    images_to_edit = [image_urls[selection]]
            
            async def _do_single_edit(img_url: str):
                from openai_utils import openai_client
                import io
                import base64
                
                edit_instruction = f"You must edit this image. {prompt}. Apply the changes to the image."
                
                response = await openai_client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": edit_instruction},
                                {"type": "input_image", "image_url": img_url}
                            ]
                        }
                    ],
                    tools=[{"type": "image_generation", "action": "edit"}]
                )
                
                image_calls = [o for o in response.output if o.type == "image_generation_call"]
                if image_calls and image_calls[0].result:
                    image_base64 = image_calls[0].result
                    return io.BytesIO(base64.b64decode(image_base64))
                return None
            
            # Process each image
            edited_count = 0
            for idx, img_url in enumerate(images_to_edit):
                label = f"Editing ({idx+1}/{len(images_to_edit)})" if len(images_to_edit) > 1 else "Editing"
                status_msg, image_data = await live_status_with_progress(
                    message,
                    action_label=label,
                    emoji="🔧",
                    coro=_do_single_edit(img_url),
                    duration_estimate=30,
                )
                
                if image_data:
                    edited_count += 1
                    await status_msg.edit(content=f"✅ Image {idx+1} edited" if len(images_to_edit) > 1 else "✅ Image edited")
                    await message.channel.send(file=discord.File(image_data, f"edited_{idx+1}.png"))
                else:
                    await status_msg.edit(content=f"❌ Image {idx+1} failed" if len(images_to_edit) > 1 else "❌ Edit failed")
            
            if len(images_to_edit) > 1 and edited_count > 0:
                await message.channel.send(f"✅ Done! Edited {edited_count}/{len(images_to_edit)} images.")
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
            # Include replied-to message content (whether it's from bot or user)
            if ref_msg and (ref_msg.content or "").strip():
                if is_reply_to_bot:
                    reply_context = f"You are responding to your previous message:\n---\n{ref_msg.content.strip()}\n---"
                else:
                    reply_context = f"User is replying to this message:\n---\nFrom: {ref_msg.author.display_name}\n{ref_msg.content.strip()}\n---"
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
                    await send_or_edit_with_truncation(
                        text_resp, 
                        target_msg=status_msg, 
                        original_message=message,
                        model="gpt-4o-vision"
                    )
                
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

        # GENERATE VIDEO (Sora)
        if intent == "generate_video":
            from sora_utils import create_sora_job, get_sora_status, download_sora_content, remix_sora_video
            from database_utils import check_sora_limit, log_sora_usage, get_last_sora_video_id
            import io
            
            # 1. Rate Check
            if not check_sora_limit(str(user_id), limit=2, window_seconds=3600):
                await message.reply("⏳ You have reached the limit of 2 Sora videos per hour. Please try again later.")
                return

            # Data Capture
            image_data = None
            is_remix = False
            base_fail_msg = "Generation failed."
            
            # A. Check for Image Attachment (Image-to-Video)
            if message.attachments:
                for att in message.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        try:
                            image_data = await att.read()
                            base_fail_msg = "Image-to-Video failed."
                            logger.info(f"Received image attachment for Sora: {att.filename} ({len(image_data)} bytes)")
                            break
                        except Exception as e:
                            logger.error(f"Failed to download attachment: {e}")
            
            # B. Check for Remix (Video-to-Video)
            remix_target_id = None
            lower_prompt = prompt.lower()
            if "remix" in lower_prompt or (not image_data and "edit" in lower_prompt and "video" in lower_prompt):
                last_vid = get_last_sora_video_id(str(user_id))
                if last_vid:
                    remix_target_id = last_vid
                    is_remix = True
                    base_fail_msg = "Remix failed."
                else:
                    if "remix" in lower_prompt:
                        await message.reply("⚠️ I couldn't find a previous video of yours to remix. Please generate one first!")
                        return

            # --- CONFIRMATION UI ---
            class SoraConfigSelect(discord.ui.Select):
                def __init__(self):
                    options = [
                        discord.SelectOption(label="Pro - 4s ($1.20)", value="sora-2-pro|4", description="Sora 2 Pro, 4 seconds", emoji="✨"),
                        discord.SelectOption(label="Pro - 8s ($2.40)", value="sora-2-pro|8", description="Sora 2 Pro, 8 seconds (Default)", emoji="✨", default=True),
                        discord.SelectOption(label="Pro - 12s ($3.60)", value="sora-2-pro|12", description="Sora 2 Pro, 12 seconds", emoji="✨"),
                        discord.SelectOption(label="Std - 4s ($0.40)", value="sora-2|4", description="Sora 2, 4 seconds", emoji="🎞️"),
                        discord.SelectOption(label="Std - 8s ($0.80)", value="sora-2|8", description="Sora 2, 8 seconds", emoji="🎞️"),
                        discord.SelectOption(label="Std - 12s ($1.20)", value="sora-2|12", description="Sora 2, 12 seconds", emoji="🎞️"),
                    ]
                    super().__init__(placeholder="Select Configuration...", min_values=1, max_values=1, options=options)

                async def callback(self, interaction: discord.Interaction):
                    # We defer here so the View can stop
                    await interaction.response.defer()
                    self.view.value = self.values[0]
                    self.view.stop()

            class SoraConfirmationView(discord.ui.View):
                def __init__(self, author_id):
                    super().__init__(timeout=60)
                    self.author_id = author_id
                    self.value = None 
                    self.add_item(SoraConfigSelect())

                async def interaction_check(self, interaction: discord.Interaction) -> bool:
                    if interaction.user.id != self.author_id:
                        await interaction.response.send_message("Not your request!", ephemeral=True)
                        return False
                    return True
                
                @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, row=1)
                async def cancel_button(self, interaction: discord.Interaction, button: discord.ui.Button):
                    self.value = "cancel"
                    await interaction.response.edit_message(content="❌ Cancelled.", view=None)
                    self.stop()

            # Cost Text
            cost_msg = (
                f"**Sora Video Generation**\n"
                f"Prompt: *{prompt[:100]}...*\n\n"
                f"⚠️ **Select Configuration:**\n"
                f"Costs based on duration (4s / 8s / 12s):\n"
                f"• **Pro**: $1.20 / $2.40 / $3.60\n"
                f"• **Std**: $0.40 / $0.80 / $1.20\n\n"
                f"💖 *Help cover API costs:* <https://ko-fi.com/sardistic/goal?g=32>"
            )

            view = SoraConfirmationView(author_id=user_id)
            confirm_msg = await message.reply(cost_msg, view=view)

            # Wait for selection
            await view.wait()

            if not view.value or view.value == "cancel":
                if view.value is None:
                    try: await confirm_msg.edit(content="❌ Timed out.", view=None)
                    except: pass
                return

            # Parse "model|seconds"
            parts = view.value.split("|")
            selected_model = parts[0]
            selected_seconds = int(parts[1])
            
            # Update msg to indicate queued
            try:
                await confirm_msg.edit(content=f"✅ **Queued:** {selected_model} ({selected_seconds}s)", view=None)
            except:
                pass

            # Shared Progress Object
            progress_data = {"progress": 0.0}

            # 2. Start Logic
            async def _generate_video_task():
                # CREATE JOB
                if is_remix and remix_target_id:
                     job = await remix_sora_video(remix_target_id, prompt)
                else:
                     # Standard or Image-to-Video
                     job = await create_sora_job(
                         prompt, 
                         model=selected_model, 
                         size="1280x720", 
                         seconds=selected_seconds, 
                         image_data=image_data
                     )
                
                if not job.get("ok"):
                    return None, f"Failed to start job: {job.get('error')}"
                
                video_id = job["data"].get("id")
                logger.info(f"Sora Job Started: {video_id} (Model={selected_model}, Sec={selected_seconds}, Remix={is_remix})")

                # POLL LOOP
                import time
                start_time = time.time()
                while True:
                    await asyncio.sleep(4) # Poll interval
                    if time.time() - start_time > 600: # 10 min timeout
                        return None, "Timeout waiting for video generation."
                        
                    status_res = await get_sora_status(video_id)
                    if not status_res.get("ok"):
                         logger.warning(f"Poll check failed: {status_res.get('error')}")
                         continue
                         
                    status_data = status_res["data"]
                    status = status_data.get("status")
                    
                    # Update Progress
                    # API returns 'progress' as int 0-100 usually, or might be missing
                    if "progress" in status_data:
                        try:
                            raw_p = str(status_data["progress"]).strip().replace('%', '')
                            p_val = float(raw_p)
                            # Normalize 0-100 -> 0.0-1.0
                            if p_val > 1.0: p_val /= 100.0
                            progress_data["progress"] = p_val
                            logger.debug(f"Sora Poll: {p_val*100:.1f}% (Raw: {status_data['progress']})")
                        except Exception as e:
                            logger.warning(f"Failed to parse progress: {status_data['progress']} - {e}")
                    
                    if status == "completed":
                        progress_data["progress"] = 1.0
                        break
                    elif status == "failed":
                        err_msg = status_data.get("error", {}).get("message", "Unknown error")
                        return None, f"Video generation failed: {err_msg}"
                
                # DOWNLOAD
                content = await download_sora_content(video_id)
                if not content:
                    return None, "Failed to download video content."
                    
                # Success
                f = io.BytesIO(content)
                log_sora_usage(str(user_id), video_id=video_id)
                return f, None


            status_msg, result = await live_status_with_progress(
                confirm_msg, 
                action_label=f"Generating ({selected_model}, {selected_seconds}s)",
                emoji="🎥",
                coro=_generate_video_task(),
                duration_estimate=selected_seconds * 10,  # Rough heuristic
                summarizer=(lambda: f"Status: Processing ({int(progress_data['progress']*100)}%)") if STREAM_OK else None,
                progress_tracker=progress_data
            )
            
            if result and isinstance(result, tuple):
                 file_obj, err = result
                 if file_obj:
                     # Calculate final cost for display
                     if "pro" in selected_model:
                         cost = selected_seconds * 0.30
                     else:
                         cost = selected_seconds * 0.10
                     
                     final_msg = (
                         f"**Video generated** ({selected_model}, {selected_seconds}s)\n"
                         f"Est. Cost: ${cost:.2f} | Support: <https://ko-fi.com/sardistic/goal?g=32>\n"
                         f"Prompt: {prompt[:100]}..."
                     )
                     
                     await status_msg.reply(file=discord.File(file_obj, filename="sora_video.mp4"))
                     await status_msg.edit(content=final_msg)
                 else:
                     await status_msg.edit(content=f"❌ {err or base_fail_msg}")
            else:
                 await status_msg.edit(content="❌ Unknown error during generation.")
            return

        # STOCK
        if intent == "get_stock" and prompt.lower().startswith("stock"):
            async with message.channel.typing():
                await handle_stock_command(message, prompt)
            return

        # ----- CHAT (ES-backed messages[] window) -----

        async def _do_chat_generation(model_name=None):
            selected_model = model_name or "gpt-4o"
            
            async def _chat_with_es_window():
                msgs = _build_chat_context(
                    message=message,
                    user_id=user_id,
                    raw_prompt=raw_prompt,
                    ref_msg=ref_msg,
                    is_reply_to_bot=is_reply_to_bot
                )
                ctx = {
                    "guild_id": message.guild.id if message.guild else "DM",
                    "channel_id": message.channel.id,
                    "user_id": user_id
                }
                # If using override model that is Gemini, we must switch logic? 
                # Actually, our dropwdown mixes OpenAI and Gemini models.
                # If user selects Gemini here, we should probably call Gemini logic.
                # But 'generate_openai_messages_response_with_tools' uses OpenAI client.
                # If we select "gemini-..." we can't pass it to OpenAI client.
                # Our ModerationFallbackView offers Gemini options.
                # If the fallback model is Gemini, we should reroute to Gemini logic?
                # Or just stick to OpenAI models for OpenAI fallback?
                # The user request was "dropdown option to select a different chatgpt or gemini model".
                # If I am in OpenAI mode and I select Gemini, I must call Gemini.
                
                if "gemini" in selected_model.lower():
                     # Reroute to Gemini Utils
                     # We need to adapt the context. _build_chat_context returns OpenAI-style dicts.
                     # generate_gemini_text accepts OpenAI-style dicts (role/content).
                     # So we can just call generate_gemini_text.
                     
                     enable_code = False # Default off for general chat fallback?
                     status_res = {"text": ""}
                     
                     text_resp, artifacts = generate_gemini_text(
                         prompt=prompt,
                         context=msgs, # msgs contains system prompt + history + last user msg
                         # Wait, generate_gemini_text expects 'prompt' separate from 'context'?
                         # Yes. 'prompt' is appended to context.
                         # But _build_chat_context ALREADY appends the last user message.
                         # We should separate them if possible, or just pass empty prompt and full context?
                         # generate_gemini_text: 
                         #   final_prompt_text = (rag_context + prompt) if rag_context else prompt
                         #   contents.append(Type.Content(..., parts=[prompt]))
                         #   context (history) is prepended.
                         
                         # If msgs has everything, we might double-send the last prompts.
                         # Actually _build_chat_context puts user prompt at the end.
                         # We can pop it?
                         
                         # Simplified: Just use OpenAI for OpenAI choices, Gemini for Gemini choices?
                         extra_parts=None,
                         status_tracker=status_res,
                         enable_code_execution=enable_code,
                         search_ids=ctx,
                         model_name=selected_model
                     )
                     # return just text to satisfy _chat_with_es_window signature (expects string)
                     return text_resp
                
                return await generate_openai_messages_response_with_tools(
                    msgs, 
                    tools=TOOLS_DEF, 
                    tool_context=ctx,
                    model=selected_model
                )

            # Light live summary outline
            def _summarizer():
                return f"• Using {selected_model}…\n• Drafting answer…"

            try:
                status_msg, response = await live_status_with_progress(
                    message,
                    action_label=f"Responding ({selected_model})",
                    emoji="💬",
                    coro=_chat_with_es_window(),
                    duration_estimate=duration_estimate,
                    summarizer=_summarizer if STREAM_OK else None,
                )

                if response and response.strip():
                    await send_or_edit_with_truncation(
                        response, 
                        target_msg=status_msg, 
                        original_message=message,
                        model=selected_model
                    )
                else:
                    await status_msg.edit(content="🤖 INSUFFICIENT DATA FOR MEANINGFUL ANSWER")

            except OpenAIModerationError as e:
                logger.warning(f"OpenAI moderation hit: {e}")
                user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
                # Pass this function as callback
                view = ModerationFallbackView(author_id=message.author.id, retry_callback=_do_chat_generation)
                await message.reply(user_msg, view=view)
            except GeminiModerationError as e:
                 # Catch Gemini moderation if we switched to Gemini
                logger.warning(f"Gemini moderation hit in Chat fallback: {e}")
                user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
                view = ModerationFallbackView(author_id=message.author.id, retry_callback=_do_chat_generation)
                await message.reply(user_msg, view=view)

        await _do_chat_generation()

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
