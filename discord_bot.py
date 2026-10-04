# discord_bot.py
# Discord bot triggered by mentions/replies: classifies intent, then routes to
# chat (bounded multi-speaker channel context plus per-user memory), search, weather/stock, URL summarize,
# image/video generation and editing. Replies use a live progress bar and
# expand/collapse UI.

from __future__ import annotations

import os
import re
import io
import logging
import mimetypes
import asyncio
import contextlib
import collections
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from datetime import datetime, timezone

import discord
from discord.ext import commands

CHANGE_DASHBOARD_URL = "https://sardistic.github.io/Multivac-Refactored/"

# IMPORTANT: importing config mirrors GCE metadata → os.environ for all keys
# Importing GOOGLE_* here ensures that side-effect runs even if this file
# doesn't directly use the variables. (We also log simple configured yes/no checks.)
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
from services.database_utils import (
    activate_behavior_change,
    get_message_expansion,
    get_channel_last_seen,
    set_channel_last_seen,
    get_user_seen,
    set_user_seen,
    list_user_facts,
    delete_user_facts_matching,
    get_user_profile,
    get_behavior_change,
    list_behavior_changes,
    propose_behavior_change,
    rollback_behavior_change,
    create_code_proposal,
    get_code_proposal,
    get_any_code_proposal,
    get_code_deployment,
    list_code_proposals,
    list_all_code_proposals,
    review_code_proposal,
    review_any_code_proposal,
    set_code_proposal_patch,
    set_code_proposal_validation,
    set_code_proposal_approval_message,
    request_code_rollback,
    request_any_code_rollback,
    set_conversation_persona_enabled,
)
from services.code_changes import MAX_PATCH_BYTES, get_baseline_sha, validate_patch
from services.code_generator import (
    IDEA_CODE_MODEL,
    generate_code_patch,
    generate_planned_idea_patch,
)
from services.command_control import CommandControlWorker
from services.command_runtime import CommandModuleLoader
from services.behavior_control import BehaviorControlWorker
from services.behavior_registry import BEHAVIOR_REGISTRY, dispatch_event
from services.behavior_runtime import BehaviorModuleLoader
from services.tool_control import ToolControlWorker
from services.tools_registry import TOOL_REGISTRY, ToolModuleLoader
from services import usage_costs
from services.reflection_worker import ReflectionErrorHandler, ReflectionWorker
from services.progress import (
    pick_style,
    render_progress_status,
    render_stage_checklist,
    start_progress_bar,
)
from services.security_limits import check_rate_limit
from services.security_utils import public_error_detail, sanitize_diagnostic_text
from services.user_profile import maybe_refresh_profile
from providers.claude_utils import ANTHROPIC_API_KEY
from bot.intent_dispatcher import (
    DispatchContext,
    dispatch_intent,
    resolve_keyword_intent,
    validate_classified_intent,
)
from bot.channel_context import fetch_recent_channel_context
from bot.message_inputs import (
    collect_gemini_parts,
    collect_image_inputs,
    has_visual_inputs,
    resolve_reference_message,
    strip_mention_and_trigger,
)
from bot.moderation_view import ModerationFallbackView
from bot.persona import PERSONA_NAME, message_persona_scope, parse_persona_toggle
from bot.response_policy import should_suppress_response
from bot.ui_messages import (
    EXPAND_EMOJI,
    COLLAPSE_EMOJI,
    handle_expansion_reaction,
    send_or_edit_with_truncation as ui_send_or_edit_with_truncation,
    live_status_with_progress as ui_live_status_with_progress,
)

# The shared progress renderer now folds live summaries into the same
# rate-limited edit loop, so no optional editor dependency is required.
STREAM_OK = True

# ---- Logging ----
# Configured centrally by services.logging_config (called from main.py).
logger = logging.getLogger("discord_bot")


def _configured(value: Optional[str]) -> str:
    return "yes" if bool(value) else "no"


logger.info(
    "Google CSE configured: GOOGLE_API_KEY=%s, GOOGLE_CSE_ID=%s",
    _configured(GOOGLE_API_KEY),
    _configured(GOOGLE_CSE_ID),
)
logger.info("Anthropic configured: ANTHROPIC_API_KEY=%s", _configured(ANTHROPIC_API_KEY))

# ---- Discord ----
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="/", intents=intents)

_tool_hotload_root = (os.getenv("MULTIVAC_TOOL_HOTLOAD_DIR") or "").strip()
_tool_module_loader = (
    ToolModuleLoader(TOOL_REGISTRY, _tool_hotload_root)
    if _tool_hotload_root
    else None
)
_tool_control_root = (os.getenv("MULTIVAC_TOOL_CONTROL_DIR") or "").strip()
_tool_control_worker = (
    ToolControlWorker(_tool_module_loader, _tool_control_root)
    if _tool_module_loader is not None and _tool_control_root
    else None
)
_tool_control_task: asyncio.Task | None = None
_command_module_loader = (
    CommandModuleLoader(bot, _tool_hotload_root) if _tool_hotload_root else None
)
_command_control_worker = (
    CommandControlWorker(_command_module_loader, _tool_control_root)
    if _command_module_loader is not None and _tool_control_root
    else None
)
_command_control_task: asyncio.Task | None = None
_behavior_module_loader = (
    BehaviorModuleLoader(BEHAVIOR_REGISTRY, _tool_hotload_root, bot=bot)
    if _tool_hotload_root
    else None
)
_behavior_control_worker = (
    BehaviorControlWorker(_behavior_module_loader, _tool_control_root)
    if _behavior_module_loader is not None and _tool_control_root
    else None
)
_behavior_control_task: asyncio.Task | None = None
_reflection_worker: ReflectionWorker | None = None
_reflection_task: asyncio.Task | None = None
_usage_metrics_task: asyncio.Task | None = None
_reflection_error_handler: ReflectionErrorHandler | None = None

# ---- Backfill high-water marks (per guild:channel), stored in SQLite ----
def _state_key(guild_id: int | None, channel_id: int) -> str:
    g = str(guild_id) if guild_id else "DM"
    return f"{g}:{channel_id}"

# ---- Multi-Image Selection ----
# Track messages recently processed to prevent gateway replays / feedback loops
_processed_msg_ids: "collections.OrderedDict[int, None]" = collections.OrderedDict()
_PROCESSED_CACHE_SIZE = 1000

def _already_processed(message_id: int) -> bool:
    """Record message_id; return True if it was already seen recently."""
    if message_id in _processed_msg_ids:
        return True
    _processed_msg_ids[message_id] = None
    while len(_processed_msg_ids) > _PROCESSED_CACHE_SIZE:
        _processed_msg_ids.popitem(last=False)
    return False

# Track users currently being prompted (to prevent on_message from double-processing)
_pending_image_selection: set[int] = set()  # user IDs awaiting reply

# Track message IDs currently being expanded to prevent race conditions
_expansion_locks: set[int] = set()
# Track preflight status messages so progress can start immediately.
_preflight_status_by_message_id: dict[int, discord.Message] = {}


# The preflight path has genuinely discrete stages, so it renders them as a
# checklist. A smooth bar across four steps invents continuity that is not
# there and hides which stage a stall happened in.
PREFLIGHT_STAGES = (
    "Opening the response path",
    "Indexing fresh context",
    "Reading intent and available inputs",
    "Routing to the right capability",
)


def _preflight_status(step: int, detail: str, total: int = 3) -> str:
    current = max(0, min(int(step), len(PREFLIGHT_STAGES)))
    body = render_stage_checklist(PREFLIGHT_STAGES, current, phase=current)
    if detail and current < len(PREFLIGHT_STAGES):
        return f"{body}\n↳ {str(detail).strip()[:900]}"
    return body


async def _run_context_progress(
    ctx,
    *,
    action_label,
    emoji: str,
    coro,
    duration_estimate: int,
    indeterminate: bool = False,
):
    """Animate an owner-triggered command, then let the caller replace it."""
    if ctx.interaction:
        await ctx.defer()
    seed = getattr(getattr(ctx, "message", None), "id", None)
    style = "barberpole" if indeterminate else pick_style(seed)
    try:
        status_msg = await ctx.reply(
            render_progress_status(
                action_label,
                emoji=emoji,
                detail="Starting the reviewed operation…",
                style=style,
            )
        )
    except Exception:
        if asyncio.iscoroutine(coro):
            coro.close()
        raise
    task = asyncio.create_task(coro)
    progress_task = asyncio.create_task(
        start_progress_bar(
            status_msg,
            task,
            action_label=action_label,
            emoji=emoji,
            duration_estimate=duration_estimate,
            style=style,
        )
    )
    try:
        result = await task
    except Exception:
        with contextlib.suppress(Exception):
            await progress_task
        with contextlib.suppress(Exception):
            await status_msg.delete()
        raise
    finally:
        if not progress_task.done():
            with contextlib.suppress(Exception):
                await progress_task
    return status_msg, result


async def prompt_for_image_selection(message, image_count: int, timeout: float = 30.0):
    """
    Ask user which image to process when multiple are present.
    Returns: int (0-based index), "all", or 0 on timeout.

    Only messages that look like a selection (a number or "all") are consumed;
    anything else the user says in the meantime is left for normal handling.
    """
    user_id = message.author.id
    _pending_image_selection.add(user_id)

    try:
        prompt_msg = await message.reply(
            f"📷 I see **{image_count} images**. Which one should I edit?\n"
            "Reply with a number (1, 2, ...) or **all**."
        )

        def check(m):
            if m.author.id != user_id or m.channel != message.channel:
                return False
            text = m.content.strip().lower()
            return text == "all" or text.isdigit()

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                reply = await bot.wait_for("message", check=check, timeout=remaining)
            except asyncio.TimeoutError:
                await prompt_msg.edit(content="⏰ Timed out. Using the first image.")
                return 0
            text = reply.content.strip().lower()
            if text == "all":
                return "all"
            idx = int(text) - 1
            if 0 <= idx < image_count:
                return idx
            await message.channel.send(
                f"⚠️ Pick a number between 1 and {image_count}, or **all**."
            )
    except asyncio.TimeoutError:
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

async def _index_message_async(**kwargs) -> Optional[str]:
    """Run the synchronous OpenSearch indexing call off the event loop."""
    return await asyncio.to_thread(index_message, **kwargs)


async def _auto_index_bot_message(sent_message, full_text: str, *, original_message=None, reply_to=None, model: Optional[str] = None):
    src_msg = original_message or reply_to
    if not src_msg:
        return
    try:
        await _index_message_async(
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
    try:
        await _get_reflection_worker().observe_message(
            guild_id=str(src_msg.guild.id) if src_msg.guild else "DM",
            channel_id=str(src_msg.channel.id),
            author_id=str(bot.user.id) if bot.user else "assistant",
            message_id=str(sent_message.id),
            content=full_text,
            role="assistant",
        )
    except Exception:
        logger.warning("Unable to extend reflection session for bot reply", exc_info=True)


async def send_or_edit_with_truncation(*args, **kwargs):
    kwargs.setdefault("index_callback", _auto_index_bot_message)
    return await ui_send_or_edit_with_truncation(*args, **kwargs)


async def live_status_with_progress(*args, **kwargs):
    base_message = args[0] if args else kwargs.get("message")
    if isinstance(base_message, discord.Message):
        explicit_status_msg = kwargs.get("existing_status_msg")
        existing = _preflight_status_by_message_id.pop(base_message.id, None)
        if existing is not None:
            if explicit_status_msg is not None:
                with contextlib.suppress(Exception):
                    await existing.delete()
            else:
                kwargs.setdefault("existing_status_msg", existing)
    kwargs.setdefault("stream_ok", STREAM_OK)
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
      - Uses a channel-scoped 'last_seen_id' (stored in SQLite) as a high-water mark.
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

    # Read the high-water mark
    key = _state_key(guild_id, channel_id)
    last_seen_id = await asyncio.to_thread(get_channel_last_seen, key)

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
            await _index_message_async(
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
            logger.exception("Recent backfill index error")
            continue

    if max_id_seen and max_id_seen != (int(last_seen_id) if last_seen_id else 0):
        await asyncio.to_thread(set_channel_last_seen, key, str(max_id_seen))

    return indexed

# --------------------------
# Events
# --------------------------

_app_commands_synced = False


async def _run_tool_control() -> None:
    try:
        restored = await asyncio.to_thread(_tool_control_worker.restore_active)
        if restored["errors"]:
            logger.error("Live tool restore errors: %s", restored["errors"])
        elif restored["restored"]:
            logger.info("Restored live tool sources: %s", restored["restored"])
    except Exception:
        logger.exception("Unable to restore active live tools")

    last_request_id = None
    while not bot.is_closed():
        if not bot.is_ready():
            await asyncio.sleep(2)
            continue
        try:
            result = await asyncio.to_thread(_tool_control_worker.process_once)
            if result and result.get("request_id") != last_request_id:
                last_request_id = result.get("request_id")
                if result.get("ok"):
                    logger.warning(
                        "Supervisor tool activation completed request=%s generation=%s",
                        last_request_id,
                        result.get("generation"),
                    )
                else:
                    logger.error(
                        "Supervisor tool activation failed request=%s detail=%s",
                        last_request_id,
                        result.get("detail"),
                    )
        except Exception:
            logger.exception("Live tool control poll failed")
        await asyncio.sleep(2)


async def _run_command_control() -> None:
    try:
        restored = await _command_control_worker.restore_active()
        if restored["errors"]:
            logger.error("Live command restore errors: %s", restored["errors"])
        elif restored["restored"]:
            logger.info("Restored live command sources: %s", restored["restored"])
    except Exception:
        logger.exception("Unable to restore active live commands")

    last_request_id = None
    while not bot.is_closed():
        if not bot.is_ready():
            await asyncio.sleep(2)
            continue
        try:
            result = await _command_control_worker.process_once()
            if result and result.get("request_id") != last_request_id:
                last_request_id = result.get("request_id")
                if result.get("ok"):
                    logger.warning(
                        "Supervisor command activation completed request=%s generation=%s synced=%s",
                        last_request_id,
                        result.get("generation"),
                        result.get("synced_commands"),
                    )
                else:
                    logger.error(
                        "Supervisor command activation failed request=%s detail=%s",
                        last_request_id,
                        result.get("detail"),
                    )
        except Exception:
            logger.exception("Live command control poll failed")
        await asyncio.sleep(2)


async def _run_behavior_control() -> None:
    try:
        restored = await _behavior_control_worker.restore_active()
        if restored["errors"]:
            logger.error("Live behavior restore errors: %s", restored["errors"])
        elif restored["restored"]:
            logger.info("Restored live behavior sources: %s", restored["restored"])
    except Exception:
        logger.exception("Unable to restore active live behaviors")

    last_request_id = None
    while not bot.is_closed():
        if not bot.is_ready():
            await asyncio.sleep(2)
            continue
        try:
            result = await _behavior_control_worker.process_once()
            if result and result.get("request_id") != last_request_id:
                last_request_id = result.get("request_id")
                if result.get("ok"):
                    logger.warning(
                        "Supervisor behavior activation completed request=%s generation=%s",
                        last_request_id,
                        result.get("generation"),
                    )
                else:
                    logger.error(
                        "Supervisor behavior activation failed request=%s detail=%s",
                        last_request_id,
                        result.get("detail"),
                    )
        except Exception:
            logger.exception("Live behavior control poll failed")
        await asyncio.sleep(2)


async def _fetch_reflection_channel_window(session: dict) -> list[dict]:
    """Fetch ephemeral surrounding chat for one already-authorized session."""
    channel = bot.get_channel(int(session["channel_id"]))
    if channel is None:
        try:
            channel = await bot.fetch_channel(int(session["channel_id"]))
        except Exception:
            channel = None
    if channel is None or not hasattr(channel, "history"):
        return await asyncio.to_thread(
            ReflectionWorker._fetch_indexed_evidence, session
        )

    try:
        after = datetime.fromisoformat(session["started_at"])
        before = datetime.fromisoformat(session["expires_at"])
        rows = []
        async for msg in channel.history(
            limit=100, after=after, before=before, oldest_first=False
        ):
            if msg.author.bot and (not bot.user or msg.author.id != bot.user.id):
                continue
            content = (msg.content or "").strip()
            if not content:
                continue
            if bot.user and msg.author.id == bot.user.id:
                role = "assistant"
            elif str(msg.author.id) == str(session["user_id"]):
                role = "requester"
            else:
                role = "participant"
            rows.append(
                {"message_id": str(msg.id), "role": role, "content": content}
            )
        rows.reverse()
        return rows
    except Exception:
        logger.warning(
            "reflection surrounding-history fetch failed for session %s",
            session.get("id"),
            exc_info=True,
        )
        return await asyncio.to_thread(
            ReflectionWorker._fetch_indexed_evidence, session
        )


def _get_reflection_worker() -> ReflectionWorker:
    global _reflection_worker, _reflection_error_handler
    if _reflection_worker is None:
        _reflection_worker = ReflectionWorker(
            history_fetcher=_fetch_reflection_channel_window
        )
    if (
        _reflection_worker.enabled
        and _reflection_error_handler is None
    ):
        _reflection_error_handler = ReflectionErrorHandler(_reflection_worker)
        logging.getLogger().addHandler(_reflection_error_handler)
    return _reflection_worker


async def _run_usage_metrics_publisher() -> None:
    while True:
        await asyncio.to_thread(usage_costs.publish_metrics_snapshot)
        await asyncio.sleep(300)


@bot.event
async def on_ready():
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="Graphs Go BRRR 📈")
    )
    logger.info("Bot is online and ready!")

    # Register slash (application) commands with Discord. Global sync; if the
    # commands don't appear, the bot needs re-inviting with the
    # 'applications.commands' OAuth scope.
    global _app_commands_synced, _tool_control_task, _command_control_task, _behavior_control_task, _reflection_task, _usage_metrics_task
    if usage_costs.METRICS_PATH and (
        _usage_metrics_task is None or _usage_metrics_task.done()
    ):
        _usage_metrics_task = asyncio.create_task(
            _run_usage_metrics_publisher(), name="multivac-usage-metrics"
        )
    if _tool_control_worker is not None and (
        _tool_control_task is None or _tool_control_task.done()
    ):
        _tool_control_task = asyncio.create_task(
            _run_tool_control(), name="multivac-tool-control"
        )
    if not _app_commands_synced:
        try:
            synced = await bot.tree.sync()
            _app_commands_synced = True
            logger.info("Synced %d application commands: %s", len(synced), [c.name for c in synced])
        except Exception:
            logger.exception("Application command sync failed")
    if _command_control_worker is not None and (
        _command_control_task is None or _command_control_task.done()
    ):
        _command_control_task = asyncio.create_task(
            _run_command_control(), name="multivac-command-control"
        )
    if _behavior_control_worker is not None and (
        _behavior_control_task is None or _behavior_control_task.done()
    ):
        _behavior_control_task = asyncio.create_task(
            _run_behavior_control(), name="multivac-behavior-control"
        )
    reflection = _get_reflection_worker()
    if reflection.enabled and (
        _reflection_task is None or _reflection_task.done()
    ):
        _reflection_task = asyncio.create_task(
            reflection.run_forever(), name="multivac-reflection"
        )

async def _builtin_on_raw_reaction_add(payload):
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
            # Can't tell if it's a bot; process anyway rather than drop the reaction.
            logger.warning("Could not resolve reacting user %s; processing anyway", payload.user_id)

    # Check if this message is a truncatable message (fast DB check)
    rec = await asyncio.to_thread(get_message_expansion, payload.message_id)
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


@bot.event
async def on_raw_reaction_add(payload):
    return await dispatch_event("raw_reaction_add", _builtin_on_raw_reaction_add, payload)


async def _builtin_on_command_error(ctx, error):
    """Surface command failures to the user instead of hanging a deferred
    interaction on silent 'thinking'."""
    root = getattr(error, "original", None) or error.__cause__ or error
    logger.error(
        "Command %s failed: %r",
        getattr(ctx.command, "name", "?"),
        root,
        exc_info=root,
    )
    with contextlib.suppress(Exception):
        await ctx.reply(
            f"❌ `{getattr(ctx.command, 'name', '?')}` failed because "
            f"{public_error_detail(root)}. The detailed error was recorded privately."
        )


@bot.event
async def on_command_error(ctx, error):
    return await dispatch_event("command_error", _builtin_on_command_error, ctx, error)


# --------------------------
# Commands
# --------------------------

@bot.hybrid_command(name="ping", description="Check that the bot is alive.")
async def ping(ctx):
    await ctx.reply("pong")


@bot.hybrid_command(
    name="tool_hotload",
    description="Owner: explicitly activate, unload, roll back, or inspect a trusted tool module.",
    with_app_command=False,
)
@commands.is_owner()
async def tool_hotload(
    ctx,
    action: str = "status",
    relative_path: str = "",
    allow_overrides: bool = False,
):
    """Operate the explicit live-tool loader; no directory is watched automatically."""
    action = action.strip().lower()
    if action == "status":
        status = TOOL_REGISTRY.status()
        sources = status["sources"]
        source_lines = [
            f"- `{item['source_id']}` · version `{item['version'] or 'inactive'}` · "
            f"history {item['history']}"
            for item in sources
        ]
        response = (
            f"Tool registry generation `{status['generation']}` · {len(status['tools'])} tools\n"
            + ("\n".join(source_lines) if source_lines else "No tool sources registered.")
        )
        await ctx.reply(response[:1900])
        return

    if _tool_module_loader is None:
        await ctx.reply(
            "❌ Tool hotloading is disabled. Set `MULTIVAC_TOOL_HOTLOAD_DIR` to a "
            "supervisor-controlled, read-only artifact directory and restart once."
        )
        return
    if _tool_control_worker is not None:
        await ctx.reply(
            "❌ Direct tool mutation is disabled while supervisor control is active. "
            "Use the reviewed `live_tools/*.py` proposal flow."
        )
        return
    if not relative_path:
        await ctx.reply("❌ Provide a `.py` path relative to the configured hotload directory.")
        return

    try:
        if action in {"load", "reload", "activate"}:
            result = await asyncio.to_thread(
                _tool_module_loader.activate,
                relative_path,
                allow_overrides=allow_overrides,
            )
        elif action == "unload":
            result = await asyncio.to_thread(_tool_module_loader.unload, relative_path)
        elif action == "rollback":
            result = await asyncio.to_thread(_tool_module_loader.rollback, relative_path)
        else:
            await ctx.reply("❌ Action must be `status`, `load`, `reload`, `unload`, or `rollback`.")
            return
    except Exception as exc:
        logger.warning(
            "Owner tool hotload failed action=%s path=%s: %s",
            action,
            relative_path,
            exc,
            exc_info=True,
        )
        await ctx.reply(f"❌ Tool hotload failed: `{type(exc).__name__}: {str(exc)[:500]}`")
        return

    logger.warning(
        "Owner tool hotload action=%s path=%s generation=%s actor=%s",
        action,
        relative_path,
        result["generation"],
        ctx.author.id,
    )
    tools = result.get("tools")
    suffix = f" · tools `{', '.join(tools)}`" if tools else ""
    await ctx.reply(
        f"✅ Tool `{action}` complete at registry generation `{result['generation']}`{suffix}."
    )


@bot.hybrid_command(
    name="command_hotload",
    description="Owner: explicitly activate, unload, roll back, or inspect a trusted command Cog.",
    with_app_command=False,
)
@commands.is_owner()
async def command_hotload(
    ctx,
    action: str = "status",
    relative_path: str = "",
    allow_overrides: bool = False,
):
    action = action.strip().lower()
    if action == "status":
        status = (
            _command_module_loader.status()
            if _command_module_loader is not None
            else {"generation": 0, "sources": []}
        )
        lines = [
            f"- `{item['source_id']}` · version `{item['version'] or 'inactive'}` · "
            f"cogs `{', '.join(item['cogs']) or '-'}` · history {item['history']}"
            for item in status["sources"]
        ]
        response = f"Command registry generation `{status['generation']}`\n" + (
            "\n".join(lines) if lines else "No live command sources registered."
        )
        await ctx.reply(response[:1900])
        return
    if _command_module_loader is None:
        await ctx.reply("❌ Command hotloading is disabled without a configured artifact root.")
        return
    if _command_control_worker is not None:
        await ctx.reply(
            "❌ Direct command mutation is disabled while supervisor control is active. "
            "Use the reviewed `live_commands/*.py` proposal flow."
        )
        return
    if not relative_path:
        await ctx.reply("❌ Provide a `.py` path relative to the configured hotload directory.")
        return
    try:
        if action in {"load", "reload", "activate"}:
            result = await _command_module_loader.activate(
                relative_path, allow_overrides=allow_overrides
            )
        elif action == "unload":
            result = await _command_module_loader.unload_source(
                _command_module_loader.source_id(relative_path)
            )
        elif action == "rollback":
            result = await _command_module_loader.rollback_source(
                _command_module_loader.source_id(relative_path)
            )
        else:
            await ctx.reply("❌ Action must be `status`, `load`, `reload`, `unload`, or `rollback`.")
            return
        await bot.tree.sync()
    except Exception as exc:
        logger.warning(
            "Owner command hotload failed action=%s path=%s: %s",
            action,
            relative_path,
            exc,
            exc_info=True,
        )
        await ctx.reply(f"❌ Command hotload failed: `{type(exc).__name__}: {str(exc)[:500]}`")
        return
    logger.warning(
        "Owner command hotload action=%s path=%s generation=%s actor=%s",
        action,
        relative_path,
        result["generation"],
        ctx.author.id,
    )
    await ctx.reply(
        f"✅ Command `{action}` complete at generation `{result['generation']}`."
    )


@bot.hybrid_command(
    name="behavior_hotload",
    description="Owner: activate, unload, roll back, or inspect a trusted behavior component.",
    with_app_command=False,
)
@commands.is_owner()
async def behavior_hotload(
    ctx,
    action: str = "status",
    relative_path: str = "",
    allow_overrides: bool = False,
):
    action = action.strip().lower()
    if action == "status":
        status = BEHAVIOR_REGISTRY.status()
        lines = [
            f"- `{item['source_id']}` · version `{item['version'] or 'inactive'}` · "
            f"history {item['history']}"
            for item in status["sources"]
        ]
        handlers = ", ".join(
            f"{kind}={len(names)}" for kind, names in status["handlers"].items()
        )
        response = (
            f"Behavior generation `{status['generation']}` · {handlers}\n"
            + ("\n".join(lines) if lines else "No live behavior sources registered.")
        )
        await ctx.reply(response[:1900])
        return
    if _behavior_module_loader is None:
        await ctx.reply("❌ Behavior hotloading is disabled without a configured artifact root.")
        return
    if _behavior_control_worker is not None:
        await ctx.reply(
            "❌ Direct behavior mutation is disabled while supervisor control is active. "
            "Use the reviewed `live_components/*.py` proposal flow."
        )
        return
    if not relative_path:
        await ctx.reply("❌ Provide a `.py` path relative to the configured hotload directory.")
        return
    try:
        source_id = _behavior_module_loader.source_id(relative_path)
        if action in {"load", "reload", "activate"}:
            result = await _behavior_module_loader.activate(
                relative_path, allow_overrides=allow_overrides
            )
        elif action == "unload":
            result = await _behavior_module_loader.unload_source(source_id)
        elif action == "rollback":
            result = await _behavior_module_loader.rollback_source(source_id)
        else:
            await ctx.reply("❌ Action must be `status`, `load`, `reload`, `unload`, or `rollback`.")
            return
    except Exception as exc:
        logger.warning(
            "Owner behavior hotload failed action=%s path=%s: %s",
            action,
            relative_path,
            exc,
            exc_info=True,
        )
        await ctx.reply(f"❌ Behavior hotload failed: `{type(exc).__name__}: {str(exc)[:500]}`")
        return
    logger.warning(
        "Owner behavior hotload action=%s path=%s generation=%s actor=%s",
        action,
        relative_path,
        result["generation"],
        ctx.author.id,
    )
    await ctx.reply(
        f"✅ Behavior `{action}` complete at generation `{result['generation']}`."
    )


def _behavior_change_summary(change: dict) -> str:
    instruction = change.get("instruction") or "_(baseline behavior; no custom instruction)_"
    if len(instruction) > 900:
        instruction = instruction[:897] + "..."
    parent = f" · parent `#{change['parent_id']}`" if change.get("parent_id") else ""
    return (
        f"**Behavior change `#{change['id']}`** · `{change['status']}`{parent}\n"
        f"> {instruction}"
    )


@bot.hybrid_command(name="behavior_propose", description="Draft a personal behavior change without activating it.")
async def behavior_propose(ctx, *, instruction: str):
    """Create an auditable draft. Use --clear to propose returning to baseline."""
    instruction = instruction.strip()
    if instruction.lower() == "--clear":
        instruction = ""
    change_id = await asyncio.to_thread(
        propose_behavior_change,
        str(ctx.author.id),
        instruction,
        created_by=str(ctx.author.id),
    )
    change = await asyncio.to_thread(get_behavior_change, str(ctx.author.id), change_id)
    await ctx.reply(
        f"🧪 Drafted, but **not active**.\n{_behavior_change_summary(change)}\n"
        f"Activate with `/behavior_activate {change_id}` or leave it as a draft."
    )


@bot.hybrid_command(name="behavior_show", description="Show one of your proposed behavior changes.")
async def behavior_show(ctx, change_id: int):
    change = await asyncio.to_thread(get_behavior_change, str(ctx.author.id), change_id)
    if not change:
        await ctx.reply("That behavior change does not exist in your personal history.")
        return
    await ctx.reply(_behavior_change_summary(change))


@bot.hybrid_command(name="behavior_history", description="List your recent behavior changes and their states.")
async def behavior_history(ctx, limit: int = 10):
    changes = await asyncio.to_thread(list_behavior_changes, str(ctx.author.id), limit)
    if not changes:
        await ctx.reply("You have no versioned behavior changes yet.")
        return
    lines = ["🧾 **Your behavior-change history**"]
    for change in changes:
        preview = (change.get("instruction") or "baseline behavior").replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        lines.append(f"`#{change['id']}` **{change['status']}** — {preview}")
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(name="behavior_activate", description="Activate one of your drafted behavior changes.")
async def behavior_activate(ctx, change_id: int):
    try:
        change = await asyncio.to_thread(
            activate_behavior_change, str(ctx.author.id), change_id
        )
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    await ctx.reply(f"✅ Activated immediately.\n{_behavior_change_summary(change)}")


@bot.hybrid_command(name="behavior_rollback", description="Roll back your active behavior change to its parent.")
async def behavior_rollback(ctx):
    restored = await asyncio.to_thread(rollback_behavior_change, str(ctx.author.id))
    if restored:
        await ctx.reply(f"↩️ Rolled back and restored:\n{_behavior_change_summary(restored)}")
    else:
        await ctx.reply("↩️ Rolled back to baseline behavior. No custom instruction is active.")


def _code_proposal_summary(proposal: dict) -> str:
    request = proposal["request"].replace("\n", " ")
    if len(request) > 600:
        request = request[:597] + "..."
    validation = proposal.get("validation")
    checks = "not run"
    generator = None
    if validation:
        checks = "passed" if validation.get("ok") else f"failed ({len(validation.get('errors', []))} errors)"
        generator = (validation.get("generation") or {}).get("model")
    generator_text = f" · generator: `{generator}`" if generator else ""
    return (
        f"**Code proposal `{proposal.get('public_id') or proposal['id']}`** · `{proposal['status']}`\n"
        f"Baseline: `{proposal['baseline_sha'][:12]}` · validation: **{checks}**{generator_text}\n"
        f"> {request}"
    )


def _proposal_id_from_text(text: str) -> int | str | None:
    passphrase = re.search(
        r"\b([a-z][a-z0-9]*(?:-[a-z][a-z0-9]*){1,2})\b",
        text or "",
        re.IGNORECASE,
    )
    if passphrase:
        return passphrase.group(1).lower()
    match = re.search(r"(?:proposal\s*)?#?(\d+)\b", text or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


class CodeGenerationConfirmationView(discord.ui.View):
    def __init__(self, author_id: int):
        super().__init__(timeout=60)
        self.author_id = int(author_id)
        self.confirmed: bool | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "This code-generation request belongs to someone else.",
                ephemeral=True,
            )
            return False
        return True

    @discord.ui.button(label="Generate proposal", style=discord.ButtonStyle.success)
    async def confirm_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.confirmed = True
        await interaction.response.edit_message(
            content="✅ Confirmed. Starting the proposal and code-generation pass…",
            view=None,
        )
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel_button(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ):
        self.confirmed = False
        await interaction.response.edit_message(
            content="❌ Code proposal cancelled before any generation tokens were spent.",
            view=None,
        )
        self.stop()


async def _confirm_code_generation(
    target,
    *,
    author_id: int,
    request_summary: str,
    existing_status=None,
) -> bool:
    if existing_status is not None:
        with contextlib.suppress(Exception):
            await existing_status.delete()
    summary = " ".join((request_summary or "").split())
    if len(summary) > 700:
        summary = summary[:697] + "..."
    view = CodeGenerationConfirmationView(author_id)
    confirmation = await target.reply(
        "⚠️ **Generate a code proposal?**\n"
        "This will invoke the code-generation model and spend API tokens. "
        "Nothing will be approved or deployed automatically.\n"
        f"> {summary}",
        view=view,
    )
    await view.wait()
    if view.confirmed is None:
        with contextlib.suppress(Exception):
            await confirmation.edit(
                content="⌛ Code proposal confirmation timed out. No generation tokens were spent.",
                view=None,
            )
    return view.confirmed is True


async def _natural_code_proposal(message, intent: str, prompt: str, status_msg=None, ref_msg=None) -> None:
    """Conversational front-end for the audited code-change pipeline."""
    is_app_owner = await bot.is_owner(message.author)
    requester_id = str(message.author.id)
    proposal_ref = _proposal_id_from_text(prompt)
    if proposal_ref is None and ref_msg is not None:
        proposal_ref = _proposal_id_from_text(getattr(ref_msg, "content", ""))
    proposals = await asyncio.to_thread(list_code_proposals, requester_id, 30)
    review_pool = (
        await asyncio.to_thread(list_all_code_proposals, 50)
        if is_app_owner else proposals
    )
    if isinstance(proposal_ref, str):
        referenced = next((p for p in review_pool if p.get("public_id") == proposal_ref), None)
        proposal_id = referenced["id"] if referenced else None
    else:
        proposal_id = proposal_ref

    def latest_with_status(*statuses):
        return next((p for p in review_pool if p["status"] in statuses), None)

    if intent == "code_change":
        request = re.sub(r"^/code_propose\s*", "", prompt or "", flags=re.IGNORECASE).strip()
        if not request:
            await message.reply("Tell me what you want changed in my code.")
            return
        open_proposal = next(
            (p for p in proposals if p["status"] in {"draft", "patch_uploaded", "reviewable"}),
            None,
        )
        if open_proposal:
            await message.reply(
                f"You already have open proposal `{open_proposal.get('public_id') or open_proposal['id']}`. Wait for review before proposing another change."
            )
            return
        if not await _confirm_code_generation(
            message,
            author_id=message.author.id,
            request_summary=request,
            existing_status=status_msg,
        ):
            return
        quota = check_rate_limit(
            "code",
            user_id=requester_id,
            guild_id=(
                str(message.guild.id)
                if getattr(message, "guild", None) is not None
                else "DM"
            ),
        )
        if not quota.allowed:
            await message.reply(
                "⏳ Code-generation quota reached for now. "
                f"Try again in about {quota.retry_after} seconds."
            )
            return
        status_msg = None
        baseline = await asyncio.to_thread(get_baseline_sha)
        proposal_id = await asyncio.to_thread(
            create_code_proposal, requester_id, request, baseline
        )
        proposal = await asyncio.to_thread(get_code_proposal, requester_id, proposal_id)
        proposal_name = proposal.get("public_id") or str(proposal_id)
        generation_phase = {
            "label": f"Mapping proposal {proposal_name}",
        }

        async def _generate_and_validate():
            nonlocal proposal
            generation_phase["label"] = f"Drafting proposal {proposal_name}"
            patch, generation = await generate_code_patch(request, baseline)
            generation_phase["label"] = f"Validating proposal {proposal_name}"
            await asyncio.to_thread(
                set_code_proposal_patch, requester_id, proposal_id, patch
            )
            report = await asyncio.to_thread(validate_patch, baseline, patch)
            report["generation"] = generation
            proposal = await asyncio.to_thread(
                set_code_proposal_validation, requester_id, proposal_id, report
            )
            return patch, generation, report

        try:
            status_msg, generated = await live_status_with_progress(
                message,
                action_label=lambda: generation_phase["label"],
                emoji="🧩",
                coro=_generate_and_validate(),
                duration_estimate=90,
            )
            patch, generation, report = generated
        except (RuntimeError, ValueError) as exc:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(
                    review_any_code_proposal,
                    proposal_id,
                    "rejected",
                    reviewer_id="generation-failure",
                )
            if status_msg is not None:
                with contextlib.suppress(Exception):
                    await status_msg.delete()
            await message.reply(
                f"I recorded proposal `{proposal_name}`, but patch generation failed because "
                f"{public_error_detail(exc)}. Detailed diagnostics were recorded privately."
            )
            return
        if status_msg is not None:
            with contextlib.suppress(Exception):
                await status_msg.delete()
        payload = io.BytesIO(patch.encode("utf-8"))
        if not report["ok"]:
            errors = "; ".join(report.get("errors") or ["Unknown validation error"])
            await message.reply(
                f"⚠️ I recorded proposal `{proposal_name}`, but it is **not ready for review** because validation failed.\n"
                f"{_code_proposal_summary(proposal)}\n"
                f"Reason: {errors[:1200]}\n"
                f"You can submit a corrected proposal. Track public status: {CHANGE_DASHBOARD_URL}",
                file=discord.File(payload, filename=f"proposal-{proposal_name}-failed.diff"),
            )
            return
        await message.reply(
            f"🧩 **Proposal `{proposal_name}` is ready for review.**\n"
            f"{_code_proposal_summary(proposal)}\n"
            f"Files: {', '.join(report['files'])}\n"
            f"The bot owner can reply naturally with **approve this change** or **reject it**. You can ask **what is its status?**\n"
            f"Track public status: {CHANGE_DASHBOARD_URL}",
            file=discord.File(payload, filename=f"proposal-{proposal_name}.diff"),
        )
        return

    if proposal_id is None:
        if intent == "code_approve":
            selected = latest_with_status("reviewable")
        elif intent == "code_reject":
            selected = latest_with_status("draft", "patch_uploaded", "validation_failed", "reviewable")
        else:
            selected = review_pool[0] if review_pool else None
        proposal_id = selected["id"] if selected else None
    if proposal_id is None:
        await message.reply("I couldn't find one of your code proposals to act on.")
        return

    if intent in {"code_approve", "code_reject", "code_rollback"} and not is_app_owner:
        await message.reply("Anyone can propose a code change, but only my owner can approve, reject, deploy, or roll one back.")
        return

    if intent == "code_approve":
        try:
            proposal = await asyncio.to_thread(
                review_any_code_proposal,
                proposal_id,
                "approved",
                reviewer_id=requester_id,
            )
        except ValueError as exc:
            await message.reply(f"I couldn't approve that proposal because {public_error_detail(exc)}.")
            return
        proposal_name = proposal.get("public_id") or str(proposal_id)
        approval_message = await message.reply(
            f"✅ Approved proposal `{proposal_name}`. The separate host supervisor will test and deploy it; I'll DM you the result.\n"
            f"{_code_proposal_summary(proposal)}\n\n"
            "**Deployment progress**\n"
            "✅ Owner approval recorded\n"
            f"⏳ Waiting for the host supervisor\n\nTrack public status: {CHANGE_DASHBOARD_URL}"
        )
        await asyncio.to_thread(
            set_code_proposal_approval_message,
            proposal_id,
            str(approval_message.channel.id),
            str(approval_message.id),
        )
        return

    if intent == "code_reject":
        try:
            proposal = await asyncio.to_thread(
                review_any_code_proposal,
                proposal_id,
                "rejected",
                reviewer_id=requester_id,
            )
        except ValueError as exc:
            await message.reply(f"I couldn't reject that proposal because {public_error_detail(exc)}.")
            return
        await message.reply(f"🛑 Rejected proposal `{proposal.get('public_id') or proposal_id}`. No code was deployed.")
        return

    if intent == "code_rollback":
        if proposal_ref is None:
            active = None
            for candidate in review_pool:
                deployment = await asyncio.to_thread(
                    get_code_deployment, candidate["owner_id"], candidate["id"]
                )
                if deployment and deployment["status"] == "active":
                    active = candidate
                    break
            proposal_id = active["id"] if active else None
        if proposal_id is None:
            await message.reply("You don't currently have an active code release to roll back.")
            return
        try:
            request_id = await asyncio.to_thread(
                request_any_code_rollback, requester_id, proposal_id
            )
        except ValueError as exc:
            await message.reply(f"I couldn't queue that rollback because {public_error_detail(exc)}.")
            return
        rollback_proposal = next((p for p in review_pool if p["id"] == proposal_id), None)
        rollback_name = rollback_proposal.get("public_id") if rollback_proposal else str(proposal_id)
        await message.reply(
            f"↩️ Rollback queued for proposal `{rollback_name}`. I'll DM you when the previous release is healthy."
        )
        return

    proposal = (
        await asyncio.to_thread(get_any_code_proposal, proposal_id)
        if is_app_owner
        else await asyncio.to_thread(get_code_proposal, requester_id, proposal_id)
    )
    if not proposal:
        await message.reply("I couldn't find that proposal.")
        return
    deployment = await asyncio.to_thread(
        get_code_deployment, proposal["owner_id"], proposal_id
    )
    if not deployment:
        await message.reply(
            f"{_code_proposal_summary(proposal)}\nIt has no deployment record yet."
        )
        return
    await message.reply(
        f"{_code_proposal_summary(proposal)}\n"
        f"Deployment: **{deployment['status']}** — "
        f"{sanitize_diagnostic_text(deployment.get('detail') or 'in progress', max_chars=900)}\n"
        f"Track public status: {CHANGE_DASHBOARD_URL}"
    )


@bot.hybrid_command(name="code_propose", description="Owner: create a reviewed code-change request.", with_app_command=False)
@commands.is_owner()
async def code_propose(ctx, *, request: str):
    baseline = await asyncio.to_thread(get_baseline_sha)
    proposal_id = await asyncio.to_thread(
        create_code_proposal, str(ctx.author.id), request, baseline
    )
    proposal = await asyncio.to_thread(get_code_proposal, str(ctx.author.id), proposal_id)
    await ctx.reply(
        f"🧩 Created against the current committed baseline.\n{_code_proposal_summary(proposal)}\n"
        "Attach a unified `.diff` or `.patch` with `/code_patch`."
    )


@bot.hybrid_command(name="code_patch", description="Owner: attach a unified diff to a code proposal.", with_app_command=False)
@commands.is_owner()
async def code_patch(ctx, proposal_id: int, patch_file: discord.Attachment):
    if patch_file.size > MAX_PATCH_BYTES:
        await ctx.reply(f"❌ Patch exceeds the {MAX_PATCH_BYTES:,}-byte limit.")
        return
    if not patch_file.filename.lower().endswith((".diff", ".patch")):
        await ctx.reply("❌ Attach a `.diff` or `.patch` file.")
        return
    try:
        patch = (await patch_file.read()).decode("utf-8")
        proposal = await asyncio.to_thread(
            set_code_proposal_patch, str(ctx.author.id), proposal_id, patch
        )
    except UnicodeDecodeError:
        await ctx.reply("❌ Patch must be UTF-8 text.")
        return
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    await ctx.reply(
        f"📎 Patch attached ({len(patch.encode('utf-8')):,} bytes).\n"
        f"{_code_proposal_summary(proposal)}\nRun `/code_validate {proposal_id}` next."
    )


@bot.hybrid_command(name="code_generate", description="Owner: generate and validate a patch for a proposal.", with_app_command=False)
@commands.is_owner()
async def code_generate(ctx, proposal_id: int):
    proposal = await asyncio.to_thread(get_code_proposal, str(ctx.author.id), proposal_id)
    if not proposal or proposal["status"] in {"approved", "rejected"}:
        await ctx.reply("❌ Editable code proposal not found.")
        return
    if not await _confirm_code_generation(
        ctx,
        author_id=ctx.author.id,
        request_summary=proposal["request"],
    ):
        return
    generation_phase = {"label": f"Drafting proposal {proposal_id}"}

    async def _generate_and_validate():
        patch, generation = await generate_code_patch(
            proposal["request"], proposal["baseline_sha"]
        )
        generation_phase["label"] = f"Validating proposal {proposal_id}"
        await asyncio.to_thread(
            set_code_proposal_patch, str(ctx.author.id), proposal_id, patch
        )
        report = await asyncio.to_thread(
            validate_patch, proposal["baseline_sha"], patch
        )
        report["generation"] = generation
        updated = await asyncio.to_thread(
            set_code_proposal_validation, str(ctx.author.id), proposal_id, report
        )
        return patch, generation, report, updated

    status_msg = None
    try:
        status_msg, generated = await _run_context_progress(
            ctx,
            action_label=lambda: generation_phase["label"],
            emoji="🤖",
            coro=_generate_and_validate(),
            duration_estimate=90,
        )
        patch, generation, report, proposal = generated
    except (RuntimeError, ValueError) as exc:
        if status_msg is not None:
            with contextlib.suppress(Exception):
                await status_msg.delete()
        await ctx.reply(f"❌ Patch generation failed: {str(exc)[:1200]}")
        return
    if status_msg is not None:
        with contextlib.suppress(Exception):
            await status_msg.delete()
    payload = io.BytesIO(patch.encode("utf-8"))
    files = ", ".join(report["files"])
    if not report["ok"]:
        errors = "; ".join(report.get("errors") or ["Unknown validation error"])
        await ctx.reply(
            f"⚠️ Generated with `{generation['model']}`, but validation failed.\n"
            f"{_code_proposal_summary(proposal)}\nReason: {errors[:1200]}",
            file=discord.File(payload, filename=f"proposal-{proposal_id}-failed.diff"),
        )
        return
    await ctx.reply(
        f"🤖 Generated with `{generation['model']}` and statically validated.\n"
        f"{_code_proposal_summary(proposal)}\nFiles: {files}\n"
        f"Review the attached diff, then use `/code_approve {proposal_id}`.",
        file=discord.File(payload, filename=f"proposal-{proposal_id}-generated.diff"),
    )


@bot.hybrid_command(name="code_validate", description="Owner: statically validate a proposal in a temporary snapshot.", with_app_command=False)
@commands.is_owner()
async def code_validate(ctx, proposal_id: int):
    proposal = await asyncio.to_thread(get_code_proposal, str(ctx.author.id), proposal_id)
    if not proposal or not proposal.get("patch"):
        await ctx.reply("❌ Proposal not found or it has no attached patch.")
        return
    if ctx.interaction:
        await ctx.defer()
    report = await asyncio.to_thread(
        validate_patch, proposal["baseline_sha"], proposal["patch"]
    )
    proposal = await asyncio.to_thread(
        set_code_proposal_validation, str(ctx.author.id), proposal_id, report
    )
    lines = [_code_proposal_summary(proposal)]
    lines.append("Files: " + (", ".join(report["files"]) or "none"))
    if report["syntax_checked"]:
        lines.append("Python syntax: " + ", ".join(report["syntax_checked"]))
    lines.extend(f"❌ {error}" for error in report["errors"][:5])
    lines.extend(f"⚠️ {warning}" for warning in report["warnings"][:5])
    if report["ok"]:
        lines.append("✅ Static validation passed. This patch has **not** been deployed or executed.")
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(name="code_show", description="Owner: show a code proposal and its review state.", with_app_command=False)
@commands.is_owner()
async def code_show(ctx, proposal_id: int):
    proposal = await asyncio.to_thread(get_code_proposal, str(ctx.author.id), proposal_id)
    if not proposal:
        await ctx.reply("Code proposal not found.")
        return
    await ctx.reply(_code_proposal_summary(proposal))


@bot.hybrid_command(name="code_diff", description="Owner: download the patch attached to a code proposal.", with_app_command=False)
@commands.is_owner()
async def code_diff(ctx, proposal_id: int):
    proposal = await asyncio.to_thread(get_code_proposal, str(ctx.author.id), proposal_id)
    if not proposal or not proposal.get("patch"):
        await ctx.reply("Proposal not found or it has no attached patch.")
        return
    payload = io.BytesIO(proposal["patch"].encode("utf-8"))
    await ctx.reply(
        _code_proposal_summary(proposal),
        file=discord.File(payload, filename=f"proposal-{proposal_id}.diff"),
    )


@bot.hybrid_command(name="code_history", description="Owner: list recent code proposals.", with_app_command=False)
@commands.is_owner()
async def code_history(ctx, limit: int = 10):
    proposals = await asyncio.to_thread(list_code_proposals, str(ctx.author.id), limit)
    if not proposals:
        await ctx.reply("No code proposals have been recorded.")
        return
    lines = ["🧾 **Code proposal history**"]
    for proposal in proposals:
        preview = proposal["request"].replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        lines.append(
            f"`{proposal.get('public_id') or proposal['id']}` **{proposal['status']}** · "
            f"`{proposal['baseline_sha'][:8]}` — {preview}"
        )
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(name="code_approve", description="Owner: approve a successfully validated code proposal.", with_app_command=False)
@commands.is_owner()
async def code_approve(ctx, proposal_id: int):
    try:
        proposal = await asyncio.to_thread(
            review_code_proposal,
            str(ctx.author.id),
            proposal_id,
            "approved",
            reviewer_id=str(ctx.author.id),
        )
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    approval_message = await ctx.reply(
        f"✅ Approved proposal `{proposal.get('public_id') or proposal_id}`.\n"
        f"{_code_proposal_summary(proposal)}\n\n"
        "**Deployment progress**\n"
        "✅ Owner approval recorded\n"
        f"⏳ Waiting for the host supervisor\n\nTrack public status: {CHANGE_DASHBOARD_URL}"
    )
    await asyncio.to_thread(
        set_code_proposal_approval_message,
        proposal_id,
        str(approval_message.channel.id),
        str(approval_message.id),
    )


@bot.hybrid_command(name="code_reject", description="Owner: reject a code proposal without deploying it.", with_app_command=False)
@commands.is_owner()
async def code_reject(ctx, proposal_id: int):
    try:
        proposal = await asyncio.to_thread(
            review_code_proposal,
            str(ctx.author.id),
            proposal_id,
            "rejected",
            reviewer_id=str(ctx.author.id),
        )
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    await ctx.reply(f"🛑 Rejected.\n{_code_proposal_summary(proposal)}")


@bot.hybrid_command(name="code_deployment", description="Owner: show deployment and health-check results.", with_app_command=False)
@commands.is_owner()
async def code_deployment(ctx, proposal_id: int):
    deployment = await asyncio.to_thread(
        get_code_deployment, str(ctx.author.id), proposal_id
    )
    if not deployment:
        await ctx.reply("No deployment record exists for that proposal yet.")
        return
    detail = deployment.get("detail") or "No detail recorded."
    await ctx.reply(
        f"🚦 **Deployment `#{proposal_id}`** · `{deployment['status']}`\n"
        f"Release: `{deployment['release_path']}`\n"
        f"Patch: `{(deployment.get('patch_sha256') or '')[:12]}`\n"
        f"Detail: {detail[:900]}"
    )


@bot.hybrid_command(name="code_rollback", description="Owner: request rollback of the active code release.", with_app_command=False)
@commands.is_owner()
async def code_rollback(ctx, proposal_id: int):
    try:
        request_id = await asyncio.to_thread(
            request_code_rollback, str(ctx.author.id), proposal_id
        )
    except ValueError as exc:
        await ctx.reply(f"❌ {exc}")
        return
    await ctx.reply(
        f"↩️ Rollback request `#{request_id}` queued. The host supervisor will process it within one minute."
    )

@bot.hybrid_command(name="memory_fetch_more", description="Index recent channel messages into long-term memory (mods only).")
@commands.has_permissions(manage_messages=True)
async def memory_fetch_more(ctx, chunk: int = 200):
    """
    Fetch RECENT messages for this channel only (not old history).
    Uses a per-channel 'last_seen_id' high-water mark; first run grabs latest <chunk>.
    """
    try:
        if ctx.interaction:
            await ctx.defer()
        count = await backfill_recent_channel_history_to_es(
            ctx.guild.id if ctx.guild else None, ctx.channel.id, chunk=chunk
        )
        await ctx.reply(f"Indexed ~{count} recent message(s) for this channel.")
    except Exception as e:
        logger.exception("Recent memory indexing failed")
        await ctx.reply(f"❌ Memory indexing failed because {public_error_detail(e)}.")

@bot.hybrid_command(name="memories", description="See what the bot remembers about you.")
async def memories(ctx):
    """Show remembered facts, profile freshness, and indexed-history stats."""
    if ctx.interaction:
        await ctx.defer(ephemeral=True)
    uid = str(ctx.author.id)
    facts = await asyncio.to_thread(list_user_facts, uid, 25)
    profile = await asyncio.to_thread(get_user_profile, uid)

    lines = [f"🧠 **What I remember about {ctx.author.display_name}:**"]
    if facts:
        for f in reversed(facts):
            lines.append(f"`#{f['id']}` {f['fact']}")
    else:
        lines.append("_No saved facts yet — tell me things worth remembering._")

    from services.time_context import time_ago_str
    if profile and profile.get("updated_at"):
        lines.append(f"\n_Profile last distilled {time_ago_str(profile['updated_at'])} ago — see `/profile`._")

    # Long-term history stats straight from Elasticsearch.
    try:
        from services.user_profile import user_memory_stats
        stats = await asyncio.to_thread(user_memory_stats, uid)
        if stats and stats.get("indexed_messages"):
            since = f" since {stats['first_seen'][:10]}" if stats.get("first_seen") else ""
            lines.append(f"_{stats['indexed_messages']:,} of your messages indexed in long-term memory{since}._")
    except Exception:
        logger.warning("user_memory_stats failed", exc_info=True)

    lines.append("_Use `/forget <text>` to remove anything._")
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(name="profile", description="Show the bot's distilled profile of you.")
async def profile(ctx):
    if ctx.interaction:
        await ctx.defer(ephemeral=True)
    prof = await asyncio.to_thread(get_user_profile, str(ctx.author.id))
    if not prof or not (prof.get("profile") or "").strip():
        await ctx.reply("_No distilled profile yet — chat with me a bit more and one will form._")
        return
    from services.time_context import time_ago_str
    when = f" _(refreshed {time_ago_str(prof['updated_at'])} ago)_" if prof.get("updated_at") else ""
    await ctx.reply(f"📋 **My read on you**{when}:\n{prof['profile']}"[:1990])


@bot.hybrid_command(name="forget", description="Delete remembered facts matching some text.")
async def forget(ctx, *, text: str = ""):
    """Delete remembered facts matching the given text."""
    if not text.strip():
        await ctx.reply("Usage: `/forget <text to match>`")
        return
    deleted = await asyncio.to_thread(delete_user_facts_matching, str(ctx.author.id), text.strip())
    if deleted:
        await ctx.reply(f"🗑️ Forgot {deleted} fact{'s' if deleted != 1 else ''} matching “{text.strip()}”.")
    else:
        await ctx.reply(f"Nothing stored matches “{text.strip()}”.")


@bot.hybrid_command(name="usage", description="Show API usage and costs (yours + totals).")
async def usage(ctx):
    """Show API usage: yours today, plus totals (mods see per-user breakdown)."""
    if ctx.interaction:
        await ctx.defer()
    uid = str(ctx.author.id)
    mine = await asyncio.to_thread(usage_costs.today_for_user, uid)
    day = await asyncio.to_thread(usage_costs.today)
    month = await asyncio.to_thread(usage_costs.month_to_date)
    year = await asyncio.to_thread(usage_costs.year_to_date)
    total = await asyncio.to_thread(usage_costs.all_time)
    my_break = await asyncio.to_thread(usage_costs.today_breakdown, uid, 8)

    def _fmt_row(b):
        tokens = f", {b['total_tokens']:,} tok" if b["total_tokens"] else ""
        return f"  `{b['model']}` {b['label']}: {b['calls']}×{tokens}, ${b['cost']:.4f}"

    lines = [
        "📊 **API usage**",
        f"**You today:** {mine['calls']} calls, {mine['total_tokens']:,} tokens, ${mine['cost']:.4f}",
    ]
    lines.extend(_fmt_row(b) for b in my_break)
    lines.append(f"**Everyone today:** {day['calls']} calls, {day['total_tokens']:,} tokens, ${day['cost']:.4f}")

    perms = getattr(ctx.author, "guild_permissions", None)
    is_mod = bool(perms and perms.manage_messages)
    if is_mod:
        all_break = await asyncio.to_thread(usage_costs.today_breakdown, None, 8)
        lines.extend(_fmt_row(b) for b in all_break)

    lines.append(f"**Month to date:** {month['calls']} calls, {month['total_tokens']:,} tokens, ${month['cost']:.2f}")
    lines.append(f"**Year to date:** {year['calls']} calls, {year['total_tokens']:,} tokens, ${year['cost']:.2f}")
    lines.append(f"**All time:** {total['calls']} calls, {total['total_tokens']:,} tokens, ${total['cost']:.2f}")

    if is_mod:
        top = await asyncio.to_thread(usage_costs.top_users_today, 5)
        if top:
            lines.append("\n**Top spenders today:**")
            for t in top:
                lines.append(f"<@{t['user_id']}>: {t['calls']} calls, ${t['cost']:.4f}")

        top_month = await asyncio.to_thread(usage_costs.top_users_month, 5)
        if top_month:
            lines.append("\n**Top spenders this month:**")
            for t in top_month:
                lines.append(f"<@{t['user_id']}>: {t['calls']} calls, ${t['cost']:.4f}")
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(
    name="reflection_off",
    description="Delete your pending and derived reflection state.",
)
async def reflection_off(ctx):
    worker = _get_reflection_worker()
    await asyncio.to_thread(worker.set_user_enabled, str(ctx.author.id), False)
    await ctx.reply(
        "🧠 Your pending sessions and derived reflection observations were removed. "
        "Explicitly invoking Multivac again will consent to a new five-minute-idle session."
    )


@bot.hybrid_command(
    name="reflection_status",
    description="Owner: show reflection scope, queues, models, and today's budget.",
)
@commands.is_owner()
async def reflection_status(ctx):
    status = await asyncio.to_thread(_get_reflection_worker().status)
    budget = status["budget"]
    models = status["models"]
    await ctx.reply(
        "🧠 **Reflection status**\n"
        f"Enabled: **{status['enabled']}** · invocation-scoped consent: **True**\n"
        f"Window: {status['lookback_minutes']}m lookback · closes after {status['idle_minutes']}m idle\n"
        f"Queue: {status['pending_sessions']} sessions · {status['pulse_queue']} message pulses · "
        f"{status['new_insights']} new observations · "
        f"{status['active_ideas']} active ideas\n"
        f"Models: pulse/extract `{models['pulse']}` · plan `{models['plan']}` · cleanup `{models['cleanup']}`\n"
        f"Today: ${budget['spent']:.4f} spent + ${budget['reserved']:.4f} reserved · "
        f"${budget['remaining']:.4f} of ${budget['cap']:.2f} remaining"
    )


@bot.hybrid_command(
    name="reflection_activity",
    description="Owner: show sanitized reflection observations, runs, and schedule.",
)
@commands.is_owner()
async def reflection_activity(ctx, limit: int = 5):
    activity = await asyncio.to_thread(
        _get_reflection_worker().activity, max(1, min(8, int(limit)))
    )

    def _relative_timestamp(value: str) -> str:
        try:
            epoch = int(datetime.fromisoformat(value).timestamp())
            return f"<t:{epoch}:R>"
        except (TypeError, ValueError):
            return "unknown"

    budget = activity["budget"]
    lines = [
        "🧠 **Reflection activity**",
        "_Derived telemetry only; no transcripts, identities, or private model reasoning._",
        (
            f"Live sessions: {activity['pending_sessions']} · "
            f"queued message pulses: {activity['pulse_queue']} · "
            f"closes after {activity['idle_minutes']}m idle"
        ),
        (
            f"Planning signals: {activity['signal_strength']}/{activity['signal_threshold']} · "
            f"next time-eligible {_relative_timestamp(activity['next_plan_at'])}"
        ),
        (
            f"Cleanup ideas: {activity['active_ideas']}/{activity['cleanup_threshold']} · "
            f"next time-eligible {_relative_timestamp(activity['next_cleanup_at'])}"
        ),
        f"Automatic budget: ${budget['spent']:.4f} spent · ${budget['remaining']:.4f} remaining",
    ]
    observations = activity["observations"]
    if observations:
        lines.append("\n**Recent structured observations**")
        for item in observations:
            lines.append(
                f"`#{item['id']}` `{item['kind']}` · {item['recent_occurrences']} recent/"
                f"{item['occurrences']} total · {item['confidence']:.0%} confidence · "
                f"{item['actor_count']} actor(s) · `{item['status']}`\n{item['summary'][:240]}"
            )
    else:
        lines.append("\n_No structured observations yet._")
    runs = activity["runs"]
    if runs:
        lines.append("\n**Recent runs**")
        for run in runs:
            detail = f" · {run['detail'][:120]}" if run["detail"] else ""
            lines.append(
                f"`{run['stage']}` `{run['status']}` · `{run['model']}` · "
                f"{_relative_timestamp(run['finished_at'])}{detail}"
            )
    await ctx.reply("\n".join(lines)[:1990])


@bot.hybrid_command(
    name="reflection_ideas",
    description="Owner: list evidence-backed improvements Multivac has considered.",
)
@commands.is_owner()
async def reflection_ideas(ctx, limit: int = 5):
    ideas = await asyncio.to_thread(
        _get_reflection_worker().store.list_ideas, max(1, min(10, int(limit)))
    )
    if not ideas:
        await ctx.reply("_No evidence-backed reflection ideas are ready yet._")
        return
    lines = ["🧠 **Evidence-backed improvement ideas**"]
    for idea in ideas:
        lines.append(
            f"`#{idea['id']}` **{idea['title']}** · `{idea['hotload_kind']}` · "
            f"{idea['evidence_count']} signal(s)\n{idea['problem'][:300]}"
        )
    lines.append("Use `/reflection_propose <idea_id>` to generate a normal reviewable proposal.")
    await ctx.reply("\n\n".join(lines)[:1990])


@bot.hybrid_command(
    name="reflection_propose",
    description="Owner: turn one reflection idea into a low-cost, reviewable code proposal.",
)
@commands.is_owner()
async def reflection_propose(ctx, idea_id: int):
    worker = _get_reflection_worker()
    idea = await asyncio.to_thread(worker.store.get_idea, int(idea_id))
    if not idea or idea["status"] != "active":
        await ctx.reply("That active reflection idea does not exist or was already proposed.")
        return
    owner_id = str(ctx.author.id)
    open_proposals = await asyncio.to_thread(list_code_proposals, owner_id, 30)
    existing = next(
        (
            item for item in open_proposals
            if item["status"] in {"draft", "patch_uploaded", "reviewable"}
        ),
        None,
    )
    if existing:
        await ctx.reply(
            f"Finish proposal `{existing.get('public_id') or existing['id']}` before generating another."
        )
        return
    if not await _confirm_code_generation(
        ctx,
        author_id=ctx.author.id,
        request_summary=(
            f"Reflection idea #{idea['id']}: {idea['title']}. {idea['problem']}"
        ),
    ):
        return
    request = (
        f"Implement owner-selected reflection idea #{idea['id']}: {idea['title']}. "
        f"Suggested deployment kind: {idea['hotload_kind']}. Problem: {idea['problem']} "
        f"Plan: {idea['proposal']} Expected impact: {idea['expected_impact']} "
        f"Risk to preserve against: {idea['risk']} Keep the patch minimal and reversible."
    )
    baseline = await asyncio.to_thread(get_baseline_sha)
    proposal_id = await asyncio.to_thread(
        create_code_proposal, owner_id, request, baseline
    )
    proposal = await asyncio.to_thread(get_code_proposal, owner_id, proposal_id)
    proposal_name = proposal.get("public_id") or str(proposal_id)
    usage_costs.set_request_context(user_id=owner_id, intent="reflection_code")
    generation_phase = {"label": f"Shaping reflection idea {idea_id}"}

    async def _generate_and_validate():
        nonlocal proposal
        patch, generation = await generate_planned_idea_patch(
            request,
            baseline,
            context_paths=idea.get("code_paths") or [],
        )
        generation_phase["label"] = f"Validating proposal {proposal_name}"
        await asyncio.to_thread(
            set_code_proposal_patch, owner_id, proposal_id, patch
        )
        report = await asyncio.to_thread(validate_patch, baseline, patch)
        report["generation"] = generation
        proposal = await asyncio.to_thread(
            set_code_proposal_validation, owner_id, proposal_id, report
        )
        return patch, generation, report

    status_msg = None
    try:
        status_msg, generated = await _run_context_progress(
            ctx,
            action_label=lambda: generation_phase["label"],
            emoji="🧩",
            coro=_generate_and_validate(),
            duration_estimate=90,
        )
        patch, generation, report = generated
    except (RuntimeError, ValueError) as exc:
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                review_any_code_proposal,
                proposal_id,
                "rejected",
                reviewer_id="reflection-generation-failure",
            )
        if status_msg is not None:
            with contextlib.suppress(Exception):
                await status_msg.delete()
        await ctx.reply(
            f"Idea `#{idea_id}` was recorded as proposal `{proposal_name}`, but the "
            f"{IDEA_CODE_MODEL} coding pass failed closed: {str(exc)[:1000]}"
        )
        return
    if status_msg is not None:
        with contextlib.suppress(Exception):
            await status_msg.delete()
    payload = io.BytesIO(patch.encode("utf-8"))
    if not report["ok"]:
        await ctx.reply(
            f"⚠️ Proposal `{proposal_name}` failed static validation and is not reviewable.\n"
            f"{_code_proposal_summary(proposal)}",
            file=discord.File(payload, filename=f"proposal-{proposal_name}-failed.diff"),
        )
        return
    await asyncio.to_thread(worker.store.mark_idea_proposed, int(idea_id), proposal_id)
    await ctx.reply(
        f"🧩 Reflection idea `#{idea_id}` is now reviewable proposal `{proposal_name}`.\n"
        f"{_code_proposal_summary(proposal)}\n"
        f"Files: {', '.join(report['files'])}\n"
        "Approval and deployment use the unchanged owner-reviewed pipeline.",
        file=discord.File(payload, filename=f"proposal-{proposal_name}.diff"),
    )


# --------------------------
# Main message handler
# --------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

async def _builtin_on_message(message: discord.Message):
    if message.author.bot:
        return
    
    if _already_processed(message.id):
        return

    observed_sessions: set[int] = set()
    try:
        observed_sessions = await _get_reflection_worker().observe_message(
            guild_id=str(message.guild.id) if message.guild else "DM",
            channel_id=str(message.channel.id),
            author_id=str(message.author.id),
            message_id=str(message.id),
            content=message.content or "",
        )
    except Exception:
        logger.warning("Unable to extend active reflection session", exc_info=True)

    # Pass-through slash/! commands
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    # Trigger: direct mention or reply to bot
    raw_prompt = strip_mention_and_trigger(message.content, bot.user.id if bot.user else None)
    prompt = raw_prompt
    user_id = message.author.id

    # Only an EXPLICIT @bot mention in the message text counts as a trigger.
    # mentioned_in() is also true for role mentions (any @role the bot holds),
    # which made the bot jump into conversations nobody addressed to it.
    is_direct_mention = bool(bot.user) and (
        f"<@{bot.user.id}>" in (message.content or "")
        or f"<@!{bot.user.id}>" in (message.content or "")
    )
    ref_msg, is_reply_to_bot = await resolve_reference_message(message, bot.user)

    if not (is_direct_mention or is_reply_to_bot) or message.mention_everyone:
        return
    
    # Skip if user is currently responding to an image selection prompt
    if message.author.id in _pending_image_selection:
        return

    # Silence is a terminal decision before quota notices, progress messages,
    # intent classification, or any chat/provider path can emit a response.
    if await should_suppress_response(raw_prompt):
        return

    try:
        is_app_owner = await bot.is_owner(message.author)
    except Exception:
        logger.warning("Unable to resolve application owner status", exc_info=True)
        is_app_owner = False

    request_quota = check_rate_limit(
        "request",
        user_id=str(user_id),
        guild_id=str(message.guild.id) if message.guild else "DM",
    )
    if not request_quota.allowed:
        await message.reply(
            "⏳ I'm receiving too many requests right now. "
            f"Try again in about {request_quota.retry_after} seconds."
        )
        return

    try:
        reflection = _get_reflection_worker()
        await reflection.observe_invocation(
            guild_id=str(message.guild.id) if message.guild else "DM",
            channel_id=str(message.channel.id),
            user_id=str(message.author.id),
            message_id=str(message.id),
            content=message.content or "",
            already_observed=observed_sessions,
        )
    except Exception:
        logger.warning("Unable to record bounded reflection session", exc_info=True)

    preflight_status = None
    try:
        preflight_status = await message.reply(
            _preflight_status(0, "Opening the response path…")
        )
    except Exception:
        try:
            preflight_status = await message.channel.send(
                _preflight_status(0, "Opening the response path…")
            )
        except Exception:
            preflight_status = None

    if preflight_status is not None:
        _preflight_status_by_message_id[message.id] = preflight_status

    # Pre-log the user's message (live indexing)
    try:
        await _index_message_async(
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
        logger.warning("Live indexing of user message %s failed", message.id, exc_info=True)
    finally:
        if preflight_status is not None:
            with contextlib.suppress(Exception):
                await preflight_status.edit(
                    content=_preflight_status(1, "Indexing fresh context…")
                )

    persona_toggle = parse_persona_toggle(raw_prompt)
    if persona_toggle is not None:
        scope_key = message_persona_scope(message, user_id)
        try:
            await asyncio.to_thread(
                set_conversation_persona_enabled,
                scope_key,
                persona_toggle,
            )
        except Exception:
            logger.exception("Unable to persist conversation persona setting")
            failure = "❌ I couldn't save the persona setting for this conversation."
            if preflight_status is not None:
                await preflight_status.edit(content=failure)
            else:
                await message.reply(failure)
            _preflight_status_by_message_id.pop(message.id, None)
            return

        acknowledgement = (
            f"**{PERSONA_NAME}** resumed for this conversation."
            if persona_toggle
            else "Persona disabled for this conversation. I'll answer neutrally until you ask me to resume it."
        )
        _preflight_status_by_message_id.pop(message.id, None)
        if preflight_status is not None:
            await send_or_edit_with_truncation(
                acknowledgement,
                target_msg=preflight_status,
                original_message=message,
                model="local",
            )
        else:
            await send_or_edit_with_truncation(
                acknowledgement,
                channel=message.channel,
                reply_to=message,
                model="local",
            )
        await bot.process_commands(message)
        return

    source_image_urls = []
    unusable_images = []
    image_urls = await collect_image_inputs(
        message,
        ref_msg,
        image_url_to_base64,
        source_image_urls,
        unusable_images,
    )
    video_inputs = []
    unusable_videos = []
    gemini_parts = await collect_gemini_parts(
        message,
        ref_msg,
        image_urls,
        video_inputs,
        unusable_videos,
    )

    # Say so instead of routing an image request onward with no image — that
    # fell through to plain chat and asked for a picture the user had attached.
    if unusable_images and not (image_urls or gemini_parts):
        subject = f"`{unusable_images[0]}`" if len(unusable_images) == 1 else f"{len(unusable_images)} attached images"
        notice = (
            f"🖼️ I couldn't read {subject} — too large for me to process even after resizing. "
            "Try reposting it as a JPEG or a smaller screenshot."
        )
        _preflight_status_by_message_id.pop(message.id, None)
        if preflight_status is not None:
            with contextlib.suppress(Exception):
                await preflight_status.edit(content=notice)
        else:
            await message.reply(notice)
        return

    # Say so when a clip arrived that nothing downstream can watch, instead of
    # answering as if the message had come with no attachment at all.
    if unusable_videos and not (image_urls or video_inputs):
        subject = f"`{unusable_videos[0]}`" if len(unusable_videos) == 1 else f"{len(unusable_videos)} attached videos"
        notice = (
            f"🎞️ I couldn't watch {subject} — clips have to be under 10 MB and in a common "
            "format (mp4, mov, webm) for me to see them. Try a shorter or smaller clip."
        )
        _preflight_status_by_message_id.pop(message.id, None)
        if preflight_status is not None:
            with contextlib.suppress(Exception):
                await preflight_status.edit(content=notice)
        else:
            await message.reply(notice)
        return

    # Images only. Counting any attachment here told the classifier a picture
    # was present when a video or text file was attached, and every image
    # handler then received nothing.
    has_attachments = bool(image_urls) or has_visual_inputs(message, ref_msg)
    channel_context = await fetch_recent_channel_context(message, bot.user)
    if preflight_status is not None:
        with contextlib.suppress(Exception):
            await preflight_status.edit(
                content=_preflight_status(2, "Reading intent and available inputs…")
            )
    
    # Attribute all API spend during this message to the requesting user.
    usage_costs.reset_request_totals()
    usage_costs.set_request_context(
        user_id=str(user_id),
        guild_id=str(message.guild.id) if message.guild else "DM",
        channel_id=str(message.channel.id),
    )

    # Intent: explicit keyword routing first, LLM classifier otherwise.
    # The classifier sees the user's previous request so follow-ups like
    # "another one" route to the same intent.
    seen = None
    try:
        seen = await asyncio.to_thread(get_user_seen, str(user_id))
    except Exception:
        logger.warning("get_user_seen failed", exc_info=True)

    intent = resolve_keyword_intent(raw_prompt, prompt, has_attachments)
    if intent is None:
        # Give the classifier the last few real channel turns, including other
        # speakers, so a follow-up can resolve the shared topic without
        # collapsing the channel into the requester's private history.
        recent_turns = None
        if channel_context:
            recent_turns = [
                f"{m.get('role', 'user')}: {(m.get('content') or '')[:300]}"
                for m in channel_context[-4:]
                if (m.get("content") or "").strip()
            ] or None
        try:
            from services.memory_utils import build_message_window

            if not recent_turns:
                window = await asyncio.to_thread(
                    build_message_window,
                    guild_id=message.guild.id if message.guild else "DM",
                    channel_id=message.channel.id,
                    user_id=user_id,
                    limit_msgs=4,
                )
                recent_turns = [
                    f"{m.get('role', 'user')}: {(m.get('content') or '')[:200]}"
                    for m in (window or [])
                    if (m.get("content") or "").strip()
                ] or None
        except Exception:
            logger.warning("classifier context window failed", exc_info=True)
        if not recent_turns and seen and seen.get("last_prompt"):
            recent_turns = [f"user: {seen['last_prompt']}"]

        intent = await classify_intent(
            prompt,
            has_images=has_attachments,
            recent_turns=recent_turns,
            prev_intent=seen.get("last_intent") if seen else None,
        )
        intent = validate_classified_intent(
            intent,
            prompt,
            has_attachments=has_attachments,
        )

    usage_costs.set_request_context(
        user_id=str(user_id),
        guild_id=str(message.guild.id) if message.guild else "DM",
        channel_id=str(message.channel.id),
        intent=intent,
    )

    if intent in {"generate_image", "edit_image", "generate_video"}:
        media_quota = check_rate_limit(
            "image",
            user_id=str(user_id),
            guild_id=str(message.guild.id) if message.guild else "DM",
        )
        if not media_quota.allowed:
            notice = (
                "⏳ Media-generation quota reached for now. "
                f"Try again in about {media_quota.retry_after} seconds."
            )
            _preflight_status_by_message_id.pop(message.id, None)
            if preflight_status is not None:
                with contextlib.suppress(Exception):
                    await preflight_status.edit(content=notice)
            else:
                await message.reply(notice)
            return

    logger.info(
        f"Intent identified as: {intent} (has_attachments={has_attachments}, "
        f"image_inputs={len(image_urls)}, video_inputs={len(video_inputs)})"
    )
    if preflight_status is not None:
        with contextlib.suppress(Exception):
            await preflight_status.edit(
                content=_preflight_status(3, "Routing to the right capability…")
            )

    # Any URL (for summarize)
    general_url_match = re.search(r"https?://[^\s]+", message.content)

    try:
        if intent in {
            "code_change", "code_approve", "code_reject", "code_status", "code_rollback"
        }:
            await _natural_code_proposal(
                message, intent, prompt, preflight_status, ref_msg=ref_msg
            )
        else:
            await dispatch_intent(
                DispatchContext(
                intent=intent,
                message=message,
                prompt=prompt,
                raw_prompt=raw_prompt,
                user_id=user_id,
                bot_user=bot.user,
                ref_msg=ref_msg,
                is_reply_to_bot=is_reply_to_bot,
                is_owner=is_app_owner,
                image_urls=image_urls,
                source_image_urls=source_image_urls,
                video_inputs=video_inputs,
                gemini_parts=gemini_parts,
                channel_context=channel_context,
                general_url_match=general_url_match,
                stream_ok=STREAM_OK,
                response_gate_checked=True,
                get_location_details=get_location_details,
                get_weather_data=get_weather_data,
                live_status_with_progress=live_status_with_progress,
                send_or_edit_with_truncation=send_or_edit_with_truncation,
                prompt_for_image_selection=prompt_for_image_selection,
                moderation_view_factory=ModerationFallbackView,
                )
            )

    except Exception as e:
        logger.exception("Critical error in on_message dispatch")
        with contextlib.suppress(Exception):
            await message.reply(
                "❌ I couldn't complete that request because "
                f"{public_error_detail(e)}. The detailed error was recorded privately."
            )
        _preflight_status_by_message_id.pop(message.id, None)

    leftover_status = _preflight_status_by_message_id.pop(message.id, None)
    if leftover_status is not None:
        with contextlib.suppress(Exception):
            await leftover_status.delete()

    # Track last interaction (time-passage awareness + intent continuity) and
    # opportunistically refresh the distilled user profile off the reply path.
    try:
        await asyncio.to_thread(set_user_seen, str(user_id), intent=intent, prompt=prompt)
    except Exception:
        logger.warning("set_user_seen failed", exc_info=True)
    asyncio.create_task(
        maybe_refresh_profile(
            guild_id=str(message.guild.id) if message.guild else "DM",
            channel_id=str(message.channel.id),
            user_id=str(user_id),
        )
    )

    # let other cogs/commands run too
    await bot.process_commands(message)


@bot.event
async def on_message(message: discord.Message):
    return await dispatch_event("message", _builtin_on_message, message)

# ---- Entrypoint (used by main.py) ----
def run_bot():
    bot.run(DISCORD_TOKEN)
