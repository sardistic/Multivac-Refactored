"""Persistent guild controls and request-local model policy."""

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar

from services.database_utils import store

WORKLOADS = ("chat", "research", "deep", "vision", "image", "edit", "video")
TIERS = ("auto", "economy", "balanced", "quality")
_GUILD = ContextVar("multivac_policy_guild", default=None)


@contextmanager
def server_scope(guild_id):
    token = _GUILD.set(guild_id)
    try:
        yield
    finally:
        _GUILD.reset(token)


def current_guild():
    return _GUILD.get()


def get_policy(guild_id=None):
    guild_id = current_guild() if guild_id is None else guild_id
    if guild_id in (None, "DM"):
        return {"defaults": {}, "ceilings": {}, "throttle_percent": 100}
    with store.logs_conn() as conn:
        row = conn.execute("SELECT settings FROM guild_app_settings WHERE guild_id=?", (str(guild_id),)).fetchone()
    return json.loads(row[0]) if row else {"defaults": {}, "ceilings": {}, "throttle_percent": 100}


def update_policy(guild_id, *, workload="all", tier=None, max_tier=None, throttle_percent=None):
    if guild_id in (None, "DM"):
        raise ValueError("Server settings require a server.")
    if workload not in ("all", *WORKLOADS) or any(v is not None and v not in TIERS for v in (tier, max_tier)):
        raise ValueError("Choose a supported workload and tier.")
    if throttle_percent is not None and (type(throttle_percent) is not int or not 1 <= throttle_percent <= 100):
        raise ValueError("Server limits must be 1–100% of the bot's normal limits.")
    keys = WORKLOADS if workload == "all" else (workload,)
    with store.logs_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT settings FROM guild_app_settings WHERE guild_id=?", (str(guild_id),)).fetchone()
        settings = json.loads(row[0]) if row else {"defaults": {}, "ceilings": {}, "throttle_percent": 100}
        for field, value in (("defaults", tier), ("ceilings", max_tier)):
            if value is not None:
                for key in keys:
                    if value == "auto":
                        settings[field].pop(key, None)
                    else:
                        settings[field][key] = value
        if throttle_percent is not None:
            settings["throttle_percent"] = throttle_percent
        conn.execute("INSERT OR REPLACE INTO guild_app_settings VALUES(?,?)", (str(guild_id), json.dumps(settings)))
    return settings


def effective_tier(personal, workload, guild_id=None):
    settings = get_policy(guild_id)
    selected = settings["defaults"].get(workload, "auto") if personal == "auto" else personal
    cap = settings["ceilings"].get(workload)
    rank = {"economy": 0, "balanced": 1, "quality": 2, "auto": 2}
    return cap if cap and rank[selected] > rank[cap] else selected


def workload_for_intent(intent):
    if intent in {"generate_image", "character_render", "get_weather"}:
        return "image"
    if intent == "edit_image":
        return "edit"
    if intent in {"describe_image", "vision"}:
        return "vision"
    if intent == "generate_video":
        return "video"
    if intent in {"chat_research", "chat_reverse_image", "summarize_url"}:
        return "research"
    return "deep" if intent == "chat_deep" else "chat"


def capped(workload, guild_id=None):
    return get_policy(guild_id)["ceilings"].get(workload) in {"economy", "balanced"}


def video_config_allowed(config, guild_id=None):
    from providers.veo_utils import DEFAULT_VEO_FAST_MODEL, VEO_MODEL_CONFIGS

    model = config.get("model")
    seconds = config.get("seconds")
    model_config = VEO_MODEL_CONFIGS.get(model, {})
    if config.get("provider") != "veo" or seconds not in model_config.get("durations", ()):
        return False
    cap = get_policy(guild_id)["ceilings"].get("video")
    if cap not in {"economy", "balanced"}:
        return True
    return model == DEFAULT_VEO_FAST_MODEL and seconds <= (4 if cap == "economy" else 8)


def image_config_allowed(config, guild_id=None):
    from services.image_options import IMAGE_CONFIGS
    if config not in IMAGE_CONFIGS:
        return False
    cap = get_policy(guild_id)["ceilings"].get("image")
    rank = {"economy": 0, "balanced": 1, "quality": 2}
    return cap not in rank or rank[config.tier] <= rank[cap]


def channel_state(guild_id, channel_id):
    with store.logs_conn() as conn:
        row = conn.execute("SELECT state FROM bot_channel_activity WHERE guild_id=? AND channel_id=?", (str(guild_id), str(channel_id))).fetchone()
    return row[0] if row else 0


def note_activity(guild_id, channel_id):
    if guild_id is None:
        return
    with store.logs_conn() as conn:
        conn.execute("""INSERT INTO bot_channel_activity VALUES(?,?,1,?)
            ON CONFLICT(guild_id,channel_id) DO UPDATE SET state=1,last_active=excluded.last_active
            WHERE bot_channel_activity.state!=-1""", (str(guild_id), str(channel_id), time.time()))


def set_channel_enabled(guild_id, channel_id, enabled):
    with store.logs_conn() as conn:
        conn.execute("""INSERT INTO bot_channel_activity VALUES(?,?,?,NULL)
            ON CONFLICT(guild_id,channel_id) DO UPDATE SET
            state=CASE WHEN excluded.state=-1 THEN -1 WHEN bot_channel_activity.last_active IS NOT NULL THEN 1 ELSE 0 END""",
            (str(guild_id), str(channel_id), 0 if enabled else -1))


def interaction_guild(interaction):
    from services.premium import PremiumError
    guild = getattr(interaction, "guild", None)
    if guild is None and getattr(interaction, "guild_id", None) is not None:
        raise PremiumError("MultiVAC must be installed in this server to use media actions here.")
    return guild


async def require_active_channel(bot, guild, channel):
    """Activity is required for media app actions; DMs retain purchase recovery."""
    from services.premium import PremiumError
    if guild is None:
        return
    if channel is None or not getattr(guild, "me", None):
        raise PremiumError("MultiVAC must be installed in this server to use media actions here.")
    permissions = channel.permissions_for(guild.me)
    if not permissions.view_channel or not (permissions.send_messages_in_threads if getattr(channel, "parent_id", None) else permissions.send_messages):
        raise PremiumError("MultiVAC needs permission to view and send messages in this channel to use media actions here.")
    state = channel_state(guild.id, channel.id)
    if state == -1:
        raise PremiumError("Media app actions are disabled in this channel by a server manager. Use a DM for purchase recovery.")
    if state == 1:
        return
    # Import bounded historical evidence, without retaining any message content.
    try:
        lookups = 0
        async for previous in channel.history(limit=100):
            if previous.author.id != bot.user.id or getattr(previous, "webhook_id", None) or not previous.reference:
                continue
            if "Media app actions are available only" in previous.content or "disabled in this channel" in previous.content:
                continue
            reference = previous.reference
            original = reference.resolved
            if original is None:
                if lookups >= 8:
                    break
                lookups += 1
                try:
                    original = await channel.fetch_message(reference.message_id)
                except Exception:
                    continue
            if not getattr(original, "author", None):
                continue
            text = getattr(original, "content", "")
            import re
            text = re.sub(r"<@!?\d+>", "", text).strip()
            from services.server_effects import unattractive_action
            if unattractive_action(text) is not None:
                continue
            if text.startswith(("$", "/character", "/generate_image")) or text.lower().startswith(("model defaults", "server ")):
                continue
            if original.author.bot:
                continue
            note_activity(guild.id, channel.id)
            if channel_state(guild.id, channel.id) == 1:
                return
            break
    except Exception:
        pass
    raise PremiumError("Use a channel where MultiVAC has answered a normal request first. Media app actions are available only in active channels; DMs remain available.")
