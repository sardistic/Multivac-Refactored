from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from bot.chat_handler import GEMINI_FALLBACK_CHAT_MODEL, handle_chat_intent
from bot.research_policy import is_reverse_image_request, requires_fresh_web
from bot.response_policy import should_suppress_response
from bot.image_handler import (
    handle_describe_image_intent,
    handle_edit_image_intent,
    handle_generate_image_intent,
    send_debug_context,
)
from bot.provider_intents import (
    handle_claude_chat_intent,
    handle_gemini_chat_intent,
    handle_summarize_url_intent,
)
from bot.video_handler import handle_generate_video_intent
from services.stock_utils import handle_stock_command
from services.weather_utils import handle_weather_request
from services.behavior_registry import dispatch_intent_override, get_runtime_setting

# "imagine ..." normally means generate an image, but when an image is present
# and the ask is to translate/describe/explain/read IT, "imagine" is used in its
# plain-English sense ("imagine a translation of this image") — a describe task.
_IMAGINE_DESCRIBE_RE = re.compile(
    r"\b(?:translat\w*|describ\w*|descript\w*|explain\w*|caption\w*|transcri\w*|"
    r"summar\w*|decipher\w*|read\w*|what\s+(?:does|do|is|it|this))\b",
    re.IGNORECASE,
)

_CODE_CONTROL_RE = {
    "code_approve": re.compile(r"\b(?:approve|accept|ship|deploy)\b.*\b(?:proposal|patch|code change|change)\b", re.I),
    "code_reject": re.compile(r"\b(?:reject|cancel|discard|drop)\b.*\b(?:proposal|patch|code change|change)\b", re.I),
    "code_status": re.compile(r"\b(?:status|progress|deployed|deployment|what happened)\b.*\b(?:proposal|patch|code change|change)\b", re.I),
    "code_rollback": re.compile(r"\b(?:rollback|roll back|undo|revert)\b.*\b(?:proposal|patch|code change|change|deployment)\b", re.I),
}
_CODE_CHANGE_RE = re.compile(
    r"(?:^/code_propose\b|\b(?:change|modify|edit|rewrite|refactor|add|remove|fix)\b.{0,80}"
    r"\b(?:(?:your|the bot(?:'s)?)\s+"
    r"(?:code|codebase|source|implementation|structure|routing|readme|command|response)|"
    r"(?:the\s+)?(?:codebase|repository|repo)))",
    re.I,
)

# Keep explicit video-generation requests working even when the LLM intent
# classifier is unavailable. Provider names are deliberately irrelevant here:
# "gemini make this into a video" describes an operation first and a backend
# second. This only matches creation language, so requests such as "summarize
# this video" still fall through to the contextual classifier.
_VIDEO_GENERATION_RE = re.compile(
    r"(?:\b(?:generate|create|make|render)\s+(?:me\s+)?(?:an?\s+)?(?:\w+[ -]+){0,3}"
    r"(?:video|clip|movie|animation)\b|"
    r"\b(?:make|turn|convert)\s+(?:this|that|it|the\s+(?:image|photo|picture))\s+"
    r"(?:into|to|as)\s+(?:an?\s+)?(?:video|clip|movie|animation)\b|"
    r"\banimate\s+(?:this|that|it|the\s+(?:image|photo|picture))\b|"
    r"\b(?:image|photo|picture)\s*[- ]?to[- ]?video\b)",
    re.I,
)

_CODE_STATUS_CUE_RE = re.compile(
    r"\b(?:status|progress|deploy(?:ed|ment|ing)?|pass(?:ed|ing)?|fail(?:ed|ure|ing)?|"
    r"validat(?:e|ed|ion|ing)|active|live|finish(?:ed)?|complete(?:d)?|"
    r"approv(?:e|ed|al)|reject(?:ed|ion)?|reviewable|rolled\s+back)\b|"
    r"\bwhat\s+happened\b|\bhow(?:'s|\s+is)\s+(?:it|that|the\s+change)\s+going\b|"
    r"\bdid\s+(?:it|that|(?:that|the)\s+change)\s+(?:take|work|ship)\b",
    re.I,
)
_CODE_PROPOSAL_REF_RE = re.compile(
    r"\b(?:proposal|patch|code\s+change|change\s+request|baseline)\b|"
    r"\b[a-z]+(?:-[a-z]+){1,3}\b",
    re.I,
)

_WEATHER_CUE_RE = re.compile(
    r"\b(?:weather|forecast|temp(?:erature)?s?|rain\w*|snow\w*|humid\w*|wind\w*|"
    r"sunny|cloud\w*|storm\w*|fog\w*|hail|sleet|drizzl\w*|degrees|"
    r"heat\s+index|wind\s*chill|uv\s+index|air\s+quality|"
    r"(?:hot|cold|warm|chilly|freezing|nice)\s+(?:out(?:side)?|today|tonight|tomorrow)|"
    r"outside)\b",
    re.I,
)

def validate_classified_intent(
    intent: str,
    prompt: str,
    *,
    has_attachments: bool = False,
) -> str:
    """Reject code-status guesses based only on conversational context.

    Status lookup defaults to the newest proposal, so a false positive exposes
    an unrelated card. Require the current message to express a status idea or
    identify a proposal. Explicit keyword-routed controls bypass this guard.

    Same idea for get_weather: the dedicated handler treats the whole message
    as a location, so a false positive produces "couldn't resolve that
    location (<entire message>)". Require actual weather vocabulary; anything
    else goes to chat, which has a weather tool anyway.
    """
    text = prompt or ""
    if intent == "chat_reverse_image" and not has_attachments:
        return "clarify"
    if intent == "get_weather" and not _WEATHER_CUE_RE.search(text):
        return "chat"
    if intent in {"chat_tiny", "chat_light", "chat_standard", "chat"} and requires_fresh_web(text):
        return "chat_research"
    if intent != "code_status":
        return intent
    if _CODE_STATUS_CUE_RE.search(text) or _CODE_PROPOSAL_REF_RE.search(text):
        return intent
    return "chat"


def resolve_keyword_intent(raw_prompt: str, prompt: str, has_attachments: bool) -> Optional[str]:
    """Keywords decide the PROVIDER, the LLM classifier decides the INTENT.

    The only intent shortcut left is an explicit "claude ..." prefix (Claude
    only does chat here). Everything else — including "gemini ..." messages —
    goes to the classifier, which understands 'gemini generate an image of X'
    means image generation. Which backend renders it is decided separately
    (wants_gemini)."""
    lowered_raw = raw_prompt.lower().strip()
    lowered_prompt = prompt.lower().strip()

    for intent, pattern in _CODE_CONTROL_RE.items():
        if pattern.search(prompt or raw_prompt):
            return intent
    if _CODE_CHANGE_RE.search(prompt or raw_prompt):
        return "code_change"

    if _VIDEO_GENERATION_RE.search(prompt or raw_prompt):
        return "generate_video"

    if lowered_prompt.startswith("claude") or lowered_raw.startswith("claude"):
        return "claude_chat"

    if is_reverse_image_request(prompt or raw_prompt, has_images=has_attachments):
        return "chat_reverse_image"

    # "imagine ..." is this bot's established image command word — always an
    # image request, with or without a leading provider name. (Provider is
    # still chosen separately via wants_gemini.) Attachments do not change the
    # route: generate_image now carries them down as reference images, so
    # "imagine this in an art museum" renders the attached picture rather than
    # skipping the classifier and losing it.
    without_provider = re.sub(r"^gemini[\s,:]+", "", lowered_prompt)
    if without_provider.startswith(("imagine ", "imagine:")):
        # With an image present, "imagine a translation/description of this
        # image" is a describe task, not a new generation — defer to the
        # classifier (-> describe_image) rather than force-generating.
        if has_attachments and _IMAGINE_DESCRIBE_RE.search(without_provider):
            return None
        return "generate_image"

    return None


def wants_gemini(prompt: str) -> bool:
    """Provider selection: the user said gemini -> use gemini; otherwise the
    default provider (OpenAI). Word-boundary aware so 'the gemini zodiac
    sign' in the middle of a scene description doesn't count, but leading
    'gemini ...' or 'with/using gemini' does."""
    lowered = (prompt or "").lower().strip()
    return (
        lowered.startswith("gemini")
        or "with gemini" in lowered
        or "using gemini" in lowered
        or "use gemini" in lowered
    )


def get_duration_estimate(intent: str) -> int:
    default = {
        "generate_image": 40,
        "edit_image": 40,
        "summarize_url": 10,
        "describe_image": 8,
        "chat_tiny": 2,
        "chat_light": 3,
        "chat_standard": 4,
        "chat": 6,
        "chat_research": 8,
        "chat_reverse_image": 12,
        "chat_deep": 10,
        "clarify": 3,
        "get_weather": 5,
        "get_stock": 5,
    }.get(intent, 12)
    value = get_runtime_setting(f"intent.duration.{intent}", default)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def chat_model_for_intent(intent: str) -> str:
    from providers.openai_client import (
        OPENAI_CHAT_MODEL,
        OPENAI_DEEP_MODEL,
        OPENAI_LIGHT_MODEL,
        OPENAI_STANDARD_MODEL,
        OPENAI_TINY_MODEL,
    )

    default = {
        "chat_tiny": OPENAI_TINY_MODEL,
        "chat_light": OPENAI_LIGHT_MODEL,
        "chat_standard": OPENAI_STANDARD_MODEL,
        "chat": OPENAI_CHAT_MODEL,
        "chat_research": OPENAI_CHAT_MODEL,
        "chat_reverse_image": OPENAI_DEEP_MODEL,
        "chat_deep": OPENAI_DEEP_MODEL,
        "clarify": OPENAI_LIGHT_MODEL,
    }.get(intent, OPENAI_CHAT_MODEL)
    value = get_runtime_setting(f"intent.model.{intent}", default)
    return value if isinstance(value, str) and value.strip() else default


def chat_reasoning_for_intent(intent: str) -> str:
    """Spend reasoning only where the task shape warrants it."""
    default = {
        "chat_tiny": "none",
        "chat_light": "none",
        "chat_standard": "low",
        "chat": "low",
        "chat_research": "medium",
        "chat_reverse_image": "high",
        "chat_deep": "high",
        "clarify": "none",
    }.get(intent, "low")
    value = get_runtime_setting(f"intent.reasoning.{intent}", default)
    return value if value in {"none", "low", "medium", "high", "xhigh", "max"} else default


@dataclass
class DispatchContext:
    """Everything a single triggered message needs to be routed and answered."""

    intent: str
    message: Any
    prompt: str
    raw_prompt: str
    user_id: Any
    bot_user: Any
    ref_msg: Any = None
    is_reply_to_bot: bool = False
    is_owner: bool = False
    image_urls: List[Any] = field(default_factory=list)
    source_image_urls: List[str] = field(default_factory=list)
    gemini_parts: List[Any] = field(default_factory=list)
    video_inputs: List[str] = field(default_factory=list)
    channel_context: List[dict[str, str]] = field(default_factory=list)
    general_url_match: Any = None
    stream_ok: bool = False
    response_gate_checked: bool = False
    # Injected collaborators (Discord-side helpers owned by discord_bot.py)
    get_location_details: Optional[Callable] = None
    get_weather_data: Optional[Callable] = None
    live_status_with_progress: Optional[Callable] = None
    send_or_edit_with_truncation: Optional[Callable] = None
    prompt_for_image_selection: Optional[Callable] = None
    moderation_view_factory: Any = None


async def _dispatch_builtin_intent(ctx: DispatchContext) -> bool:
    duration_estimate = get_duration_estimate(ctx.intent)
    message = ctx.message

    if ctx.intent == "get_weather":
        response = await handle_weather_request(
            message,
            ctx.bot_user.id,
            ctx.get_location_details,
            ctx.get_weather_data,
            None,
            f"{message.guild.id}-{message.channel.id}",
            message.author.id,
        )
        if response:
            await ctx.send_or_edit_with_truncation(response, channel=message.channel, reply_to=message)
        return True

    if ctx.intent == "claude_chat":
        await handle_claude_chat_intent(
            message=message,
            prompt=ctx.prompt,
            stream_ok=ctx.stream_ok,
            live_status_with_progress=ctx.live_status_with_progress,
            send_or_edit_with_truncation=ctx.send_or_edit_with_truncation,
            image_urls=ctx.image_urls,
            source_image_urls=ctx.source_image_urls,
            ref_msg=ctx.ref_msg,
            channel_context=ctx.channel_context,
        )
        return True

    if ctx.intent == "gemini_chat":
        await handle_gemini_chat_intent(
            message=message,
            prompt=ctx.prompt,
            gemini_parts=ctx.gemini_parts,
            live_status_with_progress=ctx.live_status_with_progress,
            send_or_edit_with_truncation=ctx.send_or_edit_with_truncation,
            moderation_view_factory=ctx.moderation_view_factory,
            channel_context=ctx.channel_context,
            is_owner=ctx.is_owner,
        )
        return True

    if ctx.intent == "generate_image":
        if message.content.startswith("/debug_context"):
            await send_debug_context(message, ctx.bot_user)
            return True
        await handle_generate_image_intent(
            message=message,
            prompt=ctx.prompt,
            ref_msg=ctx.ref_msg,
            duration_estimate=duration_estimate,
            stream_ok=ctx.stream_ok,
            live_status_with_progress=ctx.live_status_with_progress,
            use_gemini=wants_gemini(ctx.prompt) or wants_gemini(ctx.raw_prompt),
            # A generation can have a subject the user attached ("imagine this
            # in an art museum"). Only edit_image used to receive these, so
            # that picture was dropped and the model invented a subject.
            image_urls=ctx.image_urls,
        )
        return True

    if ctx.intent == "edit_image" and ctx.image_urls:
        use_gemini = wants_gemini(ctx.prompt) or wants_gemini(ctx.raw_prompt)
        await handle_edit_image_intent(
            message=message,
            prompt=ctx.prompt,
            image_urls=ctx.image_urls,
            prompt_for_image_selection=ctx.prompt_for_image_selection,
            live_status_with_progress=ctx.live_status_with_progress,
            use_gemini=use_gemini,
        )
        return True

    if ctx.intent == "summarize_url" and ctx.general_url_match and not ctx.image_urls:
        await handle_summarize_url_intent(
            message=message,
            url=ctx.general_url_match.group(0),
            duration_estimate=duration_estimate,
            stream_ok=ctx.stream_ok,
            live_status_with_progress=ctx.live_status_with_progress,
            send_or_edit_with_truncation=ctx.send_or_edit_with_truncation,
        )
        return True

    if ctx.intent == "describe_image" and ctx.image_urls:
        await handle_describe_image_intent(
            message=message,
            prompt=ctx.prompt,
            image_urls=ctx.image_urls,
            ref_msg=ctx.ref_msg,
            is_reply_to_bot=ctx.is_reply_to_bot,
            duration_estimate=duration_estimate,
            stream_ok=ctx.stream_ok,
            live_status_with_progress=ctx.live_status_with_progress,
            send_or_edit_with_truncation=ctx.send_or_edit_with_truncation,
            channel_context=ctx.channel_context,
        )
        return True

    if ctx.intent == "generate_video":
        await handle_generate_video_intent(
            message=message,
            prompt=ctx.prompt,
            user_id=ctx.user_id,
            live_status_with_progress=ctx.live_status_with_progress,
            stream_ok=ctx.stream_ok,
            image_urls=ctx.image_urls,
            ref_msg=ctx.ref_msg,
        )
        return True

    if ctx.intent == "get_stock" and ctx.prompt.lower().startswith("stock"):
        async with message.channel.typing():
            await handle_stock_command(message, ctx.prompt)
        return True

    # A clip only gets watched by the backend that can watch it. OpenAI chat
    # takes images alone, so an attached video routed there reaches the model
    # as nothing at all; Gemini reads it inline from ctx.gemini_parts.
    chat_model = chat_model_for_intent(ctx.intent)
    if ctx.video_inputs and "gemini" not in chat_model.lower():
        chat_model = GEMINI_FALLBACK_CHAT_MODEL

    # 'clarify' runs as normal chat WITH full conversation history plus a
    # nudge: resolve the ambiguity from context if possible ("do that" right
    # after the bot offered something), otherwise ask one short question.
    # A blind clarifying question without history made the bot forget its own
    # previous offer.
    await handle_chat_intent(
        message=message,
        prompt=ctx.prompt,
        raw_prompt=ctx.raw_prompt,
        user_id=ctx.user_id,
        ref_msg=ctx.ref_msg,
        is_reply_to_bot=ctx.is_reply_to_bot,
        image_urls=ctx.image_urls,
        source_image_urls=ctx.source_image_urls,
        gemini_parts=ctx.gemini_parts,
        channel_context=ctx.channel_context,
        duration_estimate=duration_estimate,
        stream_ok=ctx.stream_ok,
        live_status_with_progress=ctx.live_status_with_progress,
        send_or_edit_with_truncation=ctx.send_or_edit_with_truncation,
        moderation_view_factory=ctx.moderation_view_factory,
        default_model=chat_model,
        agent_intent=ctx.intent,
        reasoning_effort=chat_reasoning_for_intent(ctx.intent),
        clarify_hint=(ctx.intent == "clarify"),
        force_web_search=(ctx.intent == "chat_research"),
        force_reverse_image_search=(ctx.intent == "chat_reverse_image"),
        is_owner=ctx.is_owner,
    )
    return True


async def dispatch_intent(ctx: DispatchContext) -> bool:
    """Dispatch through one captured live generation, then use core fallback.

    A live component can own one exact classifier intent without replacing the
    stable Discord message shell. Requests already in progress retain their
    captured callback until they finish.
    """
    # Return handled, not False: silence must not trigger a fallback response.
    # Ingress checks before any UI; keep a guard for direct dispatch callers
    # without classifying a normal Discord request twice.
    if not ctx.response_gate_checked and await should_suppress_response(ctx.raw_prompt or ctx.prompt):
        return True

    handled, result = await dispatch_intent_override(ctx.intent, ctx)
    if handled:
        return True if result is None else bool(result)
    return await _dispatch_builtin_intent(ctx)
