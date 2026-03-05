from __future__ import annotations

from bot.chat_handler import handle_chat_intent
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


def get_duration_estimate(intent: str) -> int:
    return {
        "generate_image": 40,
        "edit_image": 40,
        "summarize_url": 10,
        "describe_image": 8,
        "chat": 6,
        "get_weather": 5,
        "get_stock": 5,
    }.get(intent, 12)


async def dispatch_intent(
    *,
    intent: str,
    message,
    prompt: str,
    raw_prompt: str,
    user_id,
    ref_msg,
    is_reply_to_bot: bool,
    image_urls,
    gemini_parts,
    general_url_match,
    stream_ok: bool,
    bot_user,
    get_location_details,
    get_weather_data,
    live_status_with_progress,
    send_or_edit_with_truncation,
    prompt_for_image_selection,
    moderation_view_factory,
):
    duration_estimate = get_duration_estimate(intent)

    if intent == "get_weather":
        response = await handle_weather_request(
            message,
            bot_user.id,
            get_location_details,
            get_weather_data,
            None,
            f"{message.guild.id}-{message.channel.id}",
            message.author.id,
        )
        if response:
            await send_or_edit_with_truncation(response, channel=message.channel, reply_to=message)
        return True

    if intent == "claude_chat":
        await handle_claude_chat_intent(
            message=message,
            prompt=prompt,
            stream_ok=stream_ok,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
        )
        return True

    if intent == "gemini_chat":
        await handle_gemini_chat_intent(
            message=message,
            prompt=prompt,
            gemini_parts=gemini_parts,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
            moderation_view_factory=moderation_view_factory,
        )
        return True

    if intent == "generate_image":
        if message.content.startswith("/debug_context"):
            await send_debug_context(message, bot_user)
            return True
        await handle_generate_image_intent(
            message=message,
            prompt=prompt,
            duration_estimate=duration_estimate,
            stream_ok=stream_ok,
            live_status_with_progress=live_status_with_progress,
        )
        return True

    if intent == "edit_image" and image_urls:
        await handle_edit_image_intent(
            message=message,
            prompt=prompt,
            image_urls=image_urls,
            prompt_for_image_selection=prompt_for_image_selection,
            live_status_with_progress=live_status_with_progress,
        )
        return True

    if intent == "summarize_url" and general_url_match and not image_urls:
        await handle_summarize_url_intent(
            message=message,
            url=general_url_match.group(0),
            duration_estimate=duration_estimate,
            stream_ok=stream_ok,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
        )
        return True

    if intent == "describe_image" and image_urls:
        await handle_describe_image_intent(
            message=message,
            prompt=prompt,
            image_urls=image_urls,
            ref_msg=ref_msg,
            is_reply_to_bot=is_reply_to_bot,
            duration_estimate=duration_estimate,
            stream_ok=stream_ok,
            live_status_with_progress=live_status_with_progress,
            send_or_edit_with_truncation=send_or_edit_with_truncation,
        )
        return True

    if intent == "generate_video":
        await handle_generate_video_intent(
            message=message,
            prompt=prompt,
            user_id=user_id,
            live_status_with_progress=live_status_with_progress,
            stream_ok=stream_ok,
        )
        return True

    if intent == "get_stock" and prompt.lower().startswith("stock"):
        async with message.channel.typing():
            await handle_stock_command(message, prompt)
        return True

    await handle_chat_intent(
        message=message,
        prompt=prompt,
        raw_prompt=raw_prompt,
        user_id=user_id,
        ref_msg=ref_msg,
        is_reply_to_bot=is_reply_to_bot,
        image_urls=image_urls,
        gemini_parts=gemini_parts,
        duration_estimate=duration_estimate,
        stream_ok=stream_ok,
        live_status_with_progress=live_status_with_progress,
        send_or_edit_with_truncation=send_or_edit_with_truncation,
        moderation_view_factory=moderation_view_factory,
    )
    return True
