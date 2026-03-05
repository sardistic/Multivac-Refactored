import logging

from bot.chat_context import build_chat_context
from providers.gemini_utils import GeminiModerationError, generate_gemini_text
from providers.openai_client import OPENAI_CHAT_MODEL
from providers.openai_utils import OpenAIModerationError, TOOLS_DEF, generate_openai_messages_response_with_tools

logger = logging.getLogger("discord_bot")


async def handle_chat_intent(
    *,
    message,
    prompt: str,
    raw_prompt: str,
    user_id,
    ref_msg,
    is_reply_to_bot: bool,
    image_urls,
    gemini_parts,
    duration_estimate: int,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
    moderation_view_factory,
):
    async def _do_chat_generation(model_name=None):
        selected_model = model_name or OPENAI_CHAT_MODEL

        async def _chat_with_es_window():
            msgs = build_chat_context(
                message=message,
                user_id=user_id,
                raw_prompt=raw_prompt,
                ref_msg=ref_msg,
                is_reply_to_bot=is_reply_to_bot,
            )
            ctx = {
                "guild_id": message.guild.id if message.guild else "DM",
                "channel_id": message.channel.id,
                "user_id": user_id,
            }

            if "gemini" in selected_model.lower():
                status_res = {"text": ""}
                text_resp, artifacts = generate_gemini_text(
                    prompt=prompt,
                    context=msgs,
                    extra_parts=gemini_parts or None,
                    status_tracker=status_res,
                    enable_code_execution=False,
                    search_ids=ctx,
                    model_name=selected_model,
                )
                return text_resp

            if image_urls:
                msgs = list(msgs)
                msgs[-1] = {
                    "role": "user",
                    "content": [{"type": "text", "text": raw_prompt}] + [
                        {"type": "image_url", "image_url": {"url": u}} for u in image_urls
                    ],
                }

            return await generate_openai_messages_response_with_tools(
                msgs,
                tools=TOOLS_DEF,
                tool_context=ctx,
                model=selected_model,
            )

        def _summarizer():
            return f"• Using {selected_model}…\n• Drafting answer…"

        try:
            status_msg, response = await live_status_with_progress(
                message,
                action_label=f"Responding ({selected_model})",
                emoji="💬",
                coro=_chat_with_es_window(),
                duration_estimate=duration_estimate,
                summarizer=_summarizer if stream_ok else None,
            )

            if response and response.strip():
                await send_or_edit_with_truncation(
                    response,
                    target_msg=status_msg,
                    original_message=message,
                    model=selected_model,
                )
            else:
                await status_msg.edit(content="🤖 INSUFFICIENT DATA FOR MEANINGFUL ANSWER")

        except OpenAIModerationError as e:
            logger.warning(f"OpenAI moderation hit: {e}")
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_chat_generation)
            await message.reply(user_msg, view=view)
        except GeminiModerationError as e:
            logger.warning(f"Gemini moderation hit in Chat fallback: {e}")
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_chat_generation)
            await message.reply(user_msg, view=view)

    await _do_chat_generation()
