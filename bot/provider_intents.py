from __future__ import annotations

import asyncio
import io
import logging
import mimetypes
import re

import discord

from providers.claude_utils import generate_claude_response
from providers.gemini_utils import GeminiModerationError, generate_gemini_text
from providers.openai_utils import OpenAIModerationError, TOOLS_DEF, generate_openai_messages_response_with_tools
from services.memory_utils import build_message_window
from services.url_utils import extract_main_text, fetch_url_content, reduce_text_length

logger = logging.getLogger("discord_bot")


async def handle_claude_chat_intent(
    *,
    message,
    prompt: str,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
):
    clean_prompt = re.sub(r"^(claude|hey claude)\s*", "", prompt, flags=re.IGNORECASE).strip()
    context_msgs = build_message_window(
        guild_id=message.guild.id if message.guild else "DM",
        channel_id=message.channel.id,
        user_id=message.author.id,
        limit_msgs=20,
    )
    context_msgs = [
        m for m in context_msgs
        if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
    ]
    claude_messages = [{"role": "system", "content": "You are Claude, a helpful AI assistant."}]
    claude_messages.extend(context_msgs)
    claude_messages.append({"role": "user", "content": clean_prompt})

    status_msg, response = await live_status_with_progress(
        message,
        action_label="Thinking (Claude)",
        emoji="🧠",
        coro=generate_claude_response(claude_messages),
        duration_estimate=5,
        summarizer=(lambda: "Queries Anthropic API...") if stream_ok else None,
    )

    if response:
        await send_or_edit_with_truncation(
            response,
            target_msg=status_msg,
            original_message=message,
            model="claude-sonnet-4",
        )
    else:
        await status_msg.edit(content="❌ Claude returned no response.")


async def handle_gemini_chat_intent(
    *,
    message,
    prompt: str,
    gemini_parts,
    live_status_with_progress,
    send_or_edit_with_truncation,
    moderation_view_factory,
):
    clean_prompt = re.sub(r"^gemini\s*", "", prompt, flags=re.IGNORECASE).strip()
    is_test_mode = False
    if clean_prompt.lower() == "test" or clean_prompt.lower().startswith("test "):
        is_test_mode = True
        clean_prompt = re.sub(r"^test\s*", "", clean_prompt, flags=re.IGNORECASE).strip()

    context_msgs = build_message_window(
        guild_id=message.guild.id if message.guild else "DM",
        channel_id=message.channel.id,
        user_id=message.author.id,
        limit_msgs=20,
    )
    context_msgs = [
        m for m in context_msgs
        if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
    ]

    enable_code_execution = False
    if clean_prompt.lower().startswith("code "):
        enable_code_execution = True
        clean_prompt = clean_prompt[5:].strip()
    elif is_test_mode:
        enable_code_execution = True

    status_tracker = {"text": ""}

    def _live_code_summarizer():
        return status_tracker["text"] or "Using Gemini 1.5 Flash..."

    search_ids = {
        "guild_id": str(message.guild.id) if message.guild else "DM",
        "channel_id": str(message.channel.id),
        "user_id": str(message.author.id),
    }

    async def _do_gemini_generation(model_name=None):
        selected_model = model_name or "gemini-2.0-flash"

        async def _run_gen():
            if "gpt" in selected_model.lower():
                ctx = {
                    "guild_id": message.guild.id if message.guild else "DM",
                    "channel_id": message.channel.id,
                    "user_id": str(message.author.id),
                }
                msgs = list(context_msgs)
                msgs.append({"role": "user", "content": clean_prompt})
                txt = await generate_openai_messages_response_with_tools(
                    msgs,
                    tools=TOOLS_DEF,
                    tool_context=ctx,
                    model=selected_model,
                )
                return txt, []

            return await asyncio.to_thread(
                generate_gemini_text,
                clean_prompt,
                context=context_msgs,
                extra_parts=gemini_parts,
                status_tracker=status_tracker,
                enable_code_execution=enable_code_execution,
                search_ids=search_ids,
                model_name=selected_model,
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

            if not response:
                await status_msg.edit(content="❌ Gemini returned no response.")
                return

            if isinstance(response, tuple):
                text_resp, artifacts = response
            else:
                text_resp, artifacts = response, []

            files_to_send = []
            if artifacts:
                for i, (data, mime) in enumerate(artifacts):
                    ext = mimetypes.guess_extension(mime) or ".bin"
                    if "wav" in mime:
                        ext = ".wav"
                    files_to_send.append(discord.File(io.BytesIO(data), filename=f"artifact_{i}{ext}"))

            if text_resp:
                if is_test_mode:
                    if len(text_resp) > 1900:
                        try:
                            text_file = discord.File(io.BytesIO(text_resp.encode("utf-8")), filename="response.md")
                            await status_msg.edit(content="⚠️ **Test Result Too Long** -> Sent as file `response.md`")
                            all_files = [text_file, *files_to_send]
                            await status_msg.reply(files=all_files)
                        except Exception as e:
                            await status_msg.edit(content=f"❌ Test mode file send failed: {e}")
                    else:
                        try:
                            await status_msg.edit(content=f"```\n{text_resp[:1990]}\n```")
                            if files_to_send:
                                await status_msg.reply(files=files_to_send)
                        except Exception as e:
                            await status_msg.edit(content=f"❌ Test mode failed: {e}")
                else:
                    await send_or_edit_with_truncation(text_resp, target_msg=status_msg, extra_files=files_to_send)
            elif files_to_send:
                await status_msg.reply(files=files_to_send)
            else:
                await status_msg.edit(content="❌ Gemini returned no text or files.")
        except GeminiModerationError as e:
            logger.warning("Gemini moderation hit: %s", e)
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_gemini_generation)
            await message.reply(user_msg, view=view)
        except OpenAIModerationError as e:
            logger.warning("OpenAI moderation hit in Gemini fallback: %s", e)
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_gemini_generation)
            await message.reply(user_msg, view=view)
        except Exception as e:
            logger.exception("Gemini generation error")
            await message.reply(f"❌ Gemini Error: {e}")

    await _do_gemini_generation()


async def handle_summarize_url_intent(
    *,
    message,
    url: str,
    duration_estimate: int,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
):
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
        summarizer=(lambda: f"Fetching page…\nURL: {url}\nExtracting main content…") if stream_ok else None,
    )
    if summary:
        await send_or_edit_with_truncation(summary, target_msg=status_msg)
    else:
        await status_msg.edit(content="❌ Summary failed.")
