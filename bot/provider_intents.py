from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import re

import discord

from providers.claude_utils import CLAUDE_MODEL, generate_claude_response, image_input_to_block
from providers.gemini_utils import GeminiModerationError, generate_gemini_text
from providers.openai_utils import (
    OpenAIModerationError,
    generate_openai_messages_response,
    generate_openai_messages_response_with_tools,
)
from services.behavior_registry import invoke_provider
from bot.draft_verifier import verify_chat_draft
from bot.chat_context import build_shared_channel_system_message
from bot.research_policy import build_fresh_search_query, is_reverse_image_request, requires_fresh_web
from bot.response_policy import apply_personality_overrides, build_message_user_style_system_messages
from services.memory_utils import build_channel_message_window
from services.security_limits import check_rate_limit
from services.security_utils import public_error_detail
from services.url_utils import extract_main_text, fetch_url_content, reduce_text_length
from services.youtube_utils import extract_youtube_id, fetch_youtube_transcript

logger = logging.getLogger("discord_bot")


async def _generate_gemini_threaded(*args, **kwargs):
    return await asyncio.to_thread(generate_gemini_text, *args, **kwargs)


async def _youtube_transcript_for(text_with_url: str, max_chars: int = 9000) -> str | None:
    """If the text contains a YouTube URL, fetch the actual spoken transcript.
    Without this, summaries get built from the video description/metadata,
    which often covers 20 seconds of a 40-minute video.

    Long transcripts are map-reduce condensed through the cheap model tiers
    (so the whole video is represented, not just the first 9k chars) and the
    condensed form is cached per video id — one condensation cost per video,
    ever, no matter how many users ask about it."""
    vid = extract_youtube_id(text_with_url or "")
    if not vid:
        return None

    from services.database_utils import (
        get_cached_transcript_summary,
        set_cached_transcript_summary,
    )

    try:
        cached = await asyncio.to_thread(get_cached_transcript_summary, vid)
        if cached:
            return cached
    except Exception:
        logger.warning("transcript cache read failed", exc_info=True)

    try:
        transcript = await asyncio.to_thread(fetch_youtube_transcript, vid)
    except Exception:
        logger.warning("YouTube transcript fetch failed for %s", vid, exc_info=True)
        return None
    if not transcript:
        return None
    if len(transcript) <= max_chars:
        return transcript

    from services.condense import condense_long_text

    condensed = await condense_long_text(transcript, target_chars=max_chars)
    try:
        await asyncio.to_thread(set_cached_transcript_summary, vid, condensed)
    except Exception:
        logger.warning("transcript cache write failed", exc_info=True)
    return condensed


async def handle_claude_chat_intent(
    *,
    message,
    prompt: str,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
    image_urls=None,
    source_image_urls=None,
    ref_msg=None,
    channel_context=None,
):
    clean_prompt = re.sub(r"^(claude|hey claude)\s*", "", prompt, flags=re.IGNORECASE).strip()
    user_request_for_review = clean_prompt

    # When replying to another message, that message is usually the subject
    # ("claude fix why this happened" under a bot error). Quote it so Claude
    # sees what "this" is even if it has scrolled out of the history window.
    ref_content = (getattr(ref_msg, "content", None) or "").strip()
    if ref_content:
        ref_author = getattr(getattr(ref_msg, "author", None), "display_name", None) or "earlier message"
        clean_prompt = f'[Replying to {ref_author}: "{ref_content[:1500]}"]\n\n{clean_prompt}'

    context_msgs = list(channel_context or [])
    if not context_msgs:
        context_msgs = await asyncio.to_thread(
            build_channel_message_window,
            guild_id=message.guild.id if message.guild else "DM",
            channel_id=message.channel.id,
            current_user_id=message.author.id,
            current_display_name=getattr(message.author, "display_name", None),
            exclude_message_id=getattr(message, "id", None),
            limit_msgs=20,
        )
    context_msgs = [
        m for m in context_msgs
        if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
    ]
    claude_messages = [
        {
            "role": "system",
            "content": (
                "You are the application's user-facing conversational assistant. "
                "Answer accurately and preserve all provider safety and tool constraints."
            ),
        }
    ]
    claude_messages.append(build_shared_channel_system_message(message))
    claude_messages.extend(
        build_message_user_style_system_messages(
            message,
            intent="claude_chat",
        )
    )
    claude_messages.extend(context_msgs)

    # Attached/replied-to images become Anthropic image blocks so Claude can
    # actually see screenshots instead of asking for one that's already there.
    image_blocks = []
    for src in (image_urls or [])[:5]:
        block = image_input_to_block(src)
        if block:
            image_blocks.append(block)
    if image_blocks:
        user_content = image_blocks + ([{"type": "text", "text": clean_prompt}] if clean_prompt else [])
    else:
        user_content = clean_prompt
    claude_messages.append({"role": "user", "content": user_content})

    phase = {"detail": "Querying Anthropic API…"}
    tool_trace = []
    tool_context = {
        "guild_id": message.guild.id if message.guild else "DM",
        "channel_id": message.channel.id,
        "user_id": message.author.id,
        "image_urls": image_urls or [],
        "source_image_urls": source_image_urls or [],
        "intent": "claude_chat",
        "request_text": user_request_for_review,
    }
    force_reverse = is_reverse_image_request(
        user_request_for_review,
        has_images=bool(image_urls),
    )
    force_web = requires_fresh_web(user_request_for_review) and not force_reverse
    forced_tool = "reverse_image_search" if force_reverse else ("web_search" if force_web else None)
    forced_tool_args = (
        {"image_index": 0, "mode": "all", "max_results": 10}
        if force_reverse
        else ({"q": build_fresh_search_query(user_request_for_review)} if force_web else None)
    )

    async def _generate_claude_with_review():
        draft = await invoke_provider(
            "chat.claude",
            generate_claude_response,
            claude_messages,
            enable_tools=True,
            tool_context=tool_context,
            forced_tool=forced_tool,
            forced_tool_args=forced_tool_args,
            tool_trace=tool_trace,
            tool_call_limits=(
                {"reverse_image_search": 1, "web_search": 2, "summarize_url": 2}
                if force_reverse
                else None
            ),
        )
        if not draft or not draft.strip():
            return draft

        phase["detail"] = "Reviewing evidence, length, and voice…"
        verdict = await verify_chat_draft(
            user_id=message.author.id,
            display_name=getattr(message.author, "display_name", None),
            prompt=user_request_for_review,
            draft=draft,
            research_used=bool(
                force_web
                or force_reverse
                or any(
                    item.get("name") in {"web_search", "reverse_image_search"}
                    for item in tool_trace
                )
            ),
        )
        logger.info("Claude-route post-draft verdict=%s", verdict.action)
        if verdict.action == "accept":
            return draft
        if verdict.action == "revise":
            return verdict.revised_answer or draft

        phase["detail"] = "Checking stronger current evidence…"
        query = build_fresh_search_query(
            verdict.research_query or user_request_for_review
        )
        repair_messages = list(claude_messages)
        repair_messages.extend(
            [
                {"role": "assistant", "content": draft},
                {
                    "role": "system",
                    "content": (
                            "A post-draft verifier found insufficient fresh evidence. Search "
                            "again with the provided query, correct stale or unsupported claims, "
                            "preserve an appropriate length and the user's established style, "
                            f"and cite strong URLs. Start with: {query}"
                    ),
                },
                {
                    "role": "user",
                    "content": "Return the corrected, source-grounded answer only.",
                },
            ]
        )
        repaired = await invoke_provider(
            "chat.claude",
            generate_claude_response,
            repair_messages,
            enable_tools=True,
            tool_context=tool_context,
            forced_tool="web_search",
            forced_tool_args={"q": query},
            tool_trace=tool_trace,
            tool_call_limits=(
                {"web_search": 2, "summarize_url": 2} if force_reverse else None
            ),
        )
        return repaired or draft

    def _claude_progress():
        if tool_trace:
            latest = tool_trace[-1]
            return (
                f"{phase['detail']}\n"
                f"• Tool {latest.get('name', 'unknown')}: {latest.get('status', 'running')}"
            )
        return phase["detail"]

    status_msg, response = await live_status_with_progress(
        message,
        action_label="Thinking (Claude)",
        emoji="🧠",
        coro=_generate_claude_with_review(),
        duration_estimate=5,
        summarizer=_claude_progress if stream_ok else None,
    )

    if response:
        response = apply_personality_overrides(message.author.id, intent="claude_chat", text=response)
        await send_or_edit_with_truncation(
            response,
            target_msg=status_msg,
            original_message=message,
            model=CLAUDE_MODEL,
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
    channel_context=None,
    is_owner: bool = False,
):
    clean_prompt = re.sub(r"^gemini\s*", "", prompt, flags=re.IGNORECASE).strip()
    is_test_mode = False
    if clean_prompt.lower() == "test" or clean_prompt.lower().startswith("test "):
        is_test_mode = True
        clean_prompt = re.sub(r"^test\s*", "", clean_prompt, flags=re.IGNORECASE).strip()

    enable_code_execution = False
    if clean_prompt.lower().startswith("code "):
        enable_code_execution = True
        clean_prompt = clean_prompt[5:].strip()
    elif is_test_mode:
        enable_code_execution = True
    user_request_for_review = clean_prompt

    if enable_code_execution:
        compute_quota = check_rate_limit(
            "compute",
            user_id=str(message.author.id),
            guild_id=str(message.guild.id) if message.guild else "DM",
        )
        if not compute_quota.allowed:
            await message.reply(
                "⏳ Code-execution quota reached for now. "
                f"Try again in about {compute_quota.retry_after} seconds."
            )
            return

    if extract_youtube_id(clean_prompt):
        url_quota = check_rate_limit(
            "url",
            user_id=str(message.author.id),
            guild_id=str(message.guild.id) if message.guild else "DM",
        )
        if not url_quota.allowed:
            await message.reply(
                "⏳ URL-fetch quota reached for now. "
                f"Try again in about {url_quota.retry_after} seconds."
            )
            return

    context_msgs = []
    if not enable_code_execution:
        context_msgs = list(channel_context or [])
        if not context_msgs:
            context_msgs = await asyncio.to_thread(
                build_channel_message_window,
                guild_id=message.guild.id if message.guild else "DM",
                channel_id=message.channel.id,
                current_user_id=message.author.id,
                current_display_name=getattr(message.author, "display_name", None),
                exclude_message_id=getattr(message, "id", None),
                limit_msgs=20,
            )
        context_msgs = [
            m for m in context_msgs
            if not (m.get("role") == "user" and "gemini imagine" in m.get("content", "").lower())
        ]
        context_msgs = [
            build_shared_channel_system_message(message),
            *build_message_user_style_system_messages(
                message,
                intent="gemini_chat",
            ),
            *context_msgs,
        ]

    # Base freshness routing on the user's request. A linked video's transcript
    # may contain incidental terms such as "latest" that should not change it.
    force_web_search = requires_fresh_web(clean_prompt)
    fresh_search_query = (
        build_fresh_search_query(clean_prompt) if force_web_search else None
    )

    # Ground YouTube links in the real transcript, not the description.
    transcript = await _youtube_transcript_for(clean_prompt)
    if transcript:
        clean_prompt += (
            "\n\n[VIDEO TRANSCRIPT — this is the linked video's actual spoken content. "
            "Base your answer on THIS, not the title/description/thumbnail:]\n" + transcript
        )

    status_tracker = {"text": ""}

    def _live_code_summarizer():
        if force_web_search:
            return status_tracker["text"] or "Checking current sources…"
        return status_tracker["text"] or "Using Gemini 3.7 Flash..."

    search_ids = {
        "guild_id": str(message.guild.id) if message.guild else "DM",
        "channel_id": str(message.channel.id),
        "user_id": str(message.author.id),
    }

    async def _do_gemini_generation(model_name=None):
        selected_model = model_name or "gemini-3.8-flash"

        async def _generate_once(
            *,
            request_text: str,
            generation_context,
            must_search: bool,
            search_query: str | None,
            tool_trace,
        ):
            if "gpt" in selected_model.lower():
                ctx = {
                    "guild_id": message.guild.id if message.guild else "DM",
                    "channel_id": message.channel.id,
                    "user_id": str(message.author.id),
                    "is_owner": bool(is_owner),
                    "latest_user_text": prompt,
                }
                msgs = list(generation_context)
                msgs.append({"role": "user", "content": request_text})
                txt = await invoke_provider(
                    "chat.openai",
                    generate_openai_messages_response_with_tools,
                    msgs,
                    tool_context=ctx,
                    model=selected_model,
                    forced_tool="web_search" if must_search else None,
                    forced_tool_args={"q": search_query} if search_query else None,
                    tool_trace=tool_trace,
                )
                return txt, []

            return await invoke_provider(
                "chat.gemini",
                _generate_gemini_threaded,
                request_text,
                context=generation_context,
                extra_parts=gemini_parts,
                status_tracker=status_tracker,
                enable_code_execution=enable_code_execution,
                search_ids=search_ids,
                model_name=selected_model,
                force_web_search=must_search,
            )

        async def _run_gen():
            tool_trace = []
            response = await _generate_once(
                request_text=clean_prompt,
                generation_context=context_msgs,
                must_search=force_web_search,
                search_query=fresh_search_query,
                tool_trace=tool_trace,
            )
            if not response or enable_code_execution:
                return response

            if isinstance(response, tuple):
                draft, artifacts = response
            else:
                draft, artifacts = response, []
            if not draft or not draft.strip():
                return response

            status_tracker["text"] = "Reviewing evidence, length, and voice…"
            verdict = await verify_chat_draft(
                user_id=message.author.id,
                display_name=getattr(message.author, "display_name", None),
                prompt=user_request_for_review,
                draft=draft,
                research_used=(
                    force_web_search
                    or any(item.get("name") == "web_search" for item in tool_trace)
                ),
            )
            logger.info("Gemini-route post-draft verdict=%s", verdict.action)
            if verdict.action == "accept":
                return draft, artifacts
            if verdict.action == "revise":
                return verdict.revised_answer or draft, artifacts

            status_tracker["text"] = "Checking stronger current evidence…"
            query = build_fresh_search_query(
                verdict.research_query or user_request_for_review
            )
            repair_context = list(context_msgs)
            repair_context.extend(
                [
                    {"role": "assistant", "content": draft},
                    {
                        "role": "system",
                        "content": (
                            "A post-draft verifier found insufficient fresh evidence. "
                            "Discard stale or unsupported claims and re-answer from current "
                            f"sources. Start with this research query: {query}"
                        ),
                    },
                ]
            )
            repaired = await _generate_once(
                request_text="Return the corrected, source-grounded answer only.",
                generation_context=repair_context,
                must_search=True,
                search_query=query,
                tool_trace=[],
            )
            if isinstance(repaired, tuple):
                repaired_text, repaired_artifacts = repaired
            else:
                repaired_text, repaired_artifacts = repaired, []
            return repaired_text or draft, repaired_artifacts or artifacts

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
                text_resp = apply_personality_overrides(message.author.id, intent="gemini_chat", text=text_resp)
                if is_test_mode:
                    if len(text_resp) > 1900:
                        try:
                            text_file = discord.File(io.BytesIO(text_resp.encode("utf-8")), filename="response.md")
                            await status_msg.edit(content="⚠️ **Test Result Too Long** -> Sent as file `response.md`")
                            all_files = [text_file, *files_to_send]
                            await status_msg.reply(files=all_files)
                        except Exception as e:
                            logger.exception("Gemini test-mode file delivery failed")
                            await status_msg.edit(
                                content=f"❌ Test-mode delivery failed because {public_error_detail(e)}."
                            )
                    else:
                        try:
                            await status_msg.edit(content=f"```\n{text_resp[:1990]}\n```")
                            if files_to_send:
                                await status_msg.reply(files=files_to_send)
                        except Exception as e:
                            logger.exception("Gemini test-mode delivery failed")
                            await status_msg.edit(
                                content=f"❌ Test-mode delivery failed because {public_error_detail(e)}."
                            )
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
            await message.reply(
                f"❌ Gemini couldn't complete the request because {public_error_detail(e)}."
            )

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
    quota = check_rate_limit(
        "url",
        user_id=str(message.author.id),
        guild_id=str(message.guild.id) if message.guild else "DM",
    )
    if not quota.allowed:
        await message.reply(
            "⏳ URL-fetch quota reached for now. "
            f"Try again in about {quota.retry_after} seconds."
        )
        return

    async def _do_summarize():
        # YouTube: summarize the actual transcript, not the page metadata.
        transcript = await _youtube_transcript_for(url)
        if transcript:
            title = "YouTube video"
            condensed = reduce_text_length(transcript, max_chars=9000)
            source_note = "This is the video's spoken transcript."
        else:
            html = await asyncio.to_thread(fetch_url_content, url)
            title, text = extract_main_text(html)
            condensed = reduce_text_length(text, max_chars=3000)
            source_note = ""
        msgs = [
            {"role": "system", "content": "Summarize crisply (bullets ok) and extract key facts/figures. " + source_note},
        ]
        msgs.extend(
            build_message_user_style_system_messages(
                message,
                intent="summarize_url",
            )
        )
        msgs.append({"role": "user", "content": f"Title: {title or ''}\n\n{condensed}"})
        summary = await invoke_provider(
            "chat.openai_plain", generate_openai_messages_response, msgs
        )
        summary = apply_personality_overrides(message.author.id, intent="summarize_url", text=summary)
        return f"**{title or 'Summary'}**\n{summary}"

    status_msg, summary = await live_status_with_progress(
        message,
        action_label="Summarizing",
        emoji="📰",
        coro=_do_summarize(),
        duration_estimate=duration_estimate,
        summarizer=(lambda: f"Fetching page…\nURL: {url}\nExtracting main content…") if stream_ok else None,
        # An arbitrary page has no honest ETA until it responds, so a
        # percentage here would be invented. Use the reserved indeterminate style.
        indeterminate=True,
    )
    if summary:
        await send_or_edit_with_truncation(summary, target_msg=status_msg)
    else:
        await status_msg.edit(content="❌ Summary failed.")
