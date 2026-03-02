import asyncio
import contextlib
import logging
from typing import List, Optional

import discord

from database_utils import get_message_expansion, save_message_expansion, set_message_expanded
from progress import start_progress_bar

LINE_TRUNCATE_AT = 2
EXPAND_EMOJI = "🧾"
COLLAPSE_EMOJI = "🔼"

logger = logging.getLogger("discord_bot")


def make_preview(full_text: str, max_lines: int = LINE_TRUNCATE_AT):
    """
    Generate a 2-line preview.
    If this is a Gemini code-execution response, skip the thinking/result
    quote blocks to find the actual summary text for the preview.
    """
    lines = full_text.splitlines()

    if "> 🐍 **Thinking (Code Execution)**" in full_text:
        summary_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                continue

            summary_lines.append(line)
            if len(summary_lines) >= max_lines:
                break

        if summary_lines:
            preview = "\n".join(summary_lines).rstrip()
            return preview + "… (Summary)", True

    if len(lines) > max_lines:
        preview = "\n".join(lines[:max_lines]).rstrip()
        code_fence_count = preview.count("```")
        if code_fence_count % 2 != 0:
            preview += "\n```"

        return preview + "…", True
    return full_text, False


async def auto_collapse_task(message: discord.Message, delay: float = 600.0):
    await asyncio.sleep(delay)

    try:
        rec = get_message_expansion(message.id)
        if not rec or not rec["expanded"]:
            return

        full_text = rec["full_text"]
        preview, _ = make_preview(full_text, LINE_TRUNCATE_AT)
        footer = f"\n\n(react {EXPAND_EMOJI} to expand)"
        await message.edit(content=f"{preview}{footer}")
        set_message_expanded(message.id, False)

        with contextlib.suppress(Exception):
            await message.clear_reaction(COLLAPSE_EMOJI)
        with contextlib.suppress(Exception):
            await message.add_reaction(EXPAND_EMOJI)
    except Exception as e:
        logger.warning(f"Auto-collapse task failed for msg {message.id}: {e}")


async def handle_expansion_reaction(msg: discord.Message, emoji: str, rec, member=None):
    if emoji == EXPAND_EMOJI and not rec["expanded"]:
        full = rec["full_text"]
        footer = f"\n\n(react {COLLAPSE_EMOJI} to collapse)"

        if len(full) + len(footer) > 2000:
            import io

            try:
                f = io.BytesIO(full.encode("utf-8"))
                await msg.reply(
                    "⚠️ Response too long to expand inline. Sending as file.",
                    file=discord.File(f, filename="response.md"),
                )
                if member:
                    with contextlib.suppress(Exception):
                        await msg.remove_reaction(emoji, member)
                set_message_expanded(msg.id, True)
            except Exception as e:
                logger.error(f"Failed to send long response file: {e}")
            return

        with contextlib.suppress(Exception):
            await msg.edit(content=f"{full}{footer}")
        set_message_expanded(msg.id, True)
        with contextlib.suppress(Exception):
            await msg.clear_reaction(EXPAND_EMOJI)
        with contextlib.suppress(Exception):
            await msg.add_reaction(COLLAPSE_EMOJI)
        return

    if emoji == COLLAPSE_EMOJI and rec["expanded"]:
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


async def send_or_edit_with_truncation(
    full_text: str,
    *,
    channel: Optional[discord.abc.Messageable] = None,
    target_msg: Optional[discord.Message] = None,
    reply_to: Optional[discord.Message] = None,
    extra_files: Optional[List[discord.File]] = None,
    original_message: Optional[discord.Message] = None,
    model: Optional[str] = None,
    auto_index: bool = True,
    index_callback=None,
):
    if not isinstance(full_text, str):
        full_text = str(full_text)

    preview, did_trunc = make_preview(full_text, LINE_TRUNCATE_AT)

    if did_trunc:
        footer_expand = f"\n\n(react {EXPAND_EMOJI} to expand)"
        footer_collapse = f"\n\n(react {COLLAPSE_EMOJI} to collapse)"

        if len(full_text) + len(footer_collapse) <= 2000:
            content = f"{full_text}{footer_collapse}"

            if target_msg:
                sent = target_msg
                await target_msg.edit(content=content)
            else:
                sent = await channel.send(content, reference=reply_to, files=extra_files)

            save_message_expansion(sent.id, full_text, expanded=True)

            with contextlib.suppress(Exception):
                await sent.clear_reactions()
            with contextlib.suppress(Exception):
                await sent.add_reaction(COLLAPSE_EMOJI)

            asyncio.create_task(auto_collapse_task(sent, delay=600))

            if target_msg and extra_files:
                with contextlib.suppress(Exception):
                    await target_msg.reply(files=extra_files)
        else:
            content = f"{preview}{footer_expand}"

            if target_msg:
                sent = target_msg
                await target_msg.edit(content=content)
                if extra_files:
                    with contextlib.suppress(Exception):
                        await target_msg.reply(files=extra_files)
            else:
                sent = await channel.send(content, reference=reply_to, files=extra_files)

            save_message_expansion(sent.id, full_text, expanded=False)

            with contextlib.suppress(Exception):
                await sent.clear_reactions()
            with contextlib.suppress(Exception):
                await sent.add_reaction(EXPAND_EMOJI)

        if auto_index and index_callback:
            await index_callback(sent, full_text, original_message=original_message, reply_to=reply_to, model=model)

        return sent

    if target_msg:
        if extra_files:
            try:
                await target_msg.edit(content=full_text)
                await target_msg.reply(files=extra_files)
            except Exception:
                await channel.send(full_text, reference=reply_to, files=extra_files)
        else:
            await target_msg.edit(content=full_text)

        with contextlib.suppress(Exception):
            await target_msg.clear_reactions()
        save_message_expansion(target_msg.id, full_text, expanded=True)
        final_msg = target_msg
    else:
        final_msg = await channel.send(full_text, reference=reply_to, files=extra_files)

    if auto_index and final_msg and index_callback:
        await index_callback(final_msg, full_text, original_message=original_message, reply_to=reply_to, model=model)

    return final_msg


async def live_status_with_progress(
    message: discord.Message,
    *,
    action_label: str,
    emoji: str,
    coro,
    duration_estimate: int,
    summarizer=None,
    progress_tracker: dict = None,
    stream_ok: bool = False,
    editor_factory=None,
):
    status_msg = await message.reply(f"[{emoji} {action_label} ░░░░░░░░░░]")

    loop = asyncio.get_event_loop()
    task = loop.create_task(coro)
    progress_task = loop.create_task(
        start_progress_bar(
            status_msg,
            task,
            action_label=action_label,
            emoji=emoji,
            duration_estimate=duration_estimate,
            progress_tracker=progress_tracker,
        )
    )

    stop_summary = asyncio.Event()
    summary_task = None

    async def _summary_loop():
        if not stream_ok or summarizer is None:
            return
        editor = editor_factory(status_msg) if editor_factory else None
        while not task.done():
            try:
                s = summarizer()
                if s:
                    content = f"[{emoji} {action_label} ░░░░░░░░░░]\n{s}"
                    if editor:
                        await editor.update(content)
                    else:
                        await status_msg.edit(content=content)
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
