import io
import time
import logging
import re

import discord

from providers.openai_images import image_input_to_upload
from providers.veo_utils import (
    DEFAULT_VEO_MODEL,
    DEFAULT_VEO_FAST_MODEL,
    estimate_veo_cost,
    estimate_veo_runtime,
    generate_veo_video,
    get_veo_model_options,
    veo_is_available,
)
from services import usage_costs
from services.behavior_registry import invoke_provider
from services.model_preferences import preference_tier
from services.progress_cards import build_run_receipt, receipts_enabled, requester_label
from services.database_utils import (
    check_veo_limit,
    log_veo_usage,
)

logger = logging.getLogger("discord_bot")

_VIDEO_REPLY_PROMPT_MAX_CHARS = 1500


def _compose_reply_aware_video_prompt(prompt: str, ref_msg=None) -> str:
    """Fold a replied-to message's text into the video prompt. Without this,
    'make this a video' in reply to a greentext generated an unrelated clip —
    the reply content was never passed to the video provider."""
    base = (prompt or "").strip()
    reply_text = (getattr(ref_msg, "content", "") or "").strip()
    if not reply_text:
        return base
    reply_text = re.sub(r"\s+", " ", reply_text)
    if len(reply_text) > _VIDEO_REPLY_PROMPT_MAX_CHARS:
        reply_text = reply_text[:_VIDEO_REPLY_PROMPT_MAX_CHARS].rstrip() + "..."
    if base:
        return f"{base}\n\nBase the video on this content:\n{reply_text}"
    return reply_text


def _record_video_cost(model: str, seconds: int) -> None:
    """Ledger entry for a completed video generation, using the same price
    table shown to users in the confirmation dropdown."""
    try:
        cost = estimate_veo_cost(model, seconds)
        usage_costs.record(model, None, cost, label="video_generation")
    except Exception:
        logger.warning("video usage recording failed", exc_info=True)


def build_video_config_options(include_veo: bool | None = None):
    options = []
    if include_veo is None:
        include_veo = veo_is_available()
    if include_veo:
        for option in get_veo_model_options():
            options.append(
                {
                    **option,
                    "value": f"veo|{option['model']}|{option['seconds']}",
                    "label": f"{option['provider_label']} - {option['seconds']}s (${option['cost']:.2f})",
                    "description": f"{option['provider_label']}, {option['seconds']} seconds",
                    "default": option["model"] == DEFAULT_VEO_MODEL and option["seconds"] == 8,
                }
            )
    return options


def get_video_config_by_value(value: str, include_veo: bool | None = None):
    for option in build_video_config_options(include_veo=include_veo):
        if option["value"] == value:
            return option
    return None


def _video_progress_summary(provider_label: str, progress_data: dict) -> str:
    pct = int(float(progress_data.get("progress", 0.0)) * 100)
    status = str(progress_data.get("status") or "Processing")
    return f"{provider_label}: {status} ({pct}%)"


def _build_video_cost_message(prompt: str, include_veo: bool) -> str:
    lines = [
        "**Video Generation**",
        f"Prompt: *{prompt[:100]}...*",
        "",
        "⚠️ **Select Configuration:**",
    ]
    if include_veo:
        lines.extend(
            [
                "",
                "Veo estimates (Google 720p pricing; native audio is included):",
                "• **Veo 3.1**: 4s $1.60 | 6s $2.40 | 8s $3.20",
                "• **Veo 3.1 Fast**: 4s $0.40 | 6s $0.60 | 8s $0.80",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Video generation is currently unavailable. Please try again later.",
            ]
        )

    lines.extend(
        [
            "",
            "💖 *Help cover API costs:* <https://ko-fi.com/sardistic/goal?g=32>",
        ]
    )
    return "\n".join(lines)


async def _resolve_video_reference_upload(message, image_urls=None):
    if message.attachments:
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                try:
                    image_data = await att.read()
                    logger.info("Received image attachment for video generation: %s (%s bytes)", att.filename, len(image_data))
                    return image_data, (att.filename or "input"), (att.content_type or "image/png")
                except Exception as e:
                    logger.error("Failed to download attachment: %s", e)

    for idx, image_input in enumerate(image_urls or [], start=1):
        upload = await image_input_to_upload(image_input, fallback_name=f"input_{idx}")
        if upload:
            image_data, filename, content_type = upload
            logger.info("Resolved video reference image from collected inputs: %s (%s bytes)", filename, len(image_data))
            return image_data, filename, content_type

    return None


class VideoConfigSelect(discord.ui.Select):
    def __init__(self, video_options):
        options = [
            discord.SelectOption(
                label=option["label"],
                value=option["value"],
                description=option["description"],
                emoji=option["emoji"],
                default=option.get("default", False),
            )
            for option in video_options
        ]
        super().__init__(placeholder="Select Configuration...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.view.value = self.values[0]
        self.view.stop()


class VideoConfirmationView(discord.ui.View):
    def __init__(self, author_id, video_options):
        super().__init__(timeout=60)
        self.author_id = author_id
        self.value = None
        self.add_item(VideoConfigSelect(video_options))

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


async def handle_generate_video_intent(message, prompt: str, user_id, live_status_with_progress, stream_ok: bool, image_urls=None, ref_msg=None):
    image_data = None
    image_filename = None
    image_content_type = None
    base_fail_msg = "Generation failed."

    # The text the video is actually generated from: user instruction + the
    # replied-to message's content (so "make this a video" animates that post).
    # remix/edit detection below stays on the raw instruction, not this.
    effective_prompt = _compose_reply_aware_video_prompt(prompt, ref_msg)

    lower_prompt = prompt.lower()
    if "remix" in lower_prompt:
        await message.reply("Video remixing and editing are no longer available. You can generate a new video from a prompt or reference image with Veo.")
        return

    include_veo = veo_is_available()
    if not include_veo:
        await message.reply("Video generation is currently unavailable. Please try again later.")
        return
    video_options = build_video_config_options(include_veo=True)
    from services.server_policy import video_config_allowed
    guild_id = getattr(getattr(message, "guild", None), "id", None)
    video_options = [option for option in video_options if video_config_allowed(option, guild_id)]
    if not video_options:
        await message.reply("No video configurations are available under this server's current maximum video tier.")
        return
    reference_upload = await _resolve_video_reference_upload(message, image_urls=image_urls)
    if reference_upload:
        image_data, image_filename, image_content_type = reference_upload
        base_fail_msg = "Image-to-Video failed."
    if not image_data and "edit" in lower_prompt and "video" in lower_prompt:
        await message.reply("Video editing is no longer available. You can generate a new video from a prompt or reference image with Veo.")
        return

    tier = preference_tier(user_id, "video")
    preferred = {
        "economy": f"veo|{DEFAULT_VEO_FAST_MODEL}|4",
        "balanced": f"veo|{DEFAULT_VEO_FAST_MODEL}|8",
        "quality": f"veo|{DEFAULT_VEO_MODEL}|8",
    }.get(tier)
    if preferred:
        for option in video_options:
            option["default"] = option["value"] == preferred
    cost_msg = _build_video_cost_message(effective_prompt, include_veo=True)
    view = VideoConfirmationView(author_id=user_id, video_options=video_options)
    confirm_msg = await message.reply(cost_msg, view=view)
    await view.wait()

    if not view.value or view.value == "cancel":
        if view.value is None:
            try:
                await confirm_msg.edit(content="❌ Timed out.", view=None)
            except Exception:
                pass
        return

    selected_config = get_video_config_by_value(view.value, include_veo=include_veo)
    if not selected_config:
        await confirm_msg.edit(content="❌ This video configuration is no longer available. Start a new video request to choose Veo.", view=None)
        return
    if not video_config_allowed(selected_config, guild_id):
        await confirm_msg.edit(content="This configuration exceeds the server's current maximum video tier.", view=None)
        return

    provider = selected_config["provider"]
    provider_label = selected_config["provider_label"]
    selected_model = selected_config["model"]
    selected_seconds = int(selected_config["seconds"])
    estimated_cost = float(selected_config["cost"])
    logger.info(
        "Video configuration selected: provider=%s label=%s model=%s seconds=%s has_reference=%s",
        provider,
        provider_label,
        selected_model,
        selected_seconds,
        bool(image_data),
    )

    if not check_veo_limit(str(user_id), limit=2, window_seconds=3600):
        await confirm_msg.edit(content="⏳ You have reached the limit of 2 Veo videos per hour. Please try again later.", view=None)
        return

    try:
        await confirm_msg.edit(content=f"✅ **Queued:** {provider_label} ({selected_seconds}s)", view=None)
    except Exception:
        pass

    progress_data = {"progress": 0.0, "status": "Queued"}

    async def _generate_video_task():
        content, err = await invoke_provider(
            "video.veo.generate",
            generate_veo_video,
            effective_prompt,
            model=selected_model,
            seconds=selected_seconds,
            image_data=image_data,
            image_content_type=image_content_type,
            generate_audio=False,
            progress_state=progress_data,
        )
        if err:
            return None, f"Veo generation failed: {err}"
        if not content:
            return None, "Failed to download Veo video content."

        log_veo_usage(str(user_id), video_id=f"{selected_model}:{selected_seconds}")
        _record_video_cost(selected_model, selected_seconds)
        return io.BytesIO(content), None

    duration_estimate = estimate_veo_runtime(selected_model, selected_seconds)
    started_at = time.monotonic()
    status_msg, result = await live_status_with_progress(
        message,
        action_label=f"Generating ({provider_label}, {selected_seconds}s)",
        emoji="🎥",
        coro=_generate_video_task(),
        duration_estimate=duration_estimate,
        summarizer=(lambda: _video_progress_summary(provider_label, progress_data)) if stream_ok else None,
        progress_tracker=progress_data,
        existing_status_msg=confirm_msg,
    )

    if result and isinstance(result, tuple):
        file_obj, err = result
        if file_obj:
            final_msg = (
                f"**Video generated** ({provider_label}, {selected_seconds}s)\n"
                f"Est. Cost: ${estimated_cost:.2f} | Support: <https://ko-fi.com/sardistic/goal?g=32>\n"
                f"Prompt: {effective_prompt[:100]}..."
            )
            filename = "veo_video.mp4"
            await status_msg.reply(file=discord.File(file_obj, filename=filename))
            receipt = None
            try:
                if receipts_enabled("video"):
                    receipt = build_run_receipt(
                        "Video generated",
                        elapsed=time.monotonic() - started_at,
                        model=f"{provider_label} · {selected_seconds}s",
                        requested_by=requester_label(message),
                        extra_rows=[("est. cost", f"${estimated_cost:.2f}")],
                    )
            except Exception:
                logger.debug("receipt card skipped", exc_info=True)
            if receipt:
                await status_msg.edit(
                    content=final_msg,
                    attachments=[discord.File(receipt, "receipt.png")],
                )
            else:
                await status_msg.edit(content=final_msg)
            return

        await status_msg.edit(content=f"❌ {err or base_fail_msg}")
        return

    await status_msg.edit(content="❌ Unknown error during generation.")
