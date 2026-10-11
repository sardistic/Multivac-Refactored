import asyncio
import io
import time
import logging
import re

import discord

from providers.openai_images import image_input_to_upload
from providers.sora_utils import create_sora_job, download_sora_content, get_sora_status, remix_sora_video
from providers.veo_utils import (
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
    check_sora_limit,
    check_veo_limit,
    get_last_sora_video_id,
    log_sora_usage,
    log_veo_usage,
)

logger = logging.getLogger("discord_bot")

_VIDEO_REPLY_PROMPT_MAX_CHARS = 1500


def _compose_reply_aware_video_prompt(prompt: str, ref_msg=None) -> str:
    """Fold a replied-to message's text into the video prompt. Without this,
    'make this a video' in reply to a greentext generated an unrelated clip —
    the reply content was never passed to Sora/Veo."""
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


def _record_video_cost(provider: str, model: str, seconds: int) -> None:
    """Ledger entry for a completed video generation, using the same price
    table shown to users in the confirmation dropdown."""
    try:
        cost = 0.0
        if provider == "sora":
            for o in SORA_VIDEO_OPTIONS:
                if o["model"] == model and o["seconds"] == seconds:
                    cost = o["cost"]
                    break
        else:
            cost = estimate_veo_cost(model, seconds)
        usage_costs.record(model, None, cost, label="video_generation")
    except Exception:
        logger.warning("video usage recording failed", exc_info=True)

SORA_VIDEO_OPTIONS = [
    {
        "provider": "sora",
        "provider_label": "Sora 2 Pro",
        "model": "sora-2-pro",
        "seconds": 4,
        "cost": 1.20,
        "emoji": "✨",
        "description": "Sora 2 Pro, 4 seconds",
        "default": False,
    },
    {
        "provider": "sora",
        "provider_label": "Sora 2 Pro",
        "model": "sora-2-pro",
        "seconds": 8,
        "cost": 2.40,
        "emoji": "✨",
        "description": "Sora 2 Pro, 8 seconds",
        "default": True,
    },
    {
        "provider": "sora",
        "provider_label": "Sora 2 Pro",
        "model": "sora-2-pro",
        "seconds": 12,
        "cost": 3.60,
        "emoji": "✨",
        "description": "Sora 2 Pro, 12 seconds",
        "default": False,
    },
    {
        "provider": "sora",
        "provider_label": "Sora 2",
        "model": "sora-2",
        "seconds": 4,
        "cost": 0.40,
        "emoji": "🎞️",
        "description": "Sora 2, 4 seconds",
        "default": False,
    },
    {
        "provider": "sora",
        "provider_label": "Sora 2",
        "model": "sora-2",
        "seconds": 8,
        "cost": 0.80,
        "emoji": "🎞️",
        "description": "Sora 2, 8 seconds",
        "default": False,
    },
    {
        "provider": "sora",
        "provider_label": "Sora 2",
        "model": "sora-2",
        "seconds": 12,
        "cost": 1.20,
        "emoji": "🎞️",
        "description": "Sora 2, 12 seconds",
        "default": False,
    },
]


def build_video_config_options(include_veo: bool | None = None):
    options = []
    for option in SORA_VIDEO_OPTIONS:
        options.append(
            {
                **option,
                "value": f"sora|{option['model']}|{option['seconds']}",
                "label": f"{option['provider_label']} - {option['seconds']}s (720p ${option['cost']:.2f})",
            }
        )

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
                    "default": False,
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
        "Sora estimates (OpenAI 720p pricing):",
        "• **Sora 2 Pro**: 4s $1.20 | 8s $2.40 | 12s $3.60",
        "• **Sora 2**: 4s $0.40 | 8s $0.80 | 12s $1.20",
        "• Sora 2 Pro image references may auto-select 1024p at $0.50/second.",
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
                "Veo is currently unavailable because `GEMINI_API_KEY` is not configured for this runtime.",
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

    reference_upload = await _resolve_video_reference_upload(message, image_urls=image_urls)
    if reference_upload:
        image_data, image_filename, image_content_type = reference_upload
        base_fail_msg = "Image-to-Video failed."

    include_veo = veo_is_available()
    cost_msg = _build_video_cost_message(effective_prompt, include_veo=include_veo)
    video_options = build_video_config_options(include_veo=include_veo)
    from services.server_policy import video_config_allowed
    guild_id = getattr(getattr(message, "guild", None), "id", None)
    video_options = [option for option in video_options if video_config_allowed(option, guild_id)]
    tier = preference_tier(user_id, "video")
    preferred = {"economy": "sora|sora-2|4", "balanced": "sora|sora-2|8", "quality": "sora|sora-2-pro|8"}.get(tier)
    if preferred:
        for option in video_options:
            option["default"] = option["value"] == preferred
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
        await confirm_msg.edit(content="❌ Unknown video configuration selected.", view=None)
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

    if provider == "sora":
        if not check_sora_limit(str(user_id), limit=2, window_seconds=3600):
            await confirm_msg.edit(content="⏳ You have reached the limit of 2 Sora videos per hour. Please try again later.", view=None)
            return
    else:
        if not check_veo_limit(str(user_id), limit=2, window_seconds=3600):
            await confirm_msg.edit(content="⏳ You have reached the limit of 2 Veo videos per hour. Please try again later.", view=None)
            return

    is_remix = False
    remix_target_id = None
    lower_prompt = prompt.lower()
    if provider == "sora":
        if "remix" in lower_prompt or (not image_data and "edit" in lower_prompt and "video" in lower_prompt):
            last_vid = get_last_sora_video_id(str(user_id))
            if last_vid:
                remix_target_id = last_vid
                is_remix = True
                base_fail_msg = "Remix failed."
            elif "remix" in lower_prompt:
                await confirm_msg.edit(
                    content="⚠️ I couldn't find a previous Sora video of yours to remix. Please generate one first!",
                    view=None,
                )
                return
    elif "remix" in lower_prompt or (not image_data and "edit" in lower_prompt and "video" in lower_prompt):
        await confirm_msg.edit(
            content="⚠️ Veo is wired up for prompt-to-video and image-to-video here. Remix is still Sora-only for now.",
            view=None,
        )
        return

    try:
        await confirm_msg.edit(content=f"✅ **Queued:** {provider_label} ({selected_seconds}s)", view=None)
    except Exception:
        pass

    progress_data = {"progress": 0.0, "status": "Queued"}

    async def _generate_video_task():
        if provider == "sora":
            if is_remix and remix_target_id:
                job = await invoke_provider(
                    "video.sora.remix", remix_sora_video, remix_target_id, effective_prompt
                )
            else:
                job = await invoke_provider(
                    "video.sora.create",
                    create_sora_job,
                    effective_prompt,
                    model=selected_model,
                    size=None if image_data else "1280x720",
                    seconds=selected_seconds,
                    image_data=image_data,
                    image_filename=image_filename,
                    image_content_type=image_content_type,
                )

            if not job.get("ok"):
                return None, f"Failed to start job: {job.get('error')}"

            video_id = job["data"].get("id")
            progress_data["status"] = "Processing"
            logger.info(
                "Sora Job Started: %s (Model=%s, Sec=%s, Remix=%s)",
                video_id,
                selected_model,
                selected_seconds,
                is_remix,
            )

            start_time = asyncio.get_event_loop().time()
            while True:
                await asyncio.sleep(4)
                if asyncio.get_event_loop().time() - start_time > 600:
                    return None, "Timeout waiting for video generation."

                status_res = await invoke_provider(
                    "video.sora.status", get_sora_status, video_id
                )
                if not status_res.get("ok"):
                    logger.warning("Poll check failed: %s", status_res.get("error"))
                    continue

                status_data = status_res["data"]
                status = status_data.get("status")

                if "progress" in status_data:
                    try:
                        raw_p = str(status_data["progress"]).strip().replace("%", "")
                        p_val = float(raw_p)
                        if p_val > 1.0:
                            p_val /= 100.0
                        progress_data["progress"] = p_val
                        logger.debug("Sora Poll: %.1f%% (Raw: %s)", p_val * 100, status_data["progress"])
                    except Exception as e:
                        logger.warning("Failed to parse progress: %s - %s", status_data["progress"], e)

                if status == "completed":
                    progress_data["progress"] = 1.0
                    progress_data["status"] = "Downloading video"
                    break
                if status == "failed":
                    err_msg = status_data.get("error", {}).get("message", "Unknown error")
                    return None, f"Video generation failed: {err_msg}"

            content = await invoke_provider(
                "video.sora.download", download_sora_content, video_id
            )
            if not content:
                return None, "Failed to download video content."

            f = io.BytesIO(content)
            log_sora_usage(str(user_id), video_id=video_id)
            _record_video_cost("sora", selected_model, selected_seconds)
            return f, None

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
        _record_video_cost("veo", selected_model, selected_seconds)
        return io.BytesIO(content), None

    duration_estimate = selected_seconds * 10 if provider == "sora" else estimate_veo_runtime(selected_model, selected_seconds)
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
            filename = "sora_video.mp4" if provider == "sora" else "veo_video.mp4"
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
