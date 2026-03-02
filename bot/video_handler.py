import asyncio
import io
import logging

import discord

from services.database_utils import check_sora_limit, get_last_sora_video_id, log_sora_usage
from providers.sora_utils import create_sora_job, download_sora_content, get_sora_status, remix_sora_video

logger = logging.getLogger("discord_bot")


class SoraConfigSelect(discord.ui.Select):
    def __init__(self):
        options = [
            discord.SelectOption(label="Pro - 4s ($1.20)", value="sora-2-pro|4", description="Sora 2 Pro, 4 seconds", emoji="✨"),
            discord.SelectOption(label="Pro - 8s ($2.40)", value="sora-2-pro|8", description="Sora 2 Pro, 8 seconds (Default)", emoji="✨", default=True),
            discord.SelectOption(label="Pro - 12s ($3.60)", value="sora-2-pro|12", description="Sora 2 Pro, 12 seconds", emoji="✨"),
            discord.SelectOption(label="Std - 4s ($0.40)", value="sora-2|4", description="Sora 2, 4 seconds", emoji="🎞️"),
            discord.SelectOption(label="Std - 8s ($0.80)", value="sora-2|8", description="Sora 2, 8 seconds", emoji="🎞️"),
            discord.SelectOption(label="Std - 12s ($1.20)", value="sora-2|12", description="Sora 2, 12 seconds", emoji="🎞️"),
        ]
        super().__init__(placeholder="Select Configuration...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        self.view.value = self.values[0]
        self.view.stop()


class SoraConfirmationView(discord.ui.View):
    def __init__(self, author_id):
        super().__init__(timeout=60)
        self.author_id = author_id
        self.value = None
        self.add_item(SoraConfigSelect())

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


async def handle_generate_video_intent(message, prompt: str, user_id, live_status_with_progress, stream_ok: bool):
    if not check_sora_limit(str(user_id), limit=2, window_seconds=3600):
        await message.reply("⏳ You have reached the limit of 2 Sora videos per hour. Please try again later.")
        return

    image_data = None
    is_remix = False
    base_fail_msg = "Generation failed."

    if message.attachments:
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                try:
                    image_data = await att.read()
                    base_fail_msg = "Image-to-Video failed."
                    logger.info(f"Received image attachment for Sora: {att.filename} ({len(image_data)} bytes)")
                    break
                except Exception as e:
                    logger.error(f"Failed to download attachment: {e}")

    remix_target_id = None
    lower_prompt = prompt.lower()
    if "remix" in lower_prompt or (not image_data and "edit" in lower_prompt and "video" in lower_prompt):
        last_vid = get_last_sora_video_id(str(user_id))
        if last_vid:
            remix_target_id = last_vid
            is_remix = True
            base_fail_msg = "Remix failed."
        elif "remix" in lower_prompt:
            await message.reply("⚠️ I couldn't find a previous video of yours to remix. Please generate one first!")
            return

    cost_msg = (
        f"**Sora Video Generation**\n"
        f"Prompt: *{prompt[:100]}...*\n\n"
        f"⚠️ **Select Configuration:**\n"
        f"Costs based on duration (4s / 8s / 12s):\n"
        f"• **Pro**: $1.20 / $2.40 / $3.60\n"
        f"• **Std**: $0.40 / $0.80 / $1.20\n\n"
        f"💖 *Help cover API costs:* <https://ko-fi.com/sardistic/goal?g=32>"
    )

    view = SoraConfirmationView(author_id=user_id)
    confirm_msg = await message.reply(cost_msg, view=view)
    await view.wait()

    if not view.value or view.value == "cancel":
        if view.value is None:
            try:
                await confirm_msg.edit(content="❌ Timed out.", view=None)
            except Exception:
                pass
        return

    selected_model, selected_seconds_text = view.value.split("|")
    selected_seconds = int(selected_seconds_text)

    try:
        await confirm_msg.edit(content=f"✅ **Queued:** {selected_model} ({selected_seconds}s)", view=None)
    except Exception:
        pass

    progress_data = {"progress": 0.0}

    async def _generate_video_task():
        if is_remix and remix_target_id:
            job = await remix_sora_video(remix_target_id, prompt)
        else:
            job = await create_sora_job(
                prompt,
                model=selected_model,
                size="1280x720",
                seconds=selected_seconds,
                image_data=image_data,
            )

        if not job.get("ok"):
            return None, f"Failed to start job: {job.get('error')}"

        video_id = job["data"].get("id")
        logger.info(f"Sora Job Started: {video_id} (Model={selected_model}, Sec={selected_seconds}, Remix={is_remix})")

        start_time = asyncio.get_event_loop().time()
        while True:
            await asyncio.sleep(4)
            if asyncio.get_event_loop().time() - start_time > 600:
                return None, "Timeout waiting for video generation."

            status_res = await get_sora_status(video_id)
            if not status_res.get("ok"):
                logger.warning(f"Poll check failed: {status_res.get('error')}")
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
                    logger.debug(f"Sora Poll: {p_val * 100:.1f}% (Raw: {status_data['progress']})")
                except Exception as e:
                    logger.warning(f"Failed to parse progress: {status_data['progress']} - {e}")

            if status == "completed":
                progress_data["progress"] = 1.0
                break
            if status == "failed":
                err_msg = status_data.get("error", {}).get("message", "Unknown error")
                return None, f"Video generation failed: {err_msg}"

        content = await download_sora_content(video_id)
        if not content:
            return None, "Failed to download video content."

        f = io.BytesIO(content)
        log_sora_usage(str(user_id), video_id=video_id)
        return f, None

    status_msg, result = await live_status_with_progress(
        confirm_msg,
        action_label=f"Generating ({selected_model}, {selected_seconds}s)",
        emoji="🎥",
        coro=_generate_video_task(),
        duration_estimate=selected_seconds * 10,
        summarizer=(lambda: f"Status: Processing ({int(progress_data['progress'] * 100)}%)") if stream_ok else None,
        progress_tracker=progress_data,
    )

    if result and isinstance(result, tuple):
        file_obj, err = result
        if file_obj:
            cost = selected_seconds * (0.30 if "pro" in selected_model else 0.10)
            final_msg = (
                f"**Video generated** ({selected_model}, {selected_seconds}s)\n"
                f"Est. Cost: ${cost:.2f} | Support: <https://ko-fi.com/sardistic/goal?g=32>\n"
                f"Prompt: {prompt[:100]}..."
            )
            await status_msg.reply(file=discord.File(file_obj, filename="sora_video.mp4"))
            await status_msg.edit(content=final_msg)
            return

        await status_msg.edit(content=f"❌ {err or base_fail_msg}")
        return

    await status_msg.edit(content="❌ Unknown error during generation.")
