"""Requester-bound priced picker shared by chat and image app actions."""

import asyncio
import logging
import time

import discord

from services.image_options import DEFAULT_IMAGE_CONFIG, IMAGE_CONFIGS, available_image_providers
from services.model_preferences import image_choice, preference_tier
from services.server_policy import image_config_allowed, server_scope

logger = logging.getLogger("discord_bot")
IMAGE_CHOICE_SECONDS = 30


class ImageConfigSelect(discord.ui.Select):
    def __init__(self, configs, preferred):
        super().__init__(placeholder="Choose API, quality, size and output price…", options=[
            discord.SelectOption(label=f"{config.label} · {config.price_label}", value=config.value,
                                 description=f"{config.provider.title()} API; input costs are additional.",
                                 default=config == preferred)
            for config in configs
        ])

    async def callback(self, interaction):
        if not await self.view.interaction_check(interaction):
            return
        value = self.values[0]
        if value not in self.view.allowed:
            await interaction.response.send_message("That configuration is unavailable. Start a new image request.", ephemeral=True)
            return
        self.view.value = value
        self.view.stop()
        await interaction.response.defer()


class ImageOptionsView(discord.ui.View):
    def __init__(self, user_id, configs, preferred=DEFAULT_IMAGE_CONFIG):
        # A fixed outer deadline prevents interaction events from extending it.
        super().__init__(timeout=None)
        self.user_id = user_id
        self.value = None
        self.deadline = time.monotonic() + IMAGE_CHOICE_SECONDS
        self.allowed = {config.value: config for config in configs}
        self.select = ImageConfigSelect(configs, preferred)
        self.add_item(self.select)

    async def interaction_check(self, interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requester can choose this image's settings.", ephemeral=True)
            return False
        if self.is_finished() or time.monotonic() >= self.deadline:
            await interaction.response.send_message("This image choice has closed.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, row=1)
    async def cancel(self, interaction, button):
        if not await self.interaction_check(interaction):
            return
        self.value = "cancel"
        self.stop()
        await interaction.response.defer()

    async def on_error(self, interaction, error, item):
        self.value = "cancel"
        self.stop()
        logger.warning("Image selection failed (%s)", type(error).__name__)


async def choose_image_config(user_id, send, *, guild_id=None):
    available = available_image_providers()
    configs = [config for config in IMAGE_CONFIGS if available.get(config.provider) and image_config_allowed(config, guild_id)]
    if not configs:
        await send("No image configurations are available with the current provider configuration and server limits.")
        return None
    preferred = DEFAULT_IMAGE_CONFIG
    with server_scope(guild_id):
        if preference_tier(user_id, "image") != "auto":
            saved = image_choice(user_id)
            preferred = next((c for c in configs if c.model == saved.model and c.quality == saved.quality and c.size == "1024x1024"), preferred)
    default_available = DEFAULT_IMAGE_CONFIG in configs
    timeout_text = (
        "After 30 seconds without a choice: **OpenAI GPT Image Mini · low · 1024×1024 · ~$0.005 output**."
        if default_available else "The OpenAI default is unavailable. Choose within 30 seconds or this request will cancel."
    )
    text = (
        "**Choose image settings**\n"
        "Select an API, quality and size below to start, or cancel.\n"
        "Your saved image tier is highlighted where available; the timeout always uses Mini.\n"
        "Prices are estimated API output costs in USD; prompt/reference input costs are additional.\n"
        "Mini: ~$0.005–$0.052. Sunburst: variable, $0.030 per 1K output tokens.\n"
        "Gemini: 2K/4K output estimates are shown in the menu.\n" + timeout_text
    )
    view = ImageOptionsView(user_id, configs, preferred)
    try:
        reply = await send(text, view=view, allowed_mentions=discord.AllowedMentions.none())
        # The requester gets a full 30 seconds after the menu is delivered.
        view.deadline = time.monotonic() + IMAGE_CHOICE_SECONDS
        try:
            await asyncio.wait_for(view.wait(), timeout=IMAGE_CHOICE_SECONDS)
        except asyncio.TimeoutError:
            pass
        view.stop()
        timed_out = view.value is None
        config = DEFAULT_IMAGE_CONFIG if timed_out and default_available else view.allowed.get(view.value)
        if config and (not available_image_providers().get(config.provider) or not image_config_allowed(config, guild_id)):
            await reply.edit(content="These image settings are no longer available under the current provider configuration or server maximum. Start a new request.", view=None)
            return None
        if config is None:
            await reply.edit(content="Image request cancelled. No image request was sent to a provider.", view=None)
            return None
        prefix = "30 seconds elapsed; using default" if timed_out else "Selected"
        await reply.edit(content=f"{prefix}: **{config.label}** · {config.price_label}. Input costs are additional.", view=None)
        return config
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Image picker unavailable (%s)", type(exc).__name__)
        return None
    finally:
        view.stop()
