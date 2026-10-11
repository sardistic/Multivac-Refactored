"""Request-scoped permission to send a failed request to another provider."""

import contextlib
import logging
import time

import discord

logger = logging.getLogger("discord_bot")


class FallbackConsentView(discord.ui.View):
    def __init__(self, author_id: int, provider: str, *, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.author_id = author_id
        self.provider = provider
        self.accepted = False
        self.decided = False
        self.deadline = time.monotonic() + timeout
        self.message = None
        self.confirm.label = f"Retry with {provider}"

    async def interaction_check(self, interaction):
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Only the requester can choose this retry.", ephemeral=True)
            return False
        return True

    async def _choose(self, interaction, accepted):
        if not await self.interaction_check(interaction):
            return
        if self.decided:
            await interaction.response.send_message("This retry prompt has ended.", ephemeral=True)
            return
        if time.monotonic() >= self.deadline:
            self.decided = True
            try:
                await interaction.response.edit_message(
                    content="Retry prompt expired. No fallback request was sent.", view=None,
                )
            finally:
                self.stop()
            return
        self.decided = True
        self.accepted = accepted
        try:
            await interaction.response.edit_message(
                content=f"Retry with {self.provider} approved." if accepted else "Cancelled. No fallback request was sent.",
                view=None,
            )
        finally:
            self.stop()

    @discord.ui.button(label="Retry", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction, button):
        await self._choose(interaction, True)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction, button):
        await self._choose(interaction, False)

    async def on_timeout(self):
        if self.decided:
            return
        self.decided = True
        self.accepted = False
        if self.message is not None:
            with contextlib.suppress(Exception):
                await self.message.edit(content="Retry prompt expired. No fallback request was sent.", view=None)


async def confirm_provider_fallback(message, *, provider: str = "Gemini", model: str, reason: str) -> bool:
    from services.server_policy import capped, workload_for_intent
    from services.usage_costs import get_request_context
    context = get_request_context()
    guild_id = getattr(getattr(message, "guild", None), "id", None) or context.get("guild_id")
    if capped(workload_for_intent(context.get("intent")), guild_id):
        await message.reply("This server's maximum model tier blocks provider retries for this workload. Retry the configured model later or ask a server manager to change the maximum.")
        return False
    author_id = getattr(getattr(message, "author", None), "id", None)
    if author_id is None:
        return False
    view = FallbackConsentView(author_id, provider)
    try:
        view.message = await message.reply(
            f"{reason} Retry this request with **{model}** ({provider})? "
            "This sends the request and its attachments to that provider and may use additional API credits.",
            view=view,
        )
        await view.wait()
        return view.accepted
    except Exception as exc:
        logger.warning("Provider retry confirmation unavailable (%s)", type(exc).__name__)
        return False
    finally:
        view.stop()
