"""Self-service model defaults; callers can change only their own account."""

import asyncio

import discord
from discord import app_commands
from discord.ext import commands

from services.model_preferences import WORKLOADS, get_preferences, set_preferences, preference_tier
from services.server_policy import server_scope


def defaults_text(user_id, guild_id=None):
    saved = get_preferences(user_id)
    with server_scope(guild_id):
        lines = ["**Your model defaults**", *[f"{key.title()}: {saved.get(key, 'auto').title()} (here: {preference_tier(user_id, key).title()})" for key in WORKLOADS]]
    lines.extend([
        "Economy: Luna for text/vision, GPT Image Mini at low quality for images, Sora 2 at 4 seconds for video.",
        "Balanced: Sol for text/vision, Sunburst at medium quality, Sora 2 at 8 seconds.",
        "Quality: Astra for text/vision, Sunburst at high quality, Sora 2 Pro at 8 seconds.",
        "Auto uses server defaults, then bot defaults. Server maximums apply; explicit providers require the server to allow them. Video still requires confirmation.",
        "Gemini retries require your confirmation. Saved tiers stay selected during automatic retries.",
        "These settings change your account only. The $0.99 character product keeps Sunburst low quality.",
    ])
    return "\n".join(lines)


class ModelDefaultsView(discord.ui.View):
    def __init__(self, user_id, guild_id=None):
        super().__init__(timeout=300)
        self.user_id, self.guild_id, self.workload = user_id, guild_id, "all"

    async def interaction_check(self, interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requester can change these defaults.", ephemeral=True)
            return False
        return True

    @discord.ui.select(placeholder="Choose workload (default: all)", options=[discord.SelectOption(label=key.title(), value=key) for key in ("all", *WORKLOADS)])
    async def workload_select(self, interaction, select):
        if not await self.interaction_check(interaction):
            return
        self.workload = select.values[0]
        await interaction.response.edit_message(content=f"Editing **{self.workload}**.\n" + defaults_text(self.user_id, self.guild_id), view=self)

    @discord.ui.select(placeholder="Save a model tier", options=[discord.SelectOption(label=key.title(), value=key) for key in ("auto", "economy", "balanced", "quality")])
    async def tier_select(self, interaction, select):
        if not await self.interaction_check(interaction):
            return
        await asyncio.to_thread(set_preferences, self.user_id, self.workload, select.values[0])
        await interaction.response.edit_message(content=defaults_text(self.user_id, self.guild_id), view=self)


class ModelDefaultsCog(commands.Cog, name="ModelDefaults"):
    @app_commands.command(name="model_defaults", description="View or set your own model defaults by workload.")
    @app_commands.choices(
        workload=[app_commands.Choice(name=key.title(), value=key) for key in ("all", *WORKLOADS)],
        tier=[app_commands.Choice(name=key.title(), value=key) for key in ("auto", "economy", "balanced", "quality")],
    )
    async def defaults(self, interaction: discord.Interaction, workload: str = "all", tier: str | None = None):
        await interaction.response.defer(ephemeral=True)
        if tier is not None:
            try:
                await asyncio.to_thread(set_preferences, interaction.user.id, workload, tier)
            except ValueError as exc:
                await interaction.followup.send(str(exc), ephemeral=True)
                return
        guild_id = getattr(interaction, "guild_id", None) or getattr(getattr(interaction, "guild", None), "id", None)
        text = await asyncio.to_thread(defaults_text, interaction.user.id, guild_id)
        await interaction.followup.send(text, view=ModelDefaultsView(interaction.user.id, guild_id), ephemeral=True)
