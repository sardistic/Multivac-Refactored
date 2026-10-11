"""Self-service model defaults; callers can change only their own account."""

import asyncio

import discord
from discord import app_commands
from discord.ext import commands

from services.model_preferences import WORKLOADS, get_preferences, set_preferences


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
        saved = await asyncio.to_thread(get_preferences, interaction.user.id)
        lines = ["**Your model defaults**", *[f"{key.title()}: {saved.get(key, 'auto').title()}" for key in WORKLOADS]]
        lines.extend([
            "Economy: Luna for text/vision, GPT Image Mini at low quality for images, Sora 2 at 4 seconds for video.",
            "Balanced: Sol for text/vision, Sunburst at medium quality, Sora 2 at 8 seconds.",
            "Quality: Astra for text/vision, Sunburst at high quality, Sora 2 Pro at 8 seconds.",
            "Auto restores the bot's defaults. Explicit provider requests take priority. Video still requires confirmation.",
            "Gemini retries require your confirmation. Saved tiers stay selected during automatic retries.",
        ])
        await interaction.followup.send("\n".join(lines), ephemeral=True)
