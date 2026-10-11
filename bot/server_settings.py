"""Discord server managers control defaults, ceilings and stricter limits."""

import discord
from discord import app_commands
from discord.ext import commands

from services import server_policy


def can_manage_server(member):
    permissions = getattr(member, "guild_permissions", None)
    return bool(permissions and (permissions.manage_guild or permissions.administrator))


def settings_text(guild_id):
    settings = server_policy.get_policy(guild_id)
    lines = ["**Server settings**", f"Server and per-member server limits: **{settings['throttle_percent']}%** of normal."]
    for key in server_policy.WORKLOADS:
        lines.append(f"{key.title()}: default **{settings['defaults'].get(key, 'auto')}**, maximum **{settings['ceilings'].get(key, 'quality')}**")
    lines.append("Personal choices apply within the server maximum. Auto clears an override. Base personal/global limits and provider capacity still apply.")
    return "\n".join(lines)


class ServerSettingsCog(commands.Cog, name="ServerSettings"):
    @app_commands.command(name="server_settings", description="Manage this server's model defaults, maximum tiers, and stricter limits.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.choices(
        workload=[app_commands.Choice(name=key.title(), value=key) for key in ("all", *server_policy.WORKLOADS)],
        tier=[app_commands.Choice(name=key.title(), value=key) for key in server_policy.TIERS],
        max_tier=[app_commands.Choice(name=key.title(), value=key) for key in server_policy.TIERS],
    )
    @app_commands.describe(throttle_percent="1–100% of normal server and member-in-server limits; 100 restores normal.",
                           channel_enabled="Allow media app actions in this channel again, or disable them.",
                           channel="Channel to change; defaults to the current channel.")
    async def settings(self, interaction: discord.Interaction, workload: str = "all", tier: str | None = None,
                       max_tier: str | None = None, throttle_percent: app_commands.Range[int, 1, 100] | None = None,
                       channel_enabled: bool | None = None, channel: discord.TextChannel | None = None):
        await interaction.response.defer(ephemeral=True)
        if interaction.guild is None or not can_manage_server(interaction.user):
            await interaction.followup.send("You need Manage Server or Administrator permission to change server settings.", ephemeral=True)
            return
        try:
            target = channel or interaction.channel
            if target is None or getattr(getattr(target, "guild", None), "id", None) != interaction.guild.id:
                raise ValueError("Choose a channel in this server.")
            server_policy.update_policy(interaction.guild.id, workload=workload, tier=tier,
                                        max_tier=max_tier, throttle_percent=throttle_percent)
            if channel_enabled is not None:
                server_policy.set_channel_enabled(interaction.guild.id, target.id, channel_enabled)
            text = settings_text(interaction.guild.id)
            if channel_enabled is not None:
                text += f"\nMedia app actions {'enabled in previously active' if channel_enabled else 'disabled in'} channel <#{target.id}>."
            await interaction.followup.send(text, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
