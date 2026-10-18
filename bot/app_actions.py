"""Discord Apps menu actions, image command, and deterministic mention shortcuts."""

import asyncio
import logging
import re

import discord
from discord import app_commands
from discord.ext import commands

from bot.model_defaults import ModelDefaultsCog, defaults_text, ModelDefaultsView
from bot.image_options import choose_image_config
from bot.server_settings import can_manage_server, settings_text
from bot.throttle import shop_request, throttle_notice
from services import server_policy, usage_costs
from services.model_preferences import set_preferences
from services.premium import PremiumError
from services.security_limits import ThrottleError, check_rate_limit, provider_capacity
from services.server_effects import unattractive_action

logger = logging.getLogger("discord_bot")
_PRICE = re.compile(r"^\$(?:1(?:\.00)?|0?\.99)(?:\s+(.*))?$", re.I)


def paid_subject(prompt):
    match = _PRICE.fullmatch(prompt.strip())
    if not match:
        return False, None
    subject = re.sub(r"\s+(?:plz|please)[.!?]*$", "", (match[1] or "random").strip(), flags=re.I)
    return True, subject or "random"


class GenerateImageModal(discord.ui.Modal, title="Generate an image"):
    prompt = discord.ui.TextInput(label="Image description", style=discord.TextStyle.paragraph, max_length=1800)

    def __init__(self, cog, user_id, initial=""):
        super().__init__()
        self.cog, self.user_id = cog, user_id
        self.prompt.default = initial[:1800]

    async def on_submit(self, interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requester can submit this image.", ephemeral=True)
            return
        await self.cog.generate(interaction, str(self.prompt))


class AppActionsCog(commands.Cog, name="AppActions"):
    def __init__(self, bot):
        self.bot = bot
        self.menus = [
            app_commands.ContextMenu(name="Character portrait $0.99", callback=self.character_menu),
            app_commands.ContextMenu(name="Generate image", callback=self.image_menu),
            app_commands.ContextMenu(name="Model defaults", callback=self.defaults_menu),
            app_commands.ContextMenu(name="Character portrait $0.99", callback=self.character_profile_menu),
            app_commands.ContextMenu(name="Generate image", callback=self.image_profile_menu),
            app_commands.ContextMenu(name="Model defaults", callback=self.defaults_profile_menu),
        ]

    async def cog_load(self):
        for menu in self.menus:
            self.bot.tree.add_command(menu)

    async def cog_unload(self):
        for menu in self.menus:
            self.bot.tree.remove_command(menu.name, type=menu.type)

    async def character_menu(self, interaction: discord.Interaction, message: discord.Message):
        await self.bot.get_cog("Characters").handle_interaction(interaction)

    async def image_menu(self, interaction: discord.Interaction, message: discord.Message):
        initial = re.sub(r"<@!?\d+>", "", message.content).strip()
        await self.open_image_modal(interaction, initial)

    async def open_image_modal(self, interaction, initial=""):
        try:
            await server_policy.require_active_channel(self.bot, server_policy.interaction_guild(interaction), interaction.channel)
        except PremiumError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_modal(GenerateImageModal(self, interaction.user.id, initial))

    async def defaults_menu(self, interaction: discord.Interaction, message: discord.Message):
        await ModelDefaultsCog.defaults.callback(self.bot.get_cog("ModelDefaults"), interaction)

    async def character_profile_menu(self, interaction: discord.Interaction, user: discord.User):
        await self.bot.get_cog("Characters").handle_interaction(interaction)

    async def image_profile_menu(self, interaction: discord.Interaction, user: discord.User):
        await self.open_image_modal(interaction)

    async def defaults_profile_menu(self, interaction: discord.Interaction, user: discord.User):
        await ModelDefaultsCog.defaults.callback(self.bot.get_cog("ModelDefaults"), interaction)

    @app_commands.command(name="generate_image", description="Choose image API, quality, size and estimated cost in an active bot channel.")
    @app_commands.describe(prompt="Describe your image; this uses normal image limits, separate from the $0.99 character purchase.")
    async def image(self, interaction: discord.Interaction, prompt: app_commands.Range[str, 1, 1800]):
        await self.generate(interaction, prompt)

    async def generate(self, interaction, prompt):
        await interaction.response.defer(ephemeral=True)
        previous = usage_costs.get_request_context()
        guild_id = getattr(getattr(interaction, "guild", None), "id", None)
        try:
            guild = server_policy.interaction_guild(interaction)
            if not prompt.strip() or len(prompt) > 1800:
                raise PremiumError("Describe your image in 1–1800 characters.")
            await server_policy.require_active_channel(self.bot, guild, getattr(interaction, "channel", None))
            user_id = interaction.user.id
            request = check_rate_limit("request", user_id=user_id, guild_id=guild_id or "DM")
            if not request.allowed:
                raise ThrottleError(f"Request limits are busy. Retry in {request.retry_after}s.", request)
            usage_costs.set_request_context(user_id=str(user_id), guild_id=str(guild_id or "DM"), intent="generate_image")
            with server_policy.server_scope(guild_id):
                limit = check_rate_limit("image", user_id=user_id, guild_id=guild_id or "DM")
                if not limit.allowed:
                    raise ThrottleError(f"Image limits are busy. Retry in {limit.retry_after}s.", limit)
                async def send_picker(content, **kwargs):
                    return await interaction.followup.send(content, **kwargs, ephemeral=True, wait=True)
                choice = await choose_image_config(user_id, send_picker, guild_id=guild_id)
                if choice is None:
                    return
                async with provider_capacity(user_id):
                    await server_policy.require_active_channel(self.bot, guild, getattr(interaction, "channel", None))
                    if not server_policy.image_config_allowed(choice, guild_id):
                        raise PremiumError("These image settings exceed the server's current maximum tier. Start a new request.")
                    from providers.stability_generation import handle_image_generation
                    output = await asyncio.wait_for(handle_image_generation(None, prompt, **choice.provider_kwargs()), timeout=600)
            if output is None:
                raise PremiumError("No image was produced. Retry later; no other provider was contacted.")
            await server_policy.require_active_channel(self.bot, guild, getattr(interaction, "channel", None))
            await interaction.followup.send(f"Image generated with {choice.label}.", file=discord.File(output, "image.png"),
                                            ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
        except ThrottleError as exc:
            await interaction.followup.send(**throttle_notice(self.bot, interaction.user.id, exc.decision, str(exc)), ephemeral=True)
        except PremiumError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            logger.warning("Image app action unavailable (%s)", type(exc).__name__)
            await interaction.followup.send("Image generation is temporarily unavailable. Try again later.", ephemeral=True)
        finally:
            usage_costs.set_request_context(**previous)


async def handle_app_text(bot, message, prompt, *, reference=None):
    """Return handled only for explicit local syntax, before intent classification."""
    if shop_request(prompt, reference, getattr(getattr(bot, "user", None), "id", None)):
        cog = bot.get_cog("Premium")
        if cog is None:
            await message.reply("The shop is temporarily unavailable. Try /shop or /premium shortly.")
        else:
            async def send(**kwargs):
                await message.reply(**kwargs, allowed_mentions=discord.AllowedMentions.none())
            await cog.send_shop(message, send)
        return True
    paid, subject = paid_subject(prompt)
    lowered = prompt.strip().lower()
    defaults = re.fullmatch(r"model[_ ]defaults(?:\s+(\w+)\s+(\w+))?", lowered)
    server = re.fullmatch(r"server\s+(settings|model|cap|throttle|channel)(?:\s+(.*))?", lowered)
    effect = unattractive_action(prompt)
    if not (paid or defaults or server or effect):
        return False
    if paid:
        await bot.get_cog("Characters").respond(message.author.id, message.guild, message.channel, message.reply, subject)
        return True
    limit = check_rate_limit("request", user_id=message.author.id, guild_id=message.guild.id if message.guild else "DM")
    if not limit.allowed:
        await message.reply(**throttle_notice(bot, message.author.id, limit, f"Request limits are busy. Retry in {limit.retry_after}s."))
        return True
    try:
        if effect:
            cog = bot.get_cog("Unattractive")
            if cog is None:
                await message.reply("Unattractive is still starting up. Try again shortly.")
                return True
            await cog.respond(message.author, message.guild, message.channel, message.reply, effect)
        elif defaults:
            if defaults[1]:
                set_preferences(message.author.id, defaults[1], defaults[2])
            guild_id = message.guild.id if message.guild else None
            await message.reply(defaults_text(message.author.id, guild_id), view=ModelDefaultsView(message.author.id, guild_id), allowed_mentions=discord.AllowedMentions.none())
        else:
            if message.guild is None or not can_manage_server(message.author):
                raise ValueError("You need Manage Server or Administrator permission to change server settings.")
            action, arguments = server[1], (server[2] or "").split()
            if action in {"model", "cap"} and len(arguments) == 2:
                server_policy.update_policy(message.guild.id, workload=arguments[0], **{"tier" if action == "model" else "max_tier": arguments[1]})
            elif action == "throttle" and len(arguments) == 1:
                server_policy.update_policy(message.guild.id, throttle_percent=int(arguments[0]))
            elif action == "channel" and arguments in (["on"], ["off"]):
                server_policy.set_channel_enabled(message.guild.id, message.channel.id, arguments == ["on"])
            elif action != "settings" or arguments:
                raise ValueError("Use: server settings; server model WORKLOAD TIER; server cap WORKLOAD TIER; server throttle 1–100; server channel on/off.")
            await message.reply(settings_text(message.guild.id) + (f"\nMedia app actions in this channel: {arguments[0]}." if action == "channel" else ""), allowed_mentions=discord.AllowedMentions.none())
    except ValueError as exc:
        await message.reply(str(exc), allowed_mentions=discord.AllowedMentions.none())
    return True
