"""Purchase-triggered delivery and self-service recovery of paid character images."""

import asyncio
import io
import json
import logging
import time

import discord
from discord import app_commands
from discord.ext import commands

from services.character_renders import get_character_service
from services.premium import PremiumError
from services import server_policy
from services.security_limits import check_rate_limit

logger = logging.getLogger("discord_bot")


class CharacterChoiceModal(discord.ui.Modal, title="Choose your character before buying"):
    subject = discord.ui.TextInput(label="Character name, or random", max_length=120, default="random")

    def __init__(self, cog, user_id):
        super().__init__()
        self.cog, self.user_id = cog, user_id

    async def on_submit(self, interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requester can choose this character.", ephemeral=True)
            return
        await self.cog.handle_interaction(interaction, str(self.subject))


class CharacterOfferView(discord.ui.View):
    def __init__(self, cog, user_id, product):
        super().__init__(timeout=300)
        self.cog, self.user_id = cog, user_id
        if product.flags.available:
            self.add_item(discord.ui.Button(style=discord.ButtonStyle.premium, sku_id=product.id))

    @discord.ui.button(label="Choose character", style=discord.ButtonStyle.secondary)
    async def choose(self, interaction, button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Use /character to choose your own subject.", ephemeral=True)
            return
        await interaction.response.send_modal(CharacterChoiceModal(self.cog, self.user_id))


def character_caption(row):
    data = json.loads(row["character_json"])
    name = discord.utils.escape_markdown(data["name"])
    source = f"\nCharacter source: <{data['url']}>" if data.get("url") else ""
    return (f"**{name}** — your GFL2-inspired 3D character portrait.{source}\n"
            f"One saved PNG per purchase. Download this image with /character purchase_id:{row['entitlement_id']}.")


class CharacterCog(commands.Cog, name="Characters"):
    def __init__(self, bot, service=None):
        self.bot, self.service = bot, service or get_character_service()
        self._tasks = set()
        self._queued = set()
        self._poll_task = None

    async def cog_load(self):
        self._poll_task = asyncio.create_task(self._poll())

    async def cog_unload(self):
        tasks = [*self._tasks, *([self._poll_task] if self._poll_task else [])]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(self, entitlement):
        try:
            row = await self.service.render(self.bot, entitlement.id, entitlement.user_id)
            if not self.service.store.delivery_attempt(entitlement.id):
                return
            user = self.bot.get_user(entitlement.user_id) or await self.bot.fetch_user(entitlement.user_id)
            await self.service.checked(self.bot, entitlement.id, entitlement.user_id)
            await user.send(character_caption(row), file=discord.File(io.BytesIO(row["image"]), "character.png"), allowed_mentions=discord.AllowedMentions.none())
        except Exception as exc:
            logger.warning("Character delivery deferred (%s); /character can recover the purchase", type(exc).__name__)

    def _queue(self, entitlement):
        if entitlement.id in self._queued or len(self._tasks) >= 8:
            return
        existing = self.service.store.row(entitlement.id)
        if existing and existing["state"] in {"failed", "revoked"}:
            return
        self.service.store.reserve(entitlement)
        self._queued.add(entitlement.id)
        task = asyncio.create_task(self._deliver(entitlement))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda completed: self._queued.discard(entitlement.id))

    @commands.Cog.listener()
    async def on_entitlement_create(self, entitlement):
        if self.service.valid(entitlement, self.bot.application_id, entitlement.user_id):
            self._queue(entitlement)

    @commands.Cog.listener()
    async def on_entitlement_delete(self, entitlement):
        if entitlement.application_id == self.bot.application_id and entitlement.sku_id in self.service.sku_ids:
            self.service.store.revoke(entitlement.id)

    @commands.Cog.listener()
    async def on_entitlement_update(self, entitlement):
        if entitlement.deleted or (entitlement.ends_at and entitlement.ends_at.timestamp() <= time.time()):
            await self.on_entitlement_delete(entitlement)

    async def _poll(self):
        while True:
            try:
                if self.bot.is_ready() and self.service.sku_id:
                    await self.service.product(self.bot)
                    async for ent in self.bot.entitlements(limit=None, skus=[discord.Object(value) for value in self.service.sku_ids], exclude_ended=True):
                        row = self.service.store.row(ent.id)
                        # Failed API calls require an explicit retry, preventing runaway costs.
                        if row is None or row["state"] == "pending" or (row["state"] == "ready" and not row["delivery_attempted"]):
                            if self.service.valid(ent, self.bot.application_id, ent.user_id):
                                self._queue(ent)
            except Exception as exc:
                logger.warning("Character purchase reconciliation unavailable (%s)", type(exc).__name__)
            await asyncio.sleep(120)

    @app_commands.command(name="character", description="Get your purchased character image, or choose a name before buying.")
    @app_commands.describe(character_name="Before buying: any character/person's name, or 'random' for a random character.",
                           purchase_id="Optional purchase ID from a saved image to recover that particular image.")
    async def character(self, interaction: discord.Interaction, character_name: str | None = None, purchase_id: str | None = None):
        await self.handle_interaction(interaction, character_name, purchase_id)

    async def handle_interaction(self, interaction, character_name=None, purchase_id=None):
        await interaction.response.defer(ephemeral=True)
        try:
            guild = server_policy.interaction_guild(interaction)
        except PremiumError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        kwargs = {'purchase_id':purchase_id} if purchase_id is not None else {}
        await self.respond(interaction.user.id, guild, getattr(interaction, "channel", None), interaction.followup.send, character_name, private=True, **kwargs)

    async def respond(self, user_id, guild, channel, send, character_name=None, *, private=False, purchase_id=None):
        flags = {"ephemeral": True} if private else {}
        try:
            await server_policy.require_active_channel(self.bot, guild, channel)
            quota = check_rate_limit("request", user_id=user_id, guild_id=guild.id if guild else "DM")
            if not quota.allowed:
                raise PremiumError(f"Request limits are busy. Retry in {quota.retry_after}s; your purchase is retained.")
            if purchase_id is not None:
                if not purchase_id.isascii() or not purchase_id.isdecimal() or not 0 < int(purchase_id) < 2**64:
                    raise PremiumError("Use the purchase ID shown below your saved character image.")
                ent = await self.service.checked(self.bot, int(purchase_id), user_id)
            else:
                ent = await self.service.purchase(self.bot, user_id)
            existing = self.service.store.row(ent.id) if ent else None
            if not (existing and existing["state"] == "ready") and server_policy.get_policy(guild.id if guild else "DM")["ceilings"].get("image") == "economy":
                raise PremiumError("This server's image maximum excludes Sunburst. Use /character in a DM for this product.")
            if character_name is not None:
                normalized = None if character_name.casefold().strip() == "random" else character_name.strip()
                if existing and existing["requested_name"] != normalized and (existing['state']!='ready' or purchase_id is not None):
                    raise PremiumError("Your purchase already has a fixed selection. Download it without specifying a new name.")
                self.service.store.choice(user_id, normalized)
                if existing and existing['state']=='ready' and purchase_id is None:
                    # A completed purchase stays fixed; the new choice is for the
                    # next checkout, not a reroll of the saved entitlement.
                    ent = None
            if ent:
                async def before_render():
                    await server_policy.require_active_channel(self.bot, guild, channel)
                row = await self.service.render(self.bot, ent.id, user_id, guild_id=guild.id if guild else "DM", before_render=before_render)
                await self.service.checked(self.bot, ent.id, user_id)
                await server_policy.require_active_channel(self.bot, guild, channel)
                view = None
                if server_policy.get_policy(guild.id if guild else 'DM')['ceilings'].get('image') != 'economy':
                    try:
                        product = await self.service.product(self.bot)
                        view = CharacterOfferView(self, user_id, product)
                    except (PremiumError, discord.HTTPException, OSError, asyncio.TimeoutError):
                        pass  # Saved downloads do not depend on the new offer.
                await send(character_caption(row), file=discord.File(io.BytesIO(row["image"]), "character.png"), view=view,
                           allowed_mentions=discord.AllowedMentions.none(), **flags)
                self.service.store.delivery_attempt(ent.id)
            else:
                if server_policy.get_policy(guild.id if guild else 'DM')['ceilings'].get('image') == 'economy':
                    raise PremiumError("This server's image maximum excludes Sunburst. Use /character in a DM for this product.")
                product = await self.service.product(self.bot)
                view = CharacterOfferView(self, user_id, product)
                text = ("**GFL2-inspired character portrait — $0.99 per purchase**\n"
                        "Each purchase delivers one rendered PNG. Choose any name with /character character_name:NAME "
                        "before buying, or leave it random. The image is delivered by DM after purchase; "
                        "/character recovers it if DMs are closed. Later downloads reuse the saved image; "
                        "buy again for another image. Earlier one-time purchases stay recoverable. "
                        "No throttle benefit. Discord checkout shows the final price.")
                if not product.flags.available:
                    text += "\nThis product is not available for checkout yet."
                await send(text, view=view, allowed_mentions=discord.AllowedMentions.none(), **flags)
        except PremiumError as exc:
            await send(str(exc), **flags)
        except Exception as exc:
            logger.warning("Character request unavailable (%s)", type(exc).__name__)
            await send("Character service is temporarily unavailable. Your purchase is retained; retry /character.", **flags)
