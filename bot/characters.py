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

logger = logging.getLogger("discord_bot")


def character_caption(row):
    data = json.loads(row["character_json"])
    name = discord.utils.escape_markdown(data["name"])
    source = f"\nCharacter source: <{data['url']}>" if data.get("url") else ""
    return f"**{name}** — your GFL2-inspired 3D character portrait.{source}\nOne saved PNG image per purchase. Use /character to download it again."


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
        if entitlement.application_id == self.bot.application_id and entitlement.sku_id == self.service.sku_id:
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
                    async for ent in self.bot.entitlements(limit=None, skus=[discord.Object(self.service.sku_id)], exclude_ended=True):
                        row = self.service.store.row(ent.id)
                        # Failed API calls require an explicit retry, preventing runaway costs.
                        if row is None or row["state"] == "pending" or (row["state"] == "ready" and not row["delivery_attempted"]):
                            if self.service.valid(ent, self.bot.application_id, ent.user_id):
                                self._queue(ent)
            except Exception as exc:
                logger.warning("Character purchase reconciliation unavailable (%s)", type(exc).__name__)
            await asyncio.sleep(120)

    @app_commands.command(name="character", description="Get your purchased character image, or choose a name before buying.")
    @app_commands.describe(character_name="Before buying: any character/person's name, or 'random' for a random character.")
    async def character(self, interaction: discord.Interaction, character_name: str | None = None):
        await interaction.response.defer(ephemeral=True)
        try:
            ent = await self.service.purchase(self.bot, interaction.user.id)
            if character_name is not None:
                if ent and self.service.store.row(ent.id):
                    raise PremiumError("Your purchase already has a fixed selection. Download it without specifying a new name.")
                self.service.store.choice(interaction.user.id, None if character_name.casefold().strip() == "random" else character_name)
            if ent:
                row = await self.service.render(self.bot, ent.id, interaction.user.id)
                await self.service.checked(self.bot, ent.id, interaction.user.id)
                await interaction.followup.send(character_caption(row), file=discord.File(io.BytesIO(row["image"]), "character.png"), ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
                self.service.store.delivery_attempt(ent.id)
            else:
                product = await self.service.product(self.bot)
                view = discord.ui.View(timeout=300)
                if product.flags.available:
                    view.add_item(discord.ui.Button(style=discord.ButtonStyle.premium, sku_id=product.id))
                text = ("**GFL2-inspired character portrait — $0.99 once**\n"
                        "One rendered PNG image per account. Choose any name with /character character_name:NAME "
                        "before buying, or leave it random. The image is delivered by DM after purchase; "
                        "/character recovers it if DMs are closed. Later downloads reuse the same image. "
                        "No throttle benefit. Discord checkout shows the final price.")
                if not product.flags.available:
                    text += "\nThis product is not available for checkout yet."
                await interaction.followup.send(text, view=view if view.children else None, ephemeral=True)
        except PremiumError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            logger.warning("Character request unavailable (%s)", type(exc).__name__)
            await interaction.followup.send("Character service is temporarily unavailable. Your purchase is retained; retry /character.", ephemeral=True)
