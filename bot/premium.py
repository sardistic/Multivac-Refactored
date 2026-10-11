"""Native Discord premium purchase, status, setup, and redemption commands."""

from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from services.premium import PremiumError, PremiumService, SKUConfig
from services.security_limits import DEFAULT_RULES


logger = logging.getLogger("discord_bot")


class RedemptionView(discord.ui.View):
    def __init__(self, client, service: PremiumService, user_id: int, kind: str):
        super().__init__(timeout=60)
        self.client, self.service, self.user_id, self.kind = client, service, user_id, kind
        self._started = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requesting user can redeem this purchase.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Redeem one purchase", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self._started:
            await interaction.response.send_message("This redemption is already being processed.", ephemeral=True)
            return
        self._started = True
        await interaction.response.defer()
        try:
            result = await self.service.redeem(self.client, self.user_id, self.kind)
            if self.kind == "user":
                text = f"Personal pass redeemed for <t:{int(result['grant_start'])}:f> through <t:{int(result['grant_end'])}:f>. Use /premium_status to see your active bonuses."
            elif self.kind == "reset":
                text = "Your personal throttle counters were reset. Shared guild/global limits and provider capacity still apply."
            else:
                text = "Thank you for supporting MultiVAC! Your donation was recorded. It does not change your throttle limits."
        except PremiumError as exc:
            text = str(exc)
        except Exception as exc:
            logger.warning("Premium redemption could not be confirmed (%s)", type(exc).__name__)
            text = "Redemption could not be confirmed. Try /premium_redeem again; the same purchase can be recovered."
        await interaction.edit_original_response(content=text, view=None)
        self.stop()


class PremiumCog(commands.Cog, name="Premium"):
    def __init__(self, bot: commands.Bot, service: PremiumService):
        self.bot, self.service = bot, service
        self._poll_task: asyncio.Task | None = None

    async def _refresh(self):
        try:
            await asyncio.wait_for(self.service.refresh(self.bot), timeout=30)
            return True
        except Exception as exc:
            logger.warning("Premium verification unavailable (%s)", type(exc).__name__)
            return False

    async def cog_load(self):
        await self._refresh()
        self._poll_task = asyncio.create_task(self._poll(), name="multivac-premium-entitlements")

    async def cog_unload(self):
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _poll(self):
        while True:
            await asyncio.sleep(120)
            if self.bot.is_ready():
                await self._refresh()

    @commands.Cog.listener()
    async def on_entitlement_create(self, entitlement: discord.Entitlement):
        self.service.observe(entitlement, self.bot.application_id)

    @commands.Cog.listener()
    async def on_entitlement_update(self, entitlement: discord.Entitlement):
        self.service.observe(entitlement, self.bot.application_id)

    @commands.Cog.listener()
    async def on_entitlement_delete(self, entitlement: discord.Entitlement):
        self.service.observe(entitlement, self.bot.application_id, deleted=True)

    @app_commands.command(name="premium", description="Subscribe to fUnThrottled or support MultiVAC with a donation.")
    async def shop(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        verified = await self._refresh()
        embed = discord.Embed(title="Support MultiVAC", color=discord.Color.blurple())
        products = {
            "unthrottled": ("fUnThrottled — $4.99/month", "Auto-renewing user subscription. Bypass MultiVAC's bot-wide throttle while active, across servers and DMs. Personal limits, guild limits, and provider capacity still apply. No redemption needed."),
            "donate": ("Donate — $2.99", "One-time support purchase with no throttle benefit. Record it with /premium_redeem donate."),
            "donate_more": ("Donate More — $9.99", "One-time support purchase with no throttle benefit. Record it with /premium_redeem donate_more."),
            "guild": ("Guild Bonus", f"{self.service.guild_boost}× guild and member limits while the guild subscription is active."),
            "user": ("Personal Pass", f"Redeem with /premium_redeem user for 30 days of {self.service.user_boost}× personal limits."),
            "reset": ("Personal Throttle Reset", "Use /premium_redeem reset to clear your personal counters once."),
        }
        for key, (name, description) in products.items():
            if getattr(self.service.config, key):
                embed.add_field(name=name, value=description, inline=False)
        embed.set_footer(text="Discord checkout shows the final localized price and billing terms.")
        view = discord.ui.View(timeout=300)
        for sku_id in self.service.config.ids:
            sku = self.service.catalog.get(sku_id)
            if verified and sku is not None and sku.flags.available:
                view.add_item(discord.ui.Button(style=discord.ButtonStyle.premium, sku_id=sku_id))
        if not view.children:
            embed.description = "Purchases are not currently available. Please contact the operator."
        await interaction.followup.send(embed=embed, view=view if view.children else None, ephemeral=True)

    @app_commands.command(name="premium_status", description="See your active bonuses and effective personal/guild limits.")
    async def status(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        verified = await self._refresh()
        benefits = self.service.benefits(interaction.user.id, interaction.guild_id or "DM")
        lines = [f"Personal limits: **{benefits.user_multiplier}×**.",
                 f"This guild's bonus: **{'active' if benefits.guild_active else 'inactive'}**."]
        lines.insert(0, f"fUnThrottled: **{'active — bot-wide throttle bypassed' if benefits.bypass_global_throttle else 'inactive'}**.")
        if benefits.subscription_until:
            lines.append(f"Current subscription access ends: <t:{int(benefits.subscription_until)}:f>.")
        if benefits.user_pass_until:
            lines.append(f"Personal pass expires: <t:{int(benefits.user_pass_until)}:f>.")
        if not verified:
            lines.append("Discord verification is temporarily unavailable; recently verified bonuses may remain active for up to five minutes.")
        for action, rule in DEFAULT_RULES.items():
            lines.append(f"{action}: {rule.user * benefits.user_multiplier} personal / {rule.guild * benefits.guild_multiplier} guild per {rule.window_seconds}s.")
        lines.append("Personal/guild limits and provider capacity apply. Only active fUnThrottled subscribers bypass bot-wide throttle counters. Donations do not change limits.")
        await interaction.followup.send("\n".join(lines), ephemeral=True)

    @app_commands.command(name="premium_redeem", description="Record a donation or redeem a configured consumable purchase.")
    @app_commands.choices(product=[
        app_commands.Choice(name="30-day personal pass", value="user"),
        app_commands.Choice(name="Personal throttle reset", value="reset"),
        app_commands.Choice(name="Donate — support only", value="donate"),
        app_commands.Choice(name="Donate More — support only", value="donate_more"),
    ])
    async def redeem(self, interaction: discord.Interaction, product: str):
        if product not in {"user", "reset", "donate", "donate_more"} or not getattr(self.service.config, product, 0):
            await interaction.response.send_message("Choose a configured consumable. fUnThrottled subscriptions apply automatically.", ephemeral=True)
            return
        if product == "user":
            text = f"Redeem one personal pass for {self.service.user_boost}× personal limits for 30 days? An existing pass is extended by 30 days."
        elif product == "reset":
            text = "Redeem one reset to clear your personal request, URL, image, compute, and code counters now? It leaves guild/global counters and provider capacity unchanged."
        else:
            text = "Record one support purchase? Thank you! This purchase grants no throttle bonus or reset."
        await interaction.response.send_message(
            text, view=RedemptionView(self.bot, self.service, interaction.user.id, product), ephemeral=True
        )

    async def _owner(self, interaction: discord.Interaction) -> bool:
        if await self.bot.is_owner(interaction.user):
            return True
        await interaction.response.send_message("Only the application owner can configure premium products.", ephemeral=True)
        return False

    @app_commands.command(name="premium_skus", description="Owner: list this application's Discord SKUs and their IDs.")
    async def skus(self, interaction: discord.Interaction):
        if not await self._owner(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            catalog = await self.bot.fetch_skus()
            lines = [f"`{sku.id}` — {discord.utils.escape_markdown(sku.name)} ({sku.type.name}; available: {sku.flags.available})" for sku in catalog]
            await interaction.followup.send("\n".join(lines)[:1900] or "No SKUs found. Create the three products in the Developer Portal first.", ephemeral=True)
        except Exception as exc:
            logger.warning("Premium SKU listing failed (%s)", type(exc).__name__)
            await interaction.followup.send("Discord SKU lookup is unavailable. Try again later.", ephemeral=True)

    @app_commands.command(name="premium_configure", description="Owner: validate and save premium SKU IDs; omitted products are disabled.")
    async def configure(self, interaction: discord.Interaction, guild_sku_id: str = "0", user_sku_id: str = "0", reset_sku_id: str = "0", unthrottled_sku_id: str = "0", donate_sku_id: str = "0", donate_more_sku_id: str = "0"):
        if not await self._owner(interaction):
            return
        await interaction.response.defer(ephemeral=True)
        try:
            ids = (guild_sku_id, user_sku_id, reset_sku_id, unthrottled_sku_id, donate_sku_id, donate_more_sku_id)
            if not all(value.isascii() and value.isdigit() and 0 <= int(value) < 2**64 for value in ids):
                raise PremiumError("Provide valid Discord SKU IDs; use 0 for disabled products.")
            await self.service.configure(self.bot, SKUConfig(*(int(value) for value in ids)))
            refreshed = await self._refresh()
            text = "Premium SKU mappings saved and verified. Configure each product's price and availability in Discord."
            if not refreshed:
                text += " Entitlement synchronization is pending; try /premium_status shortly."
            await interaction.followup.send(text, ephemeral=True)
        except PremiumError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            logger.warning("Premium configuration failed (%s)", type(exc).__name__)
            await interaction.followup.send("Discord could not verify these products. No new SKU mapping was confirmed.", ephemeral=True)
