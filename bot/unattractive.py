"""Explicit server selection and confirmation for the timed Unattractive SKU."""

import asyncio
import logging
from functools import partial

import discord
from discord import app_commands
from discord.ext import commands

from bot.server_settings import can_manage_server
from services import server_policy
from services.premium import PremiumError
from services.security_limits import ThrottleError, check_rate_limit
from bot.throttle import throttle_notice
from services.server_effects import get_server_effect_service, NoUnusedEffectPurchase

logger = logging.getLogger('discord_bot')


class EffectActivationView(discord.ui.View):
    def __init__(self, cog, user_id, guild_id):
        super().__init__(timeout=120)
        self.cog, self.user_id, self.guild_id = cog,user_id,guild_id
        self._started = False

    async def interaction_check(self, interaction):
        if interaction.user.id != self.user_id or interaction.guild_id != self.guild_id:
            await interaction.response.send_message('Only the requester can activate this purchase in the selected server.',ephemeral=True)
            return False
        return True

    @discord.ui.button(label='Activate one purchase for this server',style=discord.ButtonStyle.primary)
    async def confirm(self, interaction, button):
        if self._started:
            await interaction.response.send_message('This activation is already being processed.',ephemeral=True)
            return
        self._started = True
        await interaction.response.defer(ephemeral=True)
        try:
            result = await self.cog.activate(interaction)
            text = (f"Unattractive activated for this server: <t:{int(result['grant_start'])}:T> "
                    f"through <t:{int(result['grant_end'])}:T>. Everyone's chat and newly generated "
                    "visuals use the unappealing style during that window. Server managers can stop it with /unattractive stop.")
        except NoUnusedEffectPurchase:
            async def edit_offer(content, **kwargs):
                await interaction.edit_original_response(content=content, **kwargs)

            await self.cog.respond(interaction.user, interaction.guild, interaction.channel, edit_offer, 'shop')
            self.stop()
            return
        except ThrottleError as exc:
            await interaction.edit_original_response(**throttle_notice(self.cog.bot, self.user_id, exc.decision, str(exc)))
            self.stop()
            return
        except PremiumError as exc:
            text = str(exc)
        except Exception as exc:
            logger.warning('Unattractive activation unavailable (%s)',type(exc).__name__)
            text = 'Activation could not be confirmed. Retry /unattractive activate in this server; a pending purchase is recoverable.'
        await interaction.edit_original_response(content=text,view=None)
        self.stop()


class UnattractiveCog(commands.Cog, name='Unattractive'):
    def __init__(self, bot, service=None):
        self.bot, self.service = bot, service or get_server_effect_service()
        self._poll_task = None

    async def _reconcile(self):
        try:
            await asyncio.wait_for(self.service.reconcile(self.bot),timeout=30)
        except Exception as exc:
            logger.warning('Unattractive purchase reconciliation unavailable (%s)',type(exc).__name__)

    async def cog_load(self):
        await self._reconcile()
        self._poll_task = asyncio.create_task(self._poll(),name='multivac-unattractive-entitlements')

    async def cog_unload(self):
        if self._poll_task:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task,return_exceptions=True)

    async def _poll(self):
        while True:
            await asyncio.sleep(120)
            if self.bot.is_ready():
                await self._reconcile()

    @commands.Cog.listener()
    async def on_entitlement_update(self, entitlement):
        self.service.observe(entitlement,self.bot.application_id)

    @commands.Cog.listener()
    async def on_entitlement_delete(self, entitlement):
        self.service.observe(entitlement,self.bot.application_id,deleted=True)

    async def activate(self, interaction):
        guild = server_policy.interaction_guild(interaction)
        if guild is None:
            raise PremiumError('Activate Unattractive inside the server you want it to affect.')

        async def guard():
            await server_policy.require_active_channel(self.bot,guild,interaction.channel)

        await guard()
        limit = check_rate_limit('request',user_id=interaction.user.id,guild_id=guild.id)
        if not limit.allowed:
            raise ThrottleError(f'Request limits are busy. Retry in {limit.retry_after}s; no purchase was spent.', limit)
        return await self.service.activate(self.bot,interaction.user.id,guild.id,before_consume=guard)

    @app_commands.command(name='unattractive',description='Buy or activate 10 minutes of unattractive chat and visuals for this server.')
    @app_commands.guild_only()
    @app_commands.choices(action=[app_commands.Choice(name=name.title(),value=name) for name in ('shop','activate','status','stop')])
    async def unattractive(self, interaction: discord.Interaction, action: str = 'shop'):
        await interaction.response.defer(ephemeral=True)
        try:
            guild = server_policy.interaction_guild(interaction)
        except PremiumError as exc:
            await interaction.followup.send(str(exc),ephemeral=True)
            return
        await self.respond(interaction.user,guild,interaction.channel,partial(interaction.followup.send,ephemeral=True),action)

    async def respond(self, user, guild, channel, send, action='shop'):
        try:
            if guild is None:
                raise PremiumError('Use this command in the server you want it to affect.')
            if action == 'stop':
                if not can_manage_server(user):
                    raise PremiumError('You need Manage Server or Administrator permission to stop this server effect.')
                self.service.store.stop(guild.id)
                await send('Unattractive stopped for this server, including queued time. Used purchases cannot be reactivated. New purchases can be activated later.')
                return
            end = self.service.active_until(guild.id)
            if action == 'status':
                await send(f'Unattractive is active for everyone in this server until <t:{int(end)}:T>.' if end else 'Unattractive is inactive in this server.')
                return
            await server_policy.require_active_channel(self.bot,guild,channel)
            if action == 'activate':
                await send("Activate one purchased Unattractive for **this server**? It adds **10 minutes** of unappealing chat and visual results for **everyone here**. Additional purchases queue another 10 minutes. Existing saved images stay unchanged. A server manager can stop the effect, including queued time.",view=EffectActivationView(self,user.id,guild.id))
                return
            if action != 'shop':
                raise PremiumError('Choose shop, activate, status, or stop.')
            sku = await self.service.product(self.bot)
            view = discord.ui.View(timeout=300)
            if sku.flags.available:
                view.add_item(discord.ui.Button(style=discord.ButtonStyle.premium,sku_id=sku.id))
            text = ('**Unattractive — $1.99 once per activation.** Everyone in this server gets deliberately unappealing chat and newly generated visuals for 10 minutes. '
                    'After buying, use /unattractive activate here and confirm. Repeated purchases add 10 minutes each. Existing quotas/models apply; saved images are unchanged. '
                    'Manage Server/Administrator members can stop active and queued time. Discord checkout shows the final localized price.')
            if not view.children:
                text += '\nThis SKU is not currently available for checkout.'
            await send(text,view=view if view.children else None)
        except PremiumError as exc:
            await send(str(exc))
        except Exception as exc:
            logger.warning('Unattractive command unavailable (%s)',type(exc).__name__)
            await send('Unattractive is temporarily unavailable. Try again later.')
