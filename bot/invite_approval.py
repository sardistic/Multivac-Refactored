"""Opt-in server admission. Existing installations are seeded by the operator."""

import asyncio
import logging
import os

from discord.ext import commands, tasks

from services.portal_store import PortalStore, admission_allowed
from services.portal_usage import publish_reports

logger = logging.getLogger(__name__)


class InviteApprovalCog(commands.Cog):
    def __init__(self, bot, path):
        self.bot = bot
        self.store = PortalStore(path)

    async def cog_load(self):
        self.previous_interaction_check = self.bot.tree.interaction_check
        self.bot.tree.interaction_check = self.interaction_check
        self.reconcile.start()

    async def cog_unload(self):
        self.reconcile.cancel()
        self.bot.tree.interaction_check = self.previous_interaction_check

    async def interaction_check(self, interaction):
        if not await asyncio.to_thread(admission_allowed, self.store.path, interaction.guild_id):
            await interaction.response.send_message('This server needs approved MultiVAC access. Visit the bot website to request an invite.', ephemeral=True)
            return False
        return await self.previous_interaction_check(interaction)

    async def enforce(self, guild):
        # Unavailable private state never grants access. Retry reconciliation if
        # the DB cannot be read rather than removing approved servers on I/O errors.
        try:
            allowed = await asyncio.to_thread(self.store.allowed, guild.id)
        except Exception:
            logger.exception('Invite approval state unavailable')
            return
        if not allowed:
            await guild.leave()
            logger.info('Left a server without approved invite access')

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        await self.enforce(guild)

    @tasks.loop(seconds=60)
    async def reconcile(self):
        for guild in list(self.bot.guilds):
            try:
                await self.enforce(guild)
            except Exception:
                logger.exception('Could not enforce server admission')
        report_path = os.getenv('PORTAL_REPORT_PATH', '')
        if report_path:
            try:
                await asyncio.to_thread(publish_reports, os.environ['USAGE_DB_PATH'], report_path, os.getenv('USAGE_TZ', 'America/New_York'))
            except Exception:
                logger.exception('Could not publish private portal aggregates')

    @reconcile.before_loop
    async def before_reconcile(self):
        await self.bot.wait_until_ready()


async def install(bot):
    path = os.getenv('INVITE_APPROVAL_DB_PATH', '').strip()
    if path and bot.get_cog('InviteApprovalCog') is None:
        await bot.add_cog(InviteApprovalCog(bot, path))
