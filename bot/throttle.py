"""Throttle notices with requester-bound access to the existing premium shop."""

import discord


class ThrottleShopView(discord.ui.View):
    def __init__(self, bot, user_id):
        super().__init__(timeout=300)
        self.bot, self.user_id = bot, int(user_id)

    async def interaction_check(self, interaction):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("Only the requester can open this shop button. Use /premium to browse your own shop.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Open shop", style=discord.ButtonStyle.primary)
    async def open_shop(self, interaction, button):
        cog = self.bot.get_cog("Premium")
        if cog is None:
            await interaction.response.send_message("The shop is temporarily unavailable. Try /premium shortly.", ephemeral=True)
            return
        await cog.shop.callback(cog, interaction)


def throttle_notice(bot, user_id, decision, text):
    """Keep the retry notice and explain the subscription's actual scope."""
    cog = bot.get_cog("Premium") if hasattr(bot, "get_cog") else None
    if cog is None:
        return {"content": text, "view": None}
    if not cog.service.config.unthrottled:
        text += "\nYou can also browse the shop for available purchases."
    elif getattr(decision, "scope", "") == "global":
        text += "\nBrowse the shop for fUnThrottled ($4.99/month), which bypasses bot-wide caps. Personal and server limits still apply."
    else:
        text += "\nYou can also browse the shop. fUnThrottled ($4.99/month) bypasses bot-wide caps only; it does not remove personal or server caps."
    return {"content": text, "view": ThrottleShopView(bot, user_id)}
