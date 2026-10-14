"""Persistent shop navigation and deterministic recovery from shop notices."""

import re
import discord

SHOP_CUSTOM_ID = "multivac:open_shop:v1"


def is_shop_message(message, bot_user_id):
    if message is None or bot_user_id is None or getattr(getattr(message, "author", None), "id", None) != bot_user_id:
        return False
    for row in getattr(message, "components", ()):
        if any(getattr(item, "label", None) == "Open shop" for item in row.children):
            return True
    return any(getattr(embed, "title", None) == "Support MultiVAC" for embed in getattr(message, "embeds", ()))


def shop_request(prompt, reference=None, bot_user_id=None):
    text = re.sub(r"[.!?]+$", "", prompt.strip().lower()).strip()
    if re.fullmatch(r"/?(?:shop|premium)(?:\s+(?:please|plz))?", text):
        return True
    if re.fullmatch(r"(?:the|your) shop|shop link|link to (?:the|your) shop", text):
        return True
    if re.fullmatch(r"where (?:can i (?:access|find|open)|is|do i (?:find|access)) (?:the|your) shop", text):
        return True
    if re.fullmatch(r"(?:(?:can|could) you )?(?:open|show|link|send)(?: me)?(?: to)? (?:the |your )?shop(?: (?:please|plz))?", text):
        return True
    return is_shop_message(reference, bot_user_id) and bool(re.fullmatch(
        r"(?:(?:can|could) you )?(?:link|send|show)(?: me)? (?:this|that|it)(?: again)?(?: (?:please|plz))?", text))


class ThrottleShopView(discord.ui.View):
    def __init__(self, bot):
        super().__init__(timeout=None)
        self.bot = bot

    @discord.ui.button(label="Open shop", style=discord.ButtonStyle.primary, custom_id=SHOP_CUSTOM_ID)
    async def open_shop(self, interaction, button):
        cog = self.bot.get_cog("Premium")
        if cog is None:
            await interaction.response.send_message("The shop is temporarily unavailable. Try /shop or /premium shortly.", ephemeral=True)
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
    text += " Use /shop or /premium any time."
    return {"content": text, "view": ThrottleShopView(bot)}
