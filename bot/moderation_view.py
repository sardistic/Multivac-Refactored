from __future__ import annotations

import discord
from providers.openai_client import OPENAI_CHAT_MODEL


class ModerationFallbackView(discord.ui.View):
    def __init__(self, author_id, retry_callback):
        super().__init__(timeout=120)
        self.author_id = author_id
        self.retry_callback = retry_callback

        options = [
            discord.SelectOption(
                label="Gemini 1.5 Pro (Smarter)",
                value="gemini-1.5-pro",
                description="Higher reasoning, might be less strict.",
                emoji="🧠",
            ),
            discord.SelectOption(
                label="Gemini 1.5 Flash (Fast)",
                value="gemini-1.5-flash",
                description="Fast and efficient.",
                emoji="⚡",
            ),
            discord.SelectOption(
                label="Gemini 1.5 Pro 002",
                value="gemini-1.5-pro-002",
                description="Updated Pro model.",
                emoji="🆕",
            ),
            discord.SelectOption(
                label=f"{OPENAI_CHAT_MODEL} (OpenAI)",
                value=OPENAI_CHAT_MODEL,
                description="Switch provider to OpenAI.",
                emoji="🟢",
            ),
        ]

        select = discord.ui.Select(
            placeholder="Select an alternative model...",
            min_values=1,
            max_values=1,
            options=options,
            custom_id="moderation_model_select",
        )
        select.callback = self.select_callback
        self.add_item(select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("Not your request! make your own.", ephemeral=True)
            return False
        return True

    async def select_callback(self, interaction: discord.Interaction):
        model = interaction.data["values"][0]
        await interaction.response.edit_message(content=f"🔄 **Retrying with {model}...**", view=None)
        self.stop()
        await self.retry_callback(model_name=model)
