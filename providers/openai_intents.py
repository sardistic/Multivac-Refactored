from __future__ import annotations

import logging
import re

from providers.openai_client import get_openai_client
from providers.openai_messages import OpenAIModerationError

_INTENT_SYSTEM = (
    "You are a fast, lightweight intent classifier.\n"
    "Classify a user's message into one of:\n"
    "- 'edit_image'\n"
    "- 'generate_image'\n"
    "- 'summarize_url'\n"
    "- 'describe_image'\n"
    "- 'get_weather'\n"
    "- 'get_stock'\n"
    "- 'gemini_chat'\n"
    "- 'claude_chat'\n"
    "- 'generate_video'\n"
    "- 'chat'\n\n"
    "Rules:\n"
    "- If message requests a VIDEO, MOVIE, or CLIP -> 'generate_video'.\n"
    "- If message starts with \"imagine\", \"generate\", \"draw\", \"create\", \"paint\" AND is NOT about video -> 'generate_image'.\n"
    "- If user says \"transparent background\" -> 'generate_image'.\n"
    "- If replying to an image and mentions \"change\", \"edit\", \"make transparent\", \"fix\" -> 'edit_image'.\n"
    "- Weather words (forecast, rain, snow, temperature) -> 'get_weather'.\n"
    "- If a URL is present and they want a summary -> 'summarize_url'.\n"
    "- If they ask to describe an image -> 'describe_image'.\n"
    "- Stock words or 'stock <TICKER>' -> 'get_stock'.\n"
    "- If message starts with 'gemini' and is NOT image generation/editing -> 'gemini_chat'.\n"
    "- If message starts with 'claude' or user explicitly asks for 'Claude' -> 'claude_chat'.\n"
    "- Else -> 'chat'.\n\n"
    "IMPORTANT: Output ONLY ONE label."
)


async def classify_intent(text: str, has_images: bool = False) -> str:
    try:
        if not (text or "").strip():
            return "chat"

        if has_images:
            system_prompt = (
                _INTENT_SYSTEM
                + "\n\n"
                + "CRITICAL: The user has attached one or more IMAGES with their message.\n"
                + "When images are present, assume the user's request is ABOUT those images unless explicitly stated otherwise.\n\n"
                + "Choose the intent:\n"
                + "- 'generate_image' = User wants to CREATE a NEW image from scratch (imagine, generate, draw, paint, create)\n\n"
                + "IMPORTANT: If the user says 'edit', 'change', 'make', 'transform' -> 'edit_image' (even if it involves text).\n"
                + "Only use 'describe_image' if they specifically ask what is in the image, or to transcribe/translate text WITHOUT modifying the image.\n"
                + "Only use 'chat' if the message is clearly NOT about the attached images."
            )
        else:
            system_prompt = _INTENT_SYSTEM

        resp = await get_openai_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text.strip()},
            ],
        )
        label = (resp.choices[0].message.content or "").strip().lower()
        label = re.sub(r"[^a-z_]", "", label)
        return label if label in {
            "edit_image", "generate_image", "summarize_url",
            "describe_image", "get_weather", "get_stock", "gemini_chat", "claude_chat", "chat"
        } else "chat"
    except Exception as e:
        if isinstance(e, OpenAIModerationError):
            raise
        logging.warning("[intent] fallback to chat due to: %s", e)
        return "chat"
