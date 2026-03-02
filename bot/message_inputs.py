from __future__ import annotations

import base64
import logging
import mimetypes
import re
from typing import Any, Dict, List

from google.genai import types

logger = logging.getLogger("discord_bot")


def strip_mention_and_trigger(raw: str, bot_user_id: int | None) -> str:
    s = raw
    if bot_user_id:
        s = re.sub(f"<@!?{bot_user_id}>", "", s).strip()
    return s


def looks_like_search(s: str) -> bool:
    s = s.lower().strip()
    return (
        s.startswith("search ")
        or s.startswith("look up ")
        or s.startswith("lookup ")
        or s.startswith("news ")
        or " search " in f" {s} "
        or " news " in f" {s} "
        or s in {"search", "news"}
    )


def extract_search_query(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(search|lookup|look up|news)\s*[:,-]*\s*", "", s, flags=re.I)
    return s or "latest"


def has_google_search(google_api_key: str | None, google_cse_id: str | None, env: Dict[str, str]) -> bool:
    ga = google_api_key or env.get("GOOGLE_API_KEY")
    gc = google_cse_id or env.get("GOOGLE_CSE_ID")
    ok = bool(ga and gc)
    if not ok:
        logger.debug("Google search disabled: GOOGLE_API_KEY=%s, GOOGLE_CSE_ID=%s", bool(ga), bool(gc))
    return ok


async def resolve_reference_message(message, bot_user):
    is_reply_to_bot = False
    ref_msg = None
    if message.reference:
        try:
            ref_msg = message.reference.resolved or await message.channel.fetch_message(message.reference.message_id)
        except Exception:
            ref_msg = None
        if ref_msg and bot_user and ref_msg.author.id == bot_user.id:
            is_reply_to_bot = True
    return ref_msg, is_reply_to_bot


async def collect_image_inputs(message, ref_msg, image_url_to_base64) -> List[str]:
    image_urls: List[str] = []

    if ref_msg and ref_msg.attachments:
        for attachment in ref_msg.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                b64 = await image_url_to_base64(attachment.url)
                if b64:
                    image_urls.append(b64)

    if ref_msg and ref_msg.embeds:
        for embed in ref_msg.embeds:
            if embed.image and embed.image.url:
                b64 = await image_url_to_base64(embed.image.url)
                if b64:
                    image_urls.append(b64)

    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                b64 = await image_url_to_base64(attachment.url)
                if b64:
                    image_urls.append(b64)

    matches = re.findall(
        r"https?://[^\s]+(?:\.(?:png|jpg|jpeg|webp|gif)|cdn\.discordapp\.com/attachments/[^\s]+)",
        message.content,
    )
    for raw_url in matches:
        if "cdn.discordapp.com" in raw_url:
            b64 = await image_url_to_base64(raw_url)
            if b64:
                image_urls.append(b64)
        else:
            image_urls.append(raw_url)

    seen = set()
    unique_urls = []
    for url in image_urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
    return unique_urls


async def collect_gemini_parts(message, ref_msg, image_urls) -> List[Any]:
    gemini_parts = []
    text_exts = (
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".csv",
        ".c",
        ".cpp",
        ".h",
        ".java",
        ".go",
        ".rs",
        ".sql",
        ".yaml",
        ".yml",
        ".html",
        ".css",
    )

    async def _append_attachment_parts(attachment, label: str, prefix: str) -> None:
        try:
            data = await attachment.read()
            mime = attachment.content_type or mimetypes.guess_type(attachment.filename)[0] or "application/octet-stream"
            if mime.startswith("image/"):
                gemini_parts.append(types.Part.from_bytes(data=data, mime_type=mime))
                logger.info("Added %s image part: %s", label, attachment.filename)
                return
            if mime.startswith("text/") or "/json" in mime or attachment.filename.lower().endswith(text_exts):
                try:
                    content = data.decode("utf-8")
                except UnicodeDecodeError:
                    content = data.decode("latin-1")
                if len(content) > 150_000:
                    content = content[:150_000] + "\n... [TRUNCATED] ..."
                gemini_parts.append(types.Part(text=f"--- {prefix}: {attachment.filename} ---\n{content}\n"))
                logger.info("Added %s text part: %s", label, attachment.filename)
        except Exception as e:
            logger.error("Failed to process %s attachment: %s", label, e)

    if ref_msg and ref_msg.attachments:
        for attachment in ref_msg.attachments:
            await _append_attachment_parts(attachment, "replied", "REPLIED FILE")

    if message.attachments:
        for attachment in message.attachments:
            await _append_attachment_parts(attachment, "current", "FILE")

    for url in image_urls:
        if not url.startswith("data:image/"):
            continue
        try:
            header, encoded = url.split(",", 1)
            mime = header.split(":", 1)[1].split(";", 1)[0]
            gemini_parts.append(types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime))
        except Exception as e:
            logger.error("Failed to decode data URI: %s", e)

    return gemini_parts
