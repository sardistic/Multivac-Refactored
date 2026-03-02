from __future__ import annotations

import base64
import io
import logging
import random
import re
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from providers.gemini_utils import edit_gemini_image, generate_gemini_image, generate_gemini_with_references
from providers.stability_client import (
    STABILITY_AVAILABLE,
    STABILITY_KEY,
    generation,
    get_openai_image_client,
    stability_client,
)

logger = logging.getLogger("stability_utils")


def extract_width_height_from_prompt(prompt: str) -> tuple[int, int]:
    prompt_lower = prompt.lower()
    width = height = 1024
    if "portrait" in prompt_lower or "vertical" in prompt_lower:
        width, height = 768, 1024
    elif "landscape" in prompt_lower or "horizontal" in prompt_lower:
        width, height = 1024, 768
    match = re.search(r"(\d{3,4})\s*[xX]\s*(\d{3,4})", prompt)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    return width, height


async def generate_stability_image(image_prompt: str, width: int = 960, height: int = 768) -> Optional[BytesIO]:
    if not (STABILITY_AVAILABLE and STABILITY_KEY and stability_client and generation):
        logger.warning("generate_stability_image called but Stability is not configured.")
        return None
    try:
        api = stability_client.StabilityInference(
            key=STABILITY_KEY,
            verbose=True,
            engine="stable-diffusion-v1-5",
        )
        answers = api.generate(
            prompt=image_prompt,
            seed=random.randint(0, 2**32 - 1),
            steps=50,
            cfg_scale=11.0,
            width=width,
            height=height,
            samples=1,
            sampler=generation.SAMPLER_K_EULER_ANCESTRAL,
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
    except Exception:
        logger.exception("Error generating Stability image")
    return None


async def generate_gpt_image(prompt: str) -> Optional[BytesIO]:
    try:
        background_type = "transparent" if "transparent background" in prompt.lower() else "auto"
        result = await get_openai_image_client().images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            size="1024x1024",
            background=background_type,
            quality="high",
            moderation="low",
            n=1,
        )
        b64_image = result.data[0].b64_json if result and result.data else None
        if not b64_image:
            return None
        return BytesIO(base64.b64decode(b64_image))
    except Exception:
        logger.exception("Error generating GPT image")
        return None


async def handle_image_generation(message, prompt: str, reply_msg=None) -> Optional[BytesIO]:
    try:
        width, height = extract_width_height_from_prompt(prompt)
        if prompt.lower().startswith("stable imagine"):
            image_prompt = prompt[15:].strip()
            if STABILITY_AVAILABLE:
                img = await generate_stability_image(image_prompt, width, height)
                if img:
                    return img
            return await generate_gpt_image(image_prompt)

        if prompt.lower().startswith("gemini imagine"):
            image_prompt = prompt[14:].strip()
            ref_images = []
            headers = {"User-Agent": "Mozilla/5.0"}
            if reply_msg:
                if reply_msg.attachments:
                    for att in reply_msg.attachments:
                        if att.content_type and att.content_type.startswith("image/"):
                            try:
                                r = requests.get(att.url, headers=headers, timeout=20)
                                if r.status_code == 200:
                                    ref_images.append(BytesIO(r.content))
                            except Exception as e:
                                logger.error("Failed to download reply attachment %s: %s", att.url, e)
                if reply_msg.embeds:
                    for embed in reply_msg.embeds:
                        if embed.image and embed.image.url:
                            try:
                                r = requests.get(embed.image.url, headers=headers, timeout=20)
                                if r.status_code == 200:
                                    ref_images.append(BytesIO(r.content))
                            except Exception as e:
                                logger.error("Failed to download reply embed %s: %s", embed.image.url, e)
            if message and message.attachments:
                for att in message.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        try:
                            r = requests.get(att.url, headers=headers, timeout=20)
                            if r.status_code == 200:
                                ref_images.append(BytesIO(r.content))
                        except Exception as e:
                            logger.error("Failed to download attachment %s: %s", att.url, e)
            for url in re.findall(r"(https?://\S+\.(?:png|jpg|jpeg|webp|gif))", prompt, re.IGNORECASE):
                try:
                    r = requests.get(url, headers=headers, timeout=20)
                    if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                        ref_images.append(BytesIO(r.content))
                except Exception as e:
                    logger.error("Failed to download URL %s: %s", url, e)
            if ref_images:
                img = generate_gemini_with_references(image_prompt, ref_images)
                if img:
                    return img
            img = generate_gemini_image(image_prompt, width, height)
            if img:
                return img
            if message:
                await message.channel.send("⚠️ **Gemini generation failed** (likely rate limit or error). Falling back to OpenAI... 🧠")
            return await generate_gpt_image(image_prompt)

        return await generate_gpt_image(prompt)
    except Exception:
        logger.exception("Error in handle_image_generation")
        return None


async def edit_image_with_prompt(image_input: str | list[str], prompt: str) -> Optional[BytesIO]:
    try:
        urls = [image_input] if isinstance(image_input, str) else image_input
        if not urls:
            return None

        def decode_img(u):
            if u.startswith("data:image/"):
                _, b64 = u.split(",", 1)
                return BytesIO(base64.b64decode(b64))
            if u.startswith("http"):
                r = requests.get(u, timeout=30)
                r.raise_for_status()
                return BytesIO(r.content)
            return BytesIO(base64.b64decode(u))

        if prompt.lower().startswith("gemini edit"):
            base_img = decode_img(urls[0])
            return edit_gemini_image(base_img, prompt[11:].strip())

        base_img = decode_img(urls[0])
        result = await get_openai_image_client().images.edits(
            model="gpt-image-1.5",
            image=base_img,
            prompt=prompt,
            size="1024x1024",
        )
        b64_image = result.data[0].b64_json if result and result.data else None
        if not b64_image:
            return None
        return BytesIO(base64.b64decode(b64_image))
    except Exception:
        logger.exception("Error editing image")
        return None
