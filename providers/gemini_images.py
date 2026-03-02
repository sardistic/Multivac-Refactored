from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional

from providers.gemini_client import PILImage, get_gemini_client, types

logger = logging.getLogger("gemini_utils")


def generate_gemini_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[BytesIO]:
    client = get_gemini_client()
    if not client or not types:
        return None

    aspect_ratio = "1:1"
    if width > height:
        aspect_ratio = "16:9"
    elif height > width:
        aspect_ratio = "9:16"

    model = "gemini-3-pro-image-preview"
    try:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size="1024x1024"),
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ],
        )
        response = client.models.generate_content(model=model, contents=[prompt], config=config)
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    return BytesIO(part.inline_data.data)
                if hasattr(part, "as_image"):
                    try:
                        img = part.as_image()
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        buf.seek(0)
                        return buf
                    except Exception:
                        pass
        logger.warning("Response returned no image parts: %s", response)
    except Exception as e:
        logger.exception("Gemini generation failed (model=%s): %s", model, e)
    return None


def edit_gemini_image(image_bytes: BytesIO, prompt: str) -> Optional[BytesIO]:
    client = get_gemini_client()
    if not client or not types or not PILImage:
        return None

    try:
        input_image = PILImage.open(image_bytes)
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, input_image],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                ],
            ),
        )
        for part in response.parts or []:
            if part.inline_data:
                return BytesIO(part.inline_data.data)
            if hasattr(part, "as_image"):
                try:
                    img = part.as_image()
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Gemini edit failed: %s", e)
    return None


def generate_gemini_with_references(prompt: str, reference_images: list[BytesIO]) -> Optional[BytesIO]:
    client = get_gemini_client()
    if not client or not types or not PILImage:
        return None

    try:
        pil_images = [PILImage.open(img_bytes) for img_bytes in reference_images]
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[prompt, *pil_images],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1"),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                ],
            ),
        )
        for part in response.parts or []:
            if part.inline_data:
                return BytesIO(part.inline_data.data)
            if hasattr(part, "as_image"):
                try:
                    img = part.as_image()
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Gemini ref-gen failed: %s", e)
    return None

