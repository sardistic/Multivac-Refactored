from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Optional

from providers.gemini_client import PILImage, get_gemini_client, types
from services.server_effects import visual_prompt

logger = logging.getLogger("gemini_utils")

# Output resolution for from-scratch generation. 1K and 2K cost the same on
# Gemini 3 Pro Image ($0.134/img); 4K is ~$0.24/img (and 16MP PNGs risk
# Discord's 10MB upload cap). 2K is the sweet spot for inline display — free
# vs 1K, ~45% cheaper than 4K. Override with GEMINI_IMAGE_SIZE ("1K"/"2K"/"4K").
GEMINI_IMAGE_SIZE = os.getenv("GEMINI_IMAGE_SIZE", "2K").upper()


def generate_gemini_image(prompt: str, width: int = 1024, height: int = 1024, *, image_size: str | None = None, aspect_ratio: str | None = None) -> Optional[BytesIO]:
    prompt = visual_prompt(prompt)
    client = get_gemini_client()
    if not client or not types:
        return None

    aspect_ratio = aspect_ratio or ("16:9" if width > height else "9:16" if height > width else "1:1")
    image_size = image_size or GEMINI_IMAGE_SIZE

    model = "gemini-3-pro-image"
    try:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ],
        )
        response = client.models.generate_content(model=model, contents=[prompt], config=config)
        _record_gemini_image_cost(model, "image_generation", image_size=image_size)
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


def _record_gemini_image_cost(model: str, label: str, image_size: str = "2K") -> None:
    """Per-image ledger entry priced by resolution: 4K is ~$0.24, 1K/2K ~$0.134
    (Gemini 3 Pro Image). Edit/reference paths run at the default (1K/2K) tier.
    Tier prices are overridable via GEMINI_IMAGE_COST_4K_USD / GEMINI_IMAGE_COST_USD."""
    try:
        from services import usage_costs

        from services.image_options import gemini_output_cost
        cost = gemini_output_cost(str(image_size).upper())
        usage_costs.record(model, None, cost, label=label)
    except Exception:
        logger.warning("gemini image usage recording failed", exc_info=True)


def edit_gemini_image(image_bytes: BytesIO, prompt: str) -> Optional[BytesIO]:
    prompt = visual_prompt(prompt)
    client = get_gemini_client()
    if not client or not types or not PILImage:
        return None

    try:
        input_image = PILImage.open(image_bytes)
        _record_gemini_image_cost("gemini-3-pro-image", "image_edit")
        response = client.models.generate_content(
            model="gemini-3-pro-image",
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


def generate_gemini_with_references(prompt: str, reference_images: list[BytesIO], *, image_size: str | None = None, aspect_ratio: str = "1:1") -> Optional[BytesIO]:
    prompt = visual_prompt(prompt)
    client = get_gemini_client()
    if not client or not types or not PILImage:
        return None
    image_config = {"aspect_ratio": aspect_ratio}
    if image_size is not None:
        image_config["image_size"] = image_size

    try:
        pil_images = [PILImage.open(img_bytes) for img_bytes in reference_images]
        response = client.models.generate_content(
            model="gemini-3-pro-image",
            contents=[prompt, *pil_images],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**image_config),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                ],
            ),
        )
        _record_gemini_image_cost("gemini-3-pro-image", "image_generation", image_size=image_size or "2K")
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

