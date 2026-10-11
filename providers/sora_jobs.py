from __future__ import annotations

import math
from io import BytesIO
import logging
from typing import Any, Dict, Optional

import aiohttp
from PIL import Image, ImageOps

from providers.sora_client import API_BASE, build_session, sora_headers
from services.server_effects import visual_prompt

logger = logging.getLogger("sora_utils")
DEFAULT_VIDEO_SIZE = "1280x720"
SUPPORTED_VIDEO_SIZES = ("720x1280", "1280x720", "1024x1792", "1792x1024")
STANDARD_VIDEO_SIZES = ("720x1280", "1280x720")


def _size_dims(size: str) -> tuple[int, int]:
    width_text, height_text = str(size).split("x", 1)
    return int(width_text), int(height_text)


def supported_video_sizes_for_model(model: str) -> tuple[str, ...]:
    model_name = (model or "").strip().lower()
    if model_name.startswith("sora-2-pro"):
        return SUPPORTED_VIDEO_SIZES
    if model_name.startswith("sora-2"):
        return STANDARD_VIDEO_SIZES
    return STANDARD_VIDEO_SIZES


def select_reference_video_size(
    image_data: bytes,
    default_size: str = DEFAULT_VIDEO_SIZE,
    model: str = "sora-2-pro",
) -> str:
    if not image_data:
        return default_size

    try:
        with Image.open(BytesIO(image_data)) as image:
            width, height = image.size
    except Exception as e:
        logger.warning("Failed to inspect Sora reference image size: %s", e)
        return default_size

    if width <= 0 or height <= 0 or width == height:
        return default_size

    supported_sizes = supported_video_sizes_for_model(model)
    image_ratio = width / height
    if width > height:
        candidates = [size for size in supported_sizes if _size_dims(size)[0] > _size_dims(size)[1]]
    else:
        candidates = [size for size in supported_sizes if _size_dims(size)[0] < _size_dims(size)[1]]

    def _score(size: str) -> tuple[float, int]:
        target_width, target_height = _size_dims(size)
        target_ratio = target_width / target_height
        # Log-space keeps portrait/landscape ratio comparisons symmetric.
        return abs(math.log(image_ratio / target_ratio)), 0 if size == default_size else 1

    selected = min(candidates or list(supported_sizes), key=_score)
    logger.info("Auto-selected Sora size %s for model %s and reference image %sx%s", selected, model, width, height)
    return selected


def prepare_reference_image_for_size(
    image_data: bytes,
    size: str,
    output_format: str = "PNG",
) -> bytes:
    if not image_data:
        return image_data

    target_width, target_height = _size_dims(size)
    try:
        with Image.open(BytesIO(image_data)) as image:
            src = ImageOps.exif_transpose(image)
            if src.mode not in ("RGB", "RGBA"):
                src = src.convert("RGBA" if "A" in src.getbands() else "RGB")

            # Preserve the full subject by fitting inside the target frame,
            # then center it on a solid background sized for Sora.
            fitted = ImageOps.contain(src, (target_width, target_height), method=Image.Resampling.LANCZOS)
            if fitted.mode != "RGB":
                fitted = fitted.convert("RGBA")
                background = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 255))
                offset = ((target_width - fitted.width) // 2, (target_height - fitted.height) // 2)
                background.alpha_composite(fitted, dest=offset)
                output = background.convert("RGB")
            else:
                output = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                offset = ((target_width - fitted.width) // 2, (target_height - fitted.height) // 2)
                output.paste(fitted, offset)

            buf = BytesIO()
            output.save(buf, format=output_format)
            return buf.getvalue()
    except Exception as e:
        logger.warning("Failed to normalize Sora reference image to %s: %s", size, e)
        return image_data


async def create_sora_job(
    prompt: str,
    model: str = "sora-2-pro",
    size: Optional[str] = None,
    seconds: int = 8,
    image_data: bytes = None,
    image_filename: Optional[str] = None,
    image_content_type: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = visual_prompt(prompt)
    url = f"{API_BASE}/videos"
    resolved_size = size or DEFAULT_VIDEO_SIZE
    if image_data and size in (None, "", "auto"):
        resolved_size = select_reference_video_size(
            image_data,
            default_size=DEFAULT_VIDEO_SIZE,
            model=model,
        )
    if image_data:
        image_data = prepare_reference_image_for_size(image_data, resolved_size)
        image_filename = "input.png"
        image_content_type = "image/png"

    if image_data:
        data = aiohttp.FormData()
        data.add_field("model", model)
        data.add_field("prompt", prompt)
        data.add_field("size", resolved_size)
        data.add_field("seconds", str(seconds))
        data.add_field(
            "input_reference",
            image_data,
            filename=image_filename or "input.png",
            content_type=image_content_type or "image/png",
        )
        async with build_session() as session:
            async with session.post(url, headers=sora_headers(), data=data) as resp:
                if resp.status not in (200, 201, 202):
                    text = await resp.text()
                    logger.error("Create Job Failed (%s): %s", resp.status, text)
                    return {"ok": False, "error": f"API {resp.status}: {text}"}
                return {"ok": True, "data": await resp.json()}

    payload = {"model": model, "prompt": prompt, "size": resolved_size, "seconds": str(seconds)}
    async with build_session() as session:
        async with session.post(url, headers=sora_headers(json_content=True), json=payload) as resp:
            if resp.status not in (200, 201, 202):
                text = await resp.text()
                logger.error("Create Job Failed (%s): %s", resp.status, text)
                return {"ok": False, "error": f"API {resp.status}: {text}"}
            return {"ok": True, "data": await resp.json()}


async def remix_sora_video(video_id: str, prompt: str) -> Dict[str, Any]:
    prompt = visual_prompt(prompt)
    url = f"{API_BASE}/videos/{video_id}/remix"
    payload = {"prompt": prompt}
    async with build_session() as session:
        async with session.post(url, headers=sora_headers(json_content=True), json=payload) as resp:
            if resp.status not in (200, 201, 202):
                text = await resp.text()
                logger.error("Remix Job Failed (%s): %s", resp.status, text)
                return {"ok": False, "error": f"API {resp.status}: {text}"}
            return {"ok": True, "data": await resp.json()}


async def get_sora_status(video_id: str) -> Dict[str, Any]:
    url = f"{API_BASE}/videos/{video_id}"
    async with build_session() as session:
        async with session.get(url, headers=sora_headers()) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"ok": False, "error": f"Poll Failed ({resp.status}): {text}"}
            return {"ok": True, "data": await resp.json()}


async def download_sora_content(video_id: str) -> Optional[bytes]:
    url = f"{API_BASE}/videos/{video_id}/content"
    try:
        async with build_session() as session:
            async with session.get(url, headers=sora_headers()) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("Download Failed (%s): %s", resp.status, text)
                    return None
                return await resp.read()
    except Exception as e:
        logger.error("Download exception: %s", e)
        return None
