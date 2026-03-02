from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import aiohttp

from providers.sora_client import API_BASE, build_session, sora_headers

logger = logging.getLogger("sora_utils")


async def create_sora_job(prompt: str, model: str = "sora-2-pro", size: str = "1280x720", seconds: int = 8, image_data: bytes = None) -> Dict[str, Any]:
    url = f"{API_BASE}/videos"
    if image_data:
        data = aiohttp.FormData()
        data.add_field("model", model)
        data.add_field("prompt", prompt)
        data.add_field("size", size)
        data.add_field("seconds", str(seconds))
        data.add_field("input_reference", image_data, filename="input.jpg", content_type="image/jpeg")
        async with build_session() as session:
            async with session.post(url, headers=sora_headers(), data=data) as resp:
                if resp.status not in (200, 201, 202):
                    text = await resp.text()
                    logger.error("Create Job Failed (%s): %s", resp.status, text)
                    return {"ok": False, "error": f"API {resp.status}: {text}"}
                return {"ok": True, "data": await resp.json()}

    payload = {"model": model, "prompt": prompt, "size": size, "seconds": str(seconds)}
    async with build_session() as session:
        async with session.post(url, headers=sora_headers(json_content=True), json=payload) as resp:
            if resp.status not in (200, 201, 202):
                text = await resp.text()
                logger.error("Create Job Failed (%s): %s", resp.status, text)
                return {"ok": False, "error": f"API {resp.status}: {text}"}
            return {"ok": True, "data": await resp.json()}


async def remix_sora_video(video_id: str, prompt: str) -> Dict[str, Any]:
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
