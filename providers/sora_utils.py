import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from config import OPENAI_API_KEY

logger = logging.getLogger("sora_utils")

# API Constants
API_BASE = "https://api.openai.com/v1"

async def create_sora_job(prompt: str, model: str = "sora-2-pro", size: str = "1280x720", seconds: int = 8, image_data: bytes = None) -> Dict[str, Any]:
    """
    Start a video generation job. Supports text-to-video or image-to-video.
    """
    url = f"{API_BASE}/videos"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    # If image provided, use Multipart
    if image_data:
        data = aiohttp.FormData()
        data.add_field("model", model)
        data.add_field("prompt", prompt)
        data.add_field("size", size)
        data.add_field("seconds", str(seconds))
        # Filename/content_type are crucial for the API to recognize it as a file
        data.add_field("input_reference", image_data, filename="input.jpg", content_type="image/jpeg")
        
        # When using FormData, let aiohttp set the Content-Type boundary
        logger.info(f"Creating Sora job (Multipart) with image: {len(image_data)} bytes")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as resp:
                if resp.status not in (200, 201, 202):
                    text = await resp.text()
                    logger.error(f"Create Job Failed ({resp.status}): {text}")
                    return {"ok": False, "error": f"API {resp.status}: {text}"}
                return {"ok": True, "data": await resp.json()}
    
    else:
        # JSON Payload
        headers["Content-Type"] = "application/json"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "seconds": str(seconds)
        }
        logger.info(f"Creating Sora job (JSON): {payload}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status not in (200, 201, 202):
                    text = await resp.text()
                    logger.error(f"Create Job Failed ({resp.status}): {text}")
                    return {"ok": False, "error": f"API {resp.status}: {text}"}
                return {"ok": True, "data": await resp.json()}

async def remix_sora_video(video_id: str, prompt: str) -> Dict[str, Any]:
    """
    Remix an existing video.
    """
    url = f"{API_BASE}/videos/{video_id}/remix"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"prompt": prompt}
    
    logger.info(f"Remixing video {video_id} with prompt: {prompt}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
             if resp.status not in (200, 201, 202):
                 text = await resp.text()
                 logger.error(f"Remix Job Failed ({resp.status}): {text}")
                 return {"ok": False, "error": f"API {resp.status}: {text}"}
             return {"ok": True, "data": await resp.json()}

async def get_sora_status(video_id: str) -> Dict[str, Any]:
    """
    Poll the status of a video job.
    """
    url = f"{API_BASE}/videos/{video_id}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"ok": False, "error": f"Poll Failed ({resp.status}): {text}"}
            
            data = await resp.json()
            return {"ok": True, "data": data}

async def download_sora_content(video_id: str) -> Optional[bytes]:
    """
    Download the final MP4 content.
    """
    url = f"{API_BASE}/videos/{video_id}/content"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Download Failed ({resp.status}): {text}")
                    return None
                
                return await resp.read()
    except Exception as e:
        logger.error(f"Download exception: {e}")
        return None
