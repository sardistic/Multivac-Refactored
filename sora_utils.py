import logging
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from config import OPENAI_API_KEY

logger = logging.getLogger("sora_utils")

# API Constants
API_BASE = "https://api.openai.com/v1"

async def create_sora_job(prompt: str, model: str = "sora-2-pro", size: str = "1280x720", seconds: int = 5) -> Dict[str, Any]:
    """
    Start a video generation job.
    """
    url = f"{API_BASE}/videos"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "seconds": str(seconds) # Docs show string "8"
    }
    
    logger.info(f"Creating Sora job: {payload}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status not in (200, 201, 202):
                text = await resp.text()
                logger.error(f"Create Job Failed ({resp.status}): {text}")
                return {"ok": False, "error": f"API {resp.status}: {text}"}
            
            data = await resp.json()
            return {"ok": True, "data": data}

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
