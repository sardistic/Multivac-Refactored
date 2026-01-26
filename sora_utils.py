import logging
import json
import aiohttp
from typing import Dict, Any, Optional
from config import OPENAI_API_KEY

logger = logging.getLogger("sora_utils")

async def generate_sora_video(prompt: str, size: str = "1024x1024") -> Dict[str, Any]:
    """
    Generate a video using OpenAI's Sora model via direct HTTP request.
    Endpoint: https://api.openai.com/v1/video/generations (Provisional)
    """
    url = "https://api.openai.com/v1/video/generations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sora-1.0-turbo",
        "prompt": prompt,
        "size": size,
        "quality": "standard"
    }

    try:
        logger.info(f"Generating Sora video for prompt: {prompt} via {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Sora API Error ({resp.status}): {text}")
                    return {"ok": False, "error": f"API {resp.status}: {text}"}
                
                data = await resp.json()
                # Expected format: {"data": [{"url": "..."}]}
                if "data" in data and len(data["data"]) > 0:
                     vid_url = data["data"][0].get("url")
                     revised = data["data"][0].get("revised_prompt", prompt)
                     if vid_url:
                         return {"ok": True, "url": vid_url, "revised_prompt": revised}
                
                return {"ok": False, "error": "No video URL in response", "raw": data}
        
    except Exception as e:
        logger.error(f"Sora generation failed: {e}")
        return {"ok": False, "error": str(e)}
