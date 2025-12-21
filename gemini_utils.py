import os
import requests
import logging
import base64
from io import BytesIO
from typing import Optional
from config import GEMINI_API_KEY

logger = logging.getLogger("gemini_utils")

def generate_gemini_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[BytesIO]:
    """
    Generate an image using the Google Gemini Imagen 3 API via REST.
    Returns a BytesIO object of the generated image (PNG/JPEG) or None on failure.
    """
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set; cannot generate images.")
        return None

    # Endpoint for Imagen 3 (check documentation for latest model string if needed)
    # The user linked: https://ai.google.dev/gemini-api/docs/image-generation
    # Example model: "imagen-3.0-generate-001"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # "aspectRatio" can be "1:1", "3:4", "4:3", "9:16", "16:9"
    # We'll approximate based on width/height or default to 1:1
    aspect_ratio = "1:1"
    if width > height:
        aspect_ratio = "16:9" # or 4:3
    elif height > width:
        aspect_ratio = "9:16" # or 3:4

    payload = {
        "instances": [
            {
                "prompt": prompt
            }
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"Gemini API error ({response.status_code}): {response.text}")
            return None
            
        data = response.json()
        
        # Parse response structure:
        # {
        #   "predictions": [
        #     {
        #       "bytesBase64Encoded": "..."
        #       "mimeType": "image/png"
        #     }
        #   ]
        # }
        
        predictions = data.get("predictions")
        if not predictions:
            logger.warning("No predictions returned from Gemini.")
            return None
            
        first_image = predictions[0]
        b64_data = first_image.get("bytesBase64Encoded")
        if not b64_data:
            logger.warning("No bytesBase64Encoded found in prediction.")
            return None
            
        image_bytes = base64.b64decode(b64_data)
        return BytesIO(image_bytes)

    except Exception as e:
        logger.exception(f"Exception during Gemini image generation: {e}")
        return None
