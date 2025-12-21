import logging
import base64
from io import BytesIO
from typing import Optional
from config import GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logging.warning("google-genai SDK not found. Install it to use Gemini features.")

logger = logging.getLogger("gemini_utils")

def _get_client():
    if not SDK_AVAILABLE or not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_gemini_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[BytesIO]:
    """
    Generate an image using Google Gemini (Imagen 3) via the SDK.
    """
    client = _get_client()
    if not client:
        return None

    # Supported aspect ratios for Imagen: "1:1", "3:4", "4:3", "9:16", "16:9"
    aspect_ratio = "1:1"
    if width > height:
        aspect_ratio = "16:9"
    elif height > width:
        aspect_ratio = "9:16"

    try:
        # User requested: gemini-2.5-flash-image
        # And used client.models.generate_content in their example.
        model = "gemini-2.5-flash-image"
        
        logger.info(f"Generating image with model: {model}")
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            # config=types.GenerateContentConfig(
            #    response_mime_type="image/png" 
            # ) 
            # "image/png" caused 400 INVALID_ARGUMENT. Removing config to let defaults handle it.
        )
        
        # Parse response for image
        if response.parts:
            for part in response.parts:
                # Check for inline data (common for generated images in Gemini models)
                if part.inline_data:
                    buf = BytesIO(part.inline_data.data)
                    return buf
                
                # Check for 'as_image()' method (SDK helper)
                if hasattr(part, "as_image"):
                    try:
                        img = part.as_image()
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        buf.seek(0)
                        return buf
                    except Exception:
                        pass
                        
        logger.warning(f"Response returned no image parts: {response}")

    except Exception as e:
        logger.exception(f"Gemini generation failed (model={model}): {e}")

    return None

def edit_gemini_image(image_bytes: BytesIO, prompt: str) -> Optional[BytesIO]:
    """
    Edit an image using Gemini.
    WARNING: As of late 2024, 'edit' via SDK might require specific models like gemini-2.0-flash-exp 
    handling multi-modal input for *instruction based* editing, or a specific edit endpoint.
    
    If pure 'edit' isn't supported, we might treat it as "image-to-image" generation 
    (providing image + prompt).
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Load bytes into PIL Image
        input_image = PILImage.open(image_bytes)

        # Gemini 2.5 Flash Image / 2.0 Flash Exp often support image input
        model = "gemini-2.0-flash-exp" 
        
        # Construct the prompt with image
        # Note: This is Technically "multimodal generation", closest to "edit" 
        # if we ask it to "change X to Y" in the prompt along with the image.
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt, input_image],
            # We want an image back? 
            # Currently Gemini 2.0 Flash might mostly return text unless explicitly capable of outputting images.
            # TRUE image editing (inpainting/editing) usually requires Imagen models with specific edit endpoints.
            # But the user linked docs for "image generation" which includes editing sections often.
            # If standard generate_content doesn't return image, we might need to wait for full support.
            
            # However, user snippet showed "generate_content" returning parts with "inline_data".
            # Let's try that pattern from their snippet.
        )

        for part in response.parts:
            if part.inline_data: 
                # Decode
                # The SDK might provide a helper or we handle raw bytes
                # part.inline_data.data is usually bytes
                return BytesIO(part.inline_data.data)
                
            # Or if the SDK wraps it as executable code/image?
            # User snippet: part.as_image()
            try:
                if hasattr(part, "as_image"):
                    img = part.as_image()
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
            except:
                pass

    except Exception as e:
        logger.exception(f"Gemini edit failed: {e}")

    return None
