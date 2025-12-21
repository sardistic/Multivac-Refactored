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

        # User requested: gemini-2.5-flash-image
        # And used client.models.generate_content in their example.
        model = "gemini-2.5-flash-image"
        
        logger.info(f"Generating image with model: {model}")
        
        # Using the config structure from the user's snippet
        # to ensure proper image generation triggers.
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], # We primarily want image
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="1024x1024" # Equivalent to "1K"? SDK might accept enum or string
            )
        )
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=config
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
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Load bytes into PIL Image
        input_image = PILImage.open(image_bytes)

        # Gemini 2.5 Flash Image
        model = "gemini-2.5-flash-image" 
        
        # Use config to enforce image output
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt, input_image],
            config=config
        )

        if not response.parts:
            logger.warning(f"Gemini edit returned no parts. Response: {response}")
            return None

        for part in response.parts:
            if part.inline_data: 
                return BytesIO(part.inline_data.data)
                
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

def generate_gemini_with_references(prompt: str, reference_images: list[BytesIO]) -> Optional[BytesIO]:
    """
    Generate an image using a text prompt and multiple reference images.
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Load all bytes into PIL Images
        pil_images = []
        for img_bytes in reference_images:
            pil_images.append(PILImage.open(img_bytes))

        # Use gemini-2.5-flash-image for multimodal generation
        model = "gemini-2.5-flash-image"
        
        contents = [prompt] + pil_images
        
        logger.info(f"Generating with references (count={len(pil_images)}) using model: {model}")
        
        # User snippet config pattern
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        )
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        if not response.parts:
            logger.warning(f"Gemini ref-gen returned no parts. Response: {response}")
            return None

        for part in response.parts:
            if part.inline_data: 
                return BytesIO(part.inline_data.data)
                
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
        logger.exception(f"Gemini ref-gen failed: {e}")

    return None
