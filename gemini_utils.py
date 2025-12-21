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
        # Docs suggest: gemini-2.0-flash-exp or imagen-3.0-generate-002
        # User snippet used: gemini-2.5-flash-image?
        # Let's try "imagen-3.0-generate-002" as verified earlier, or adhere to user if requested.
        # User snippet showed: model="gemini-2.5-flash-image"
        # We will try the user's suggestion first, falling back if needed? 
        # Actually, let's stick to the one we know works or the user's specific request.
        # User snippet clearly requested "gemini-2.5-flash-image".
        
        model = "imagen-3.0-generate-002" 

        response = client.models.generate_image(
            model=model,
            prompt=prompt,
            config=types.GenerateImageConfig(
                aspect_ratio=aspect_ratio,
                sample_count=1,
            )
        )
        
        if response.generated_images:
            img = response.generated_images[0]
            if img.image:
                # SDK returns a PIL Image object usually if requesting raw, 
                # but 'generated_images' entry has 'image' field which is PIL.Image?
                # The SDK docs say: .generated_images[].image is a PIL.Image.Image
                
                buf = BytesIO()
                img.image.save(buf, format="PNG")
                buf.seek(0)
                return buf

    except Exception as e:
        logger.exception(f"Gemini generation failed: {e}")

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
