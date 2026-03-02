from __future__ import annotations

from config import GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage

    SDK_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    PILImage = None
    SDK_AVAILABLE = False


def get_gemini_client():
    if not SDK_AVAILABLE or not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})

