# stability_utils.py
from __future__ import annotations

import os
import re
import io
import base64
import logging
import random
import tempfile
from io import BytesIO
from typing import Optional

import requests
from PIL import Image
from openai import AsyncOpenAI

from config import STABILITY_HOST, STABILITY_KEY, OPENAI_API_KEY

# --- Environment setup (safe) -------------------------------------------------

if STABILITY_HOST:
    os.environ["STABILITY_HOST"] = STABILITY_HOST

if STABILITY_KEY:
    os.environ["STABILITY_KEY"] = STABILITY_KEY
else:
    logging.warning("STABILITY_KEY not set; Stability image generation/editing will be disabled.")

# OpenAI Async Client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Try to import stability only if we have a key; keep optional
_STABILITY_AVAILABLE = False
if STABILITY_KEY:
    try:
        from stability_sdk import client
        import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
        _STABILITY_AVAILABLE = True
    except Exception as e:
        logging.warning("stability_sdk not available or failed to import; "
                        "falling back to OpenAI images only. %s", e)
        _STABILITY_AVAILABLE = False


from gemini_utils import generate_gemini_image, edit_gemini_image

# -- CORE IMAGE FUNCTIONS ------------------------------------------------------

async def handle_image_generation(message, prompt: str, reply_msg=None) -> Optional[BytesIO]:
    """
    Decide which image backend to use based on the prompt.
    - 'stable imagine ...' => Stability (if available)
    - 'gemini imagine ...' => Gemini (if available)
    - otherwise => GPT image.
    Returns an in-memory PNG (BytesIO) or None on failure.
    """
    try:
        width, height = extract_width_height_from_prompt(prompt)

        # 1) Stability
        if prompt.lower().startswith("stable imagine"):
            image_prompt = prompt[15:].strip()
            if _STABILITY_AVAILABLE:
                img = await generate_stability_image(image_prompt, width, height)
                if img:
                    return img
                logging.warning("Stability generation failed; falling back to GPT image.")
            return await generate_gpt_image(image_prompt)

        # 2) Gemini
        if prompt.lower().startswith("gemini imagine"):
            image_prompt = prompt[14:].strip() 
            img = generate_gemini_image(image_prompt, width, height)
            if img:
                return img
            logging.warning("Gemini generation failed; falling back to GPT image.")
            return await generate_gpt_image(image_prompt)

        # Default path: GPT image
        return await generate_gpt_image(prompt)

    except Exception:
        logging.exception("Error in handle_image_generation")
        return None


async def generate_stability_image(image_prompt: str, width: int = 960, height: int = 768) -> Optional[BytesIO]:
    """
    Generate an image via Stability if available. Returns PNG BytesIO or None.
    """
    if not (_STABILITY_AVAILABLE and STABILITY_KEY):
        logging.warning("generate_stability_image called but Stability is not configured.")
        return None

    try:
        stability_api = client.StabilityInference(
            key=os.environ["STABILITY_KEY"],
            verbose=True,
            # You can make this configurable if you switch engines frequently
            engine="stable-diffusion-v1-5",
        )
        random_seed = random.randint(0, 2**32 - 1)

        answers = stability_api.generate(
            prompt=image_prompt,
            seed=random_seed,
            steps=50,
            cfg_scale=11.0,
            width=width,
            height=height,
            samples=1,
            sampler=generation.SAMPLER_K_EULER_ANCESTRAL,
        )

        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf

        logging.warning("Stability returned no image artifacts.")
        return None

    except Exception:
        logging.exception("Error generating Stability image")
        return None


async def generate_gpt_image(prompt: str) -> Optional[BytesIO]:
    """
    Generate an image via OpenAI Images API. Returns PNG BytesIO or None.
    """
    try:
        background_type = "transparent" if "transparent background" in prompt.lower() else "auto"

        # NOTE: These params match current OpenAI Images API; if your SDK version
        # lacks 'background' or 'quality', remove them (the call will still work).
        result = await openai_client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            size="1024x1024",
            background=background_type,  # ok to omit if SDK complains
            quality="high",              # ok to omit if SDK complains
            moderation="low",            # ok to omit if SDK complains
            n=1,
        )

        b64_image = result.data[0].b64_json if result and result.data else None
        if not b64_image:
            logging.warning("No image returned from GPT-image")
            return None

        return BytesIO(base64.b64decode(b64_image))

    except Exception:
        logging.exception("Error generating GPT image")
        return None


async def edit_image_with_prompt(image_url: str, prompt: str) -> Optional[BytesIO]:
    """
    Edit an image with a text prompt.
    - If prompt starts with 'gemini edit', uses Gemini SDK.
    - Else uses OpenAI Images API.
    """
    try:
        # 1) Fetch image bytes first
        if image_url.startswith("data:image/"):
            image_b64 = image_url.split(",", 1)[1]
            image_bytes = base64.b64decode(image_b64)
        else:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(image_url, headers=headers, timeout=20)
            resp.raise_for_status()
            image_bytes = resp.content

        # 2) Gemini Edit
        if prompt.lower().startswith("gemini edit"):
            edit_prompt = prompt[11:].strip()
            # We need a BytesIO for the SDK utils
            buf = BytesIO(image_bytes)
            return edit_gemini_image(buf, edit_prompt)

        # 3) OpenAI Edit (Fallback/Default)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        with open(temp_path, "rb") as img_file:
            result = await openai_client.images.edit(
                model="gpt-image-1",
                prompt=prompt,
                image=[img_file],
            )

        edited_b64 = result.data[0].b64_json if result and result.data else None
        if not edited_b64:
            logging.warning("No edited image returned.")
            return None

        return BytesIO(base64.b64decode(edited_b64))

    except Exception:
        logging.exception("Error editing image")
        return None
    finally:
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass


# -- HELPERS -------------------------------------------------------------------

def extract_width_height_from_prompt(prompt: str) -> tuple[int, int]:
    """
    Parse "<W>x<H>" in the prompt, e.g., "768x512".
    Defaults to 1024x1024.
    """
    m = re.search(r"(\d+)x(\d+)", prompt)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 1024, 1024
