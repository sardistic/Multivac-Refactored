from __future__ import annotations

import logging
import os

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, STABILITY_HOST, STABILITY_KEY

logger = logging.getLogger("stability_utils")

if STABILITY_HOST:
    os.environ["STABILITY_HOST"] = STABILITY_HOST

if STABILITY_KEY:
    os.environ["STABILITY_KEY"] = STABILITY_KEY
else:
    logging.warning("STABILITY_KEY not set; Stability image generation/editing will be disabled.")

openai_image_client: AsyncOpenAI | None = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_openai_image_client() -> AsyncOpenAI:
    if openai_image_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return openai_image_client


STABILITY_AVAILABLE = False
stability_client = None
generation = None
if STABILITY_KEY:
    try:
        from stability_sdk import client as _stability_client
        import stability_sdk.interfaces.gooseai.generation.generation_pb2 as _generation

        stability_client = _stability_client
        generation = _generation
        STABILITY_AVAILABLE = True
    except Exception as e:
        logging.warning("stability_sdk not available or failed to import; falling back to OpenAI images only. %s", e)
        STABILITY_AVAILABLE = False
