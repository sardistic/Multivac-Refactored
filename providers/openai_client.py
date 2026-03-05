from __future__ import annotations

import os
from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_DEFAULT_MODEL

openai_client: AsyncOpenAI | None = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_openai_client() -> AsyncOpenAI:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return openai_client


USE_RESPONSES = os.getenv("OPENAI_USE_RESPONSES", "").lower() in {"1", "true", "yes", "y", "on"}
OPENAI_CHAT_MODEL = OPENAI_DEFAULT_MODEL
OPENAI_INTENT_MODEL = os.getenv("OPENAI_INTENT_MODEL", OPENAI_DEFAULT_MODEL)
