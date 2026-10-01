from __future__ import annotations

import os
from openai import AsyncOpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_CASUAL_MODEL,
    OPENAI_CODE_MODEL as _CONFIG_CODE_MODEL,
    OPENAI_DEEP_MODEL as _CONFIG_DEEP_MODEL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_USE_RESPONSES,
    OPENAI_INTENT_MODEL as _CONFIG_INTENT_MODEL,
    OPENAI_STANDARD_MODEL as _CONFIG_STANDARD_MODEL,
    OPENAI_TINY_MODEL as _CONFIG_TINY_MODEL,
)

openai_client: AsyncOpenAI | None = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_openai_client() -> AsyncOpenAI:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return openai_client


USE_RESPONSES = OPENAI_USE_RESPONSES
OPENAI_CHAT_MODEL = OPENAI_DEFAULT_MODEL
OPENAI_CODE_MODEL = _CONFIG_CODE_MODEL
OPENAI_DEEP_MODEL = _CONFIG_DEEP_MODEL
OPENAI_LIGHT_MODEL = OPENAI_CASUAL_MODEL
OPENAI_STANDARD_MODEL = _CONFIG_STANDARD_MODEL
OPENAI_TINY_MODEL = _CONFIG_TINY_MODEL
OPENAI_INTENT_MODEL = _CONFIG_INTENT_MODEL


def is_reasoning_model(model: str) -> bool:
    """GPT-5/6 and the o-series models are reasoning models: they spend
    (hidden) reasoning tokens before visible output, so token budgets must be
    generous and several classic params (temperature) are rejected."""
    m = (model or "").lower()
    return m.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def model_supports_temperature(model: str) -> bool:
    """Reasoning models reject the `temperature` parameter entirely
    (only the default is allowed)."""
    return not is_reasoning_model(model)


def temperature_kwargs(model: str, temperature: float | None) -> dict:
    """Return {"temperature": ...} only when the model accepts it, else {}."""
    if temperature is None or not model_supports_temperature(model):
        return {}
    return {"temperature": temperature}


def reasoning_kwargs(model: str, effort: str | None, *, responses: bool) -> dict:
    """Return the endpoint-specific reasoning control for supported models."""
    if not effort or not is_reasoning_model(model):
        return {}
    normalized = effort.strip().lower()
    allowed = {"low", "medium", "high"}
    if (model or "").lower().startswith(("gpt-5.6", "gpt-6")):
        allowed.update({"none", "xhigh", "max"})
    if (model or "").lower().startswith(("gpt-6-astra", "gpt-6.1-sol")):
        allowed.discard("none")
    if normalized not in allowed:
        return {}
    if responses:
        return {"reasoning": {"effort": normalized}}
    return {"reasoning_effort": normalized}
