"""Allowlisted image configurations and output prices (USD, checked 2026-10-18).

Sources: https://developers.openai.com/api/docs/guides/image-generation
https://developers.openai.com/api/docs/pricing
https://ai.google.dev/gemini-api/docs/pricing#gemini-3-pro-image
Input tokens are additional. Sunburst has no fixed per-image quote.
"""

import os
from dataclasses import dataclass


def gemini_output_cost(size):
    key, default = ("GEMINI_IMAGE_COST_4K_USD", "0.24") if size == "4K" else ("GEMINI_IMAGE_COST_USD", "0.134")
    return float(os.getenv(key, default))


@dataclass(frozen=True)
class ImageConfig:
    provider: str
    model: str
    quality: str
    size: str
    aspect: str = "1:1"

    @property
    def value(self):
        return "|".join((self.provider, self.model, self.quality, self.size, self.aspect))

    @property
    def name(self):
        return {"gpt-image-1-mini": "GPT Image Mini", "gpt-image-2.5-sunburst": "Sunburst", "gemini-3-pro-image": "Gemini 3 Pro Image"}[self.model]

    @property
    def label(self):
        size = f"{self.size} {self.aspect}" if self.provider == "gemini" else self.size
        return f"{self.name} · {self.quality} · {size}"

    @property
    def cost(self):
        if self.provider == "gemini":
            return gemini_output_cost(self.size)
        if self.model == "gpt-image-1-mini":
            square, rectangle = {"low": (0.005, 0.006), "medium": (0.011, 0.015), "high": (0.036, 0.052)}[self.quality]
            return square if self.size == "1024x1024" else rectangle
        return None

    @property
    def price_label(self):
        return "variable: $0.030/1K output tokens" if self.cost is None else f"~${self.cost:.3f} output"

    @property
    def tier(self):
        if self.provider == "gemini" or self.quality == "high":
            return "quality"
        return "economy" if self.model == "gpt-image-1-mini" and self.quality == "low" else "balanced"

    def provider_kwargs(self):
        return dict(
            use_gemini=self.provider == "gemini", image_model=self.model,
            image_quality=self.quality, image_model_label=self.label,
            image_size=self.size if self.provider == "openai" else "auto",
            gemini_image_size=self.size if self.provider == "gemini" else None,
            gemini_aspect_ratio=self.aspect if self.provider == "gemini" else None,
            allow_stability=False, allow_fallback=False, selection=self,
        )


IMAGE_CONFIGS = tuple(
    ImageConfig("openai", model, quality, size)
    for model in ("gpt-image-1-mini", "gpt-image-2.5-sunburst")
    for quality in ("low", "medium", "high")
    for size in ("1024x1024", "1536x1024", "1024x1536")
) + tuple(
    ImageConfig("gemini", "gemini-3-pro-image", "standard", size, aspect)
    for size in ("2K", "4K") for aspect in ("1:1", "16:9", "9:16")
)
DEFAULT_IMAGE_CONFIG = IMAGE_CONFIGS[0]


def available_image_providers():
    from providers import gemini_client, stability_client
    return {
        "openai": stability_client.openai_image_client is not None,
        "gemini": bool(gemini_client.SDK_AVAILABLE and gemini_client.GEMINI_API_KEY),
    }
