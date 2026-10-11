"""Explicit per-account workload defaults, independent of profile inference."""

from dataclasses import dataclass

from services.database_utils import store

WORKLOADS = ("chat", "research", "deep", "vision", "image", "edit", "video")
TIERS = ("auto", "economy", "balanced", "quality")


def set_preferences(user_id, workload: str, tier: str):
    if workload not in (*WORKLOADS, "all") or tier not in TIERS:
        raise ValueError("Choose a supported workload and tier.")
    keys = WORKLOADS if workload == "all" else (workload,)
    store.set_model_preferences(user_id, {key: tier for key in keys})


def get_preferences(user_id) -> dict[str, str]:
    return {key: value for key, value in store.get_model_preferences(user_id).items()
            if key in WORKLOADS and value in TIERS}


def preference_tier(user_id, workload: str) -> str:
    return get_preferences(user_id).get(workload, "auto")


def text_model_for_tier(tier: str, default: str) -> str:
    return {"economy": "gpt-6-luna", "balanced": "gpt-6.1-sol", "quality": "gpt-6-astra"}.get(tier, default)


@dataclass(frozen=True)
class ImageModelChoice:
    model: str = "gpt-image-2.5-sunburst"
    quality: str = "high"

    @property
    def label(self):
        name = "GPT Image Mini" if self.model == "gpt-image-1-mini" else "GPT Image 2.5 Sunburst"
        return f"{name} ({self.quality})"


def image_choice(user_id, workload="image") -> ImageModelChoice:
    tier = preference_tier(user_id, workload)
    return {
        "economy": ImageModelChoice("gpt-image-1-mini", "low"),
        "balanced": ImageModelChoice("gpt-image-2.5-sunburst", "medium"),
    }.get(tier, ImageModelChoice())
