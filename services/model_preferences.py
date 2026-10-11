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
    from services.server_policy import effective_tier
    return effective_tier(get_preferences(user_id).get(workload, "auto"), workload)


def text_model_for_tier(tier: str, default: str, *, user_id=None, workload="chat") -> str:
    selected = {"economy": "gpt-6-luna", "balanced": "gpt-6.1-sol", "quality": "gpt-6-astra"}.get(tier, default)
    if user_id is not None:
        from services.server_policy import get_policy
        policy = get_policy()
        personal = get_preferences(user_id).get(workload, "auto")
        configured = policy["defaults"].get(workload, "auto") if personal == "auto" else personal
        cap = policy["ceilings"].get(workload)
        # A ceiling alone must not upgrade the existing cheap/tiny chat routes.
        rank = {"gpt-6-luna": 0, "gpt-6.1-sol": 1, "gpt-6-sol": 1, "gpt-6-astra": 2}
        if configured == "auto" and cap and rank.get(default, 2) <= {"economy": 0, "balanced": 1, "quality": 2}[cap]:
            return default
    return selected


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
