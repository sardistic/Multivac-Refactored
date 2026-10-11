from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

from providers.gemini_client import PILImage, get_gemini_client, types

logger = logging.getLogger("gemini_utils")

DEFAULT_VEO_MODEL = os.getenv("GEMINI_VEO_MODEL", "veo-3.1-generate-preview")
DEFAULT_VEO_FAST_MODEL = os.getenv("GEMINI_VEO_FAST_MODEL", "veo-3.1-fast-generate-preview")
DEFAULT_VEO_RESOLUTION = os.getenv("GEMINI_VEO_RESOLUTION", "720p")

VEO_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    DEFAULT_VEO_MODEL: {
        "label": "Veo 3.1",
        "emoji": "🎬",
        "cost_per_second": 0.40,
        "durations": (4, 6, 8),
        "supports_audio": True,
    },
    DEFAULT_VEO_FAST_MODEL: {
        "label": "Veo 3.1 Fast",
        "emoji": "⚡",
        "cost_per_second": 0.10,
        "durations": (4, 6, 8),
        "supports_audio": True,
    },
}


def veo_is_available() -> bool:
    return get_gemini_client() is not None and types is not None


def get_veo_model_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for model, config in VEO_MODEL_CONFIGS.items():
        for seconds in config["durations"]:
            options.append(
                {
                    "provider": "veo",
                    "model": model,
                    "provider_label": config["label"],
                    "seconds": seconds,
                    "emoji": config["emoji"],
                    "cost": estimate_veo_cost(model, seconds),
                    "generate_audio": False,
                }
            )
    return options


def get_veo_model_label(model: str) -> str:
    config = VEO_MODEL_CONFIGS.get(model)
    if config:
        return str(config["label"])
    return model


def estimate_veo_cost(model: str, seconds: int, *, generate_audio: bool = False) -> float:
    config = VEO_MODEL_CONFIGS.get(model, {})
    # Veo 3.1 generates audio natively, and Gemini API pricing includes it.
    # Keep generate_audio in the signature for existing callers, but do not
    # apply a separate audio surcharge.
    rate = float(config.get("cost_per_second", 0.40))
    return round(rate * int(seconds), 2)


def estimate_veo_runtime(model: str, seconds: int) -> int:
    per_second = 12 if "fast" in (model or "").lower() else 18
    return max(45, int(seconds) * per_second)


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_error_message(error: Any) -> str:
    if error is None:
        return "Unknown error"
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        return error.get("message") or error.get("code") or str(error)
    message = getattr(error, "message", None)
    if message:
        return str(message)
    code = getattr(error, "code", None)
    if code:
        return f"{code}: {error}"
    return str(error)


def _extract_generated_video(operation: Any) -> Any:
    for container_name in ("response", "result"):
        container = _safe_attr(operation, container_name)
        generated = _safe_attr(container, "generated_videos")
        if generated:
            return generated[0]
    return None


def _serialize_veo_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(k): _serialize_veo_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_veo_value(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _serialize_veo_value(value.model_dump(exclude_none=True))
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _serialize_veo_value(vars(value))
        except Exception:
            pass
    return str(value)


def _collect_veo_operation_snapshots(operation: Any) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for container_name in ("response", "result", "metadata"):
        container = _safe_attr(operation, container_name)
        serialized = _serialize_veo_value(container)
        if serialized not in (None, {}, []):
            snapshots[container_name] = serialized
    return snapshots


def _compact_json(data: Any, *, max_len: int = 220) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        text = str(data)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _build_veo_no_video_message(operation_name: str, snapshots: dict[str, Any]) -> str:
    response = snapshots.get("response") or {}
    result = snapshots.get("result") or {}
    filtered_count = None
    filtered_reasons = None

    for payload in (response, result):
        if isinstance(payload, dict):
            if filtered_count in (None, 0):
                filtered_count = payload.get("rai_media_filtered_count", filtered_count)
            if not filtered_reasons:
                filtered_reasons = payload.get("rai_media_filtered_reasons")

    if filtered_count or filtered_reasons:
        reasons_text = ", ".join(str(r) for r in (filtered_reasons or [])) or "unspecified"
        count_text = f" filtered_count={filtered_count};" if filtered_count not in (None, 0) else ""
        return f"Veo output was filtered by safety checks.{count_text} reasons={reasons_text}"

    debug_parts = [
        f"{name}={_compact_json(payload, max_len=140)}"
        for name, payload in snapshots.items()
    ]
    if debug_parts:
        return f"Veo completed without a downloadable video. Debug: {'; '.join(debug_parts)}"
    return f"Veo completed without a downloadable video (operation={operation_name})."


def _select_veo_aspect_ratio(image_data: bytes | None) -> str:
    if not image_data or not PILImage:
        return "16:9"
    try:
        from io import BytesIO

        with PILImage.open(BytesIO(image_data)) as image:
            width, height = image.size
            if width > 0 and height > 0 and height > width:
                return "9:16"
    except Exception as e:
        logger.warning("Failed to inspect Veo reference image size: %s", e)
    return "16:9"


async def generate_veo_video(
    prompt: str,
    *,
    model: str = DEFAULT_VEO_MODEL,
    seconds: int = 8,
    image_data: bytes | None = None,
    image_content_type: Optional[str] = None,
    generate_audio: bool = False,
    progress_state: Optional[dict[str, Any]] = None,
) -> tuple[bytes | None, str | None]:
    from services.server_effects import visual_prompt
    prompt = visual_prompt(prompt)
    client = get_gemini_client()
    if not client or not types:
        return None, "GEMINI_API_KEY is not configured for Veo video generation."

    if progress_state is not None:
        progress_state["status"] = "Submitting Veo job"
        progress_state["progress"] = 0.05

    image = None
    if image_data:
        image = types.Image(
            image_bytes=image_data,
            mime_type=image_content_type or "image/png",
        )

    aspect_ratio = _select_veo_aspect_ratio(image_data)

    config = types.GenerateVideosConfig(
        duration_seconds=int(seconds),
        aspect_ratio=aspect_ratio,
        resolution=DEFAULT_VEO_RESOLUTION,
    )

    # The Gemini API docs show Veo generation without a generate_audio field.
    # Audio is native for Veo 3.1, and passing generate_audio through the
    # Gemini API currently raises a client-side validation error.
    if generate_audio:
        logger.info("Ignoring generate_audio=True for Gemini Veo request; audio is handled natively by the model.")

    logger.info(
        "Submitting Veo job (model=%s, seconds=%s, resolution=%s, aspect_ratio=%s, has_reference=%s)",
        model,
        seconds,
        DEFAULT_VEO_RESOLUTION,
        aspect_ratio,
        bool(image_data),
    )

    try:
        operation = await asyncio.to_thread(
            client.models.generate_videos,
            model=model,
            prompt=prompt,
            image=image,
            config=config,
        )
    except Exception as e:
        logger.exception("Veo generation request failed (model=%s): %s", model, e)
        return None, f"Failed to start Veo generation: {e}"

    started_at = asyncio.get_running_loop().time()
    expected_runtime = estimate_veo_runtime(model, seconds)
    operation_name = _safe_attr(operation, "name", "<unknown>")
    logger.info(
        "Veo job started: %s (model=%s, seconds=%s, expected_runtime=%ss)",
        operation_name,
        model,
        seconds,
        expected_runtime,
    )

    while not bool(_safe_attr(operation, "done", False)):
        await asyncio.sleep(8)
        elapsed = asyncio.get_running_loop().time() - started_at
        if progress_state is not None:
            progress_state["status"] = "Waiting on Veo"
            progress_state["progress"] = min(0.92, max(0.10, elapsed / expected_runtime))
        try:
            operation = await asyncio.to_thread(client.operations.get, operation)
            logger.info(
                "Veo poll: %s done=%s elapsed=%ss",
                operation_name,
                bool(_safe_attr(operation, "done", False)),
                int(elapsed),
            )
        except Exception as e:
            logger.warning("Veo poll failed: %s", e)

        if elapsed > 900:
            return None, "Timeout waiting for Veo video generation."

    error = _safe_attr(operation, "error")
    if error:
        logger.warning("Veo job failed: %s error=%s", operation_name, _extract_error_message(error))
        return None, _extract_error_message(error)

    generated_video = _extract_generated_video(operation)
    if not generated_video:
        snapshots = _collect_veo_operation_snapshots(operation)
        failure_message = _build_veo_no_video_message(operation_name, snapshots)
        logger.warning(
            "Veo completed without generated video: %s snapshots=%s",
            operation_name,
            _compact_json(snapshots, max_len=800),
        )
        return None, failure_message

    logger.info("Veo job completed: %s", operation_name)
    video_file = _safe_attr(generated_video, "video", generated_video)
    try:
        logger.info("Downloading Veo video: %s", operation_name)
        content = await asyncio.to_thread(client.files.download, file=video_file)
    except Exception as e:
        logger.exception("Veo video download failed: %s", e)
        return None, f"Failed to download Veo video: {e}"

    if progress_state is not None:
        progress_state["status"] = "Downloading video"
        progress_state["progress"] = 1.0
    logger.info("Veo video downloaded: %s (%s bytes)", operation_name, len(content))
    return content, None


__all__ = [
    "DEFAULT_VEO_FAST_MODEL",
    "DEFAULT_VEO_MODEL",
    "estimate_veo_cost",
    "estimate_veo_runtime",
    "generate_veo_video",
    "get_veo_model_label",
    "get_veo_model_options",
    "veo_is_available",
]
