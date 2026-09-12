from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import random
import re
from io import BytesIO
from typing import Optional

from PIL import Image

from providers.gemini_utils import edit_gemini_image, generate_gemini_image, generate_gemini_with_references
from providers.stability_client import (
    STABILITY_AVAILABLE,
    STABILITY_KEY,
    generation,
    get_openai_image_client,
    stability_client,
)
from services.url_utils import DEFAULT_MEDIA_BYTES, fetch_url_bytes_async

logger = logging.getLogger("stability_utils")
_REPLY_PROMPT_MAX_CHARS = 1800

# Friendly, user-facing model labels shown in the live status / ✅ line.
OPENAI_IMAGE_MODEL = "gpt-image-2.5-sunburst"
IMG_MODEL_OPENAI = "GPT Image 2.5 Sunburst"
IMG_MODEL_GEMINI = "Gemini 3 Pro Image"
IMG_MODEL_STABILITY = "Stable Diffusion 1.5"


# A render that has pictures to look at needs to be told they ARE the subject.
# "imagine this in an art museum" carries no description of "this" -- without
# this note the model is free to treat the reference as loose inspiration and
# invent its own subject.
_REFERENCE_PREAMBLE = (
    "Use the attached image(s) as the visual reference for the subject. Words such as "
    "'this', 'that', 'it', 'him', or 'her' in the request refer to what is shown there. "
    "Keep that subject recognizable in the new image.\n\n"
)


def _with_reference_note(image_prompt: str) -> str:
    return f"{_REFERENCE_PREAMBLE}{(image_prompt or '').strip()}"


def _compose_reply_aware_image_prompt(prompt: str, reply_msg=None, retry_context: str = "") -> str:
    base_prompt = (prompt or "").strip()
    retry_text = re.sub(r"\s+", " ", (retry_context or "").strip())
    if retry_text:
        if len(retry_text) > _REPLY_PROMPT_MAX_CHARS:
            retry_text = retry_text[:_REPLY_PROMPT_MAX_CHARS].rstrip() + "..."
        if base_prompt:
            return (
                "Create a new image that follows the current revision while preserving the "
                "subject and requirements from the prior image context. Do not substitute an "
                "unrelated subject.\n\n"
                f"Current revision request:\n{base_prompt}\n\n"
                "Previous image request/revision context (oldest to newest):\n"
                f"{retry_text}"
            )
        return f"Previous image request/revision context (oldest to newest):\n{retry_text}"

    reply_text = (getattr(reply_msg, "content", "") or "").strip()
    if not reply_text:
        return base_prompt

    reply_text = re.sub(r"\s+", " ", reply_text)
    if len(reply_text) > _REPLY_PROMPT_MAX_CHARS:
        reply_text = reply_text[:_REPLY_PROMPT_MAX_CHARS].rstrip() + "..."

    author = getattr(getattr(reply_msg, "author", None), "display_name", None) or getattr(
        getattr(reply_msg, "author", None), "name", None
    )
    context_label = f"Replied message context from {author}" if author else "Replied message context"
    if base_prompt:
        return f"{base_prompt}\n\n{context_label}:\n{reply_text}"
    return f"{context_label}:\n{reply_text}"


def extract_width_height_from_prompt(prompt: str) -> tuple[int, int]:
    prompt_lower = prompt.lower()
    width = height = 1024
    if "portrait" in prompt_lower or "vertical" in prompt_lower:
        width, height = 768, 1024
    elif "landscape" in prompt_lower or "horizontal" in prompt_lower:
        width, height = 1024, 768
    match = re.search(r"(\d{3,4})\s*[xX]\s*(\d{3,4})", prompt)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    return width, height


async def generate_stability_image(image_prompt: str, width: int = 960, height: int = 768) -> Optional[BytesIO]:
    if not (STABILITY_AVAILABLE and STABILITY_KEY and stability_client and generation):
        logger.warning("generate_stability_image called but Stability is not configured.")
        return None
    try:
        api = stability_client.StabilityInference(
            key=STABILITY_KEY,
            verbose=True,
            engine="stable-diffusion-v1-5",
        )
        answers = api.generate(
            prompt=image_prompt,
            seed=random.randint(0, 2**32 - 1),
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
    except Exception:
        logger.exception("Error generating Stability image")
    return None


class ImageModerationError(Exception):
    """The provider's safety system refused the image prompt."""


# An album should not turn one render into a sixteen-image request. The Discord
# layer already caps what it collects; this keeps a direct caller bounded too.
MAX_REFERENCE_IMAGES = 4


async def _reference_bytes(ref: str) -> bytes:
    """Resolve one reference entry -- data: URI or http(s) URL -- to raw bytes."""
    if ref.startswith("data:"):
        _, _, b64 = ref.partition(",")
        if len(b64) > (DEFAULT_MEDIA_BYTES * 4 // 3) + 16:
            raise ValueError("image input exceeds the allowed size")
        decoded = base64.b64decode(b64, validate=True)
        if len(decoded) > DEFAULT_MEDIA_BYTES:
            raise ValueError("image input exceeds the allowed size")
        return decoded
    fetched = await fetch_url_bytes_async(
        ref,
        timeout=20,
        max_bytes=DEFAULT_MEDIA_BYTES,
        allowed_content_types=("image/",),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return fetched.body


async def collect_reference_images(
    message=None,
    reply_msg=None,
    prompt: str = "",
    reference_urls: list[str] | None = None,
) -> list[bytes]:
    """The pictures this render should look at.

    `reference_urls` is the set the Discord layer already collected for this
    message -- attachments, the replied message, embeds, forwarded posts and
    linked images, deduplicated and size-checked. When a caller supplies it we
    trust it and skip the scrape below, so one picture is not sent twice
    because it arrived by two routes.
    """
    collected: list[bytes] = []
    seen: set[str] = set()

    async def _add(ref: str) -> None:
        if not ref or len(collected) >= MAX_REFERENCE_IMAGES:
            return
        try:
            raw = await _reference_bytes(ref)
        except Exception as e:
            logger.error("Failed to resolve image reference: %s", type(e).__name__)
            return
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen:
            return
        seen.add(digest)
        collected.append(raw)

    if reference_urls:
        for ref in reference_urls:
            await _add(ref)
        return collected

    for source in (reply_msg, message):
        for att in getattr(source, "attachments", None) or []:
            ctype = getattr(att, "content_type", None)
            if ctype and ctype.startswith("image/"):
                await _add(att.url)
        for embed in getattr(source, "embeds", None) or []:
            image = getattr(embed, "image", None)
            if image and getattr(image, "url", None):
                await _add(image.url)
    for url in re.findall(
        r"(https?://\S+\.(?:png|jpg|jpeg|webp|gif))", prompt or "", re.IGNORECASE
    ):
        await _add(url)
    return collected


def _reference_streams(raw_images: list[bytes]) -> list[BytesIO]:
    """Fresh, named file objects per call: an upload consumes the stream, and a
    fallback provider has to read the same pictures again."""
    streams = []
    for index, raw in enumerate(raw_images):
        stream = BytesIO(raw)
        stream.name = f"reference_{index + 1}.png"
        streams.append(stream)
    return streams


# Partial frames per render when a caller wants a progressive preview. Each
# one costs an attachment re-upload on the status message, so keep it low --
# this is a handful of edits across a ~40s render, not a per-frame animation.
GPT_IMAGE_PARTIALS = 2


async def _generate_gpt_image_streaming(
    prompt: str, background_type: str, partial_callback, references: list[bytes] | None = None
):
    """Stream the render so the caller can show the image resolving.

    Returns (result, b64_image). Raises on any streaming failure so the caller
    can fall back to a single-shot request.
    """
    # input_fidelity="high" is what keeps the supplied subject recognizable;
    # at the default the model treats the reference as loose inspiration and
    # the picture the user pointed at stops being the picture they get back.
    if references:
        stream = await get_openai_image_client().images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=_reference_streams(references),
            prompt=prompt,
            size="auto",
            quality="high",
            input_fidelity="high",
            stream=True,
            partial_images=GPT_IMAGE_PARTIALS,
        )
    else:
        stream = await get_openai_image_client().images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size="auto",
            background=background_type,
            quality="high",
            moderation="low",
            n=1,
            stream=True,
            partial_images=GPT_IMAGE_PARTIALS,
        )
    final = None
    b64_image = None
    async for event in stream:
        kind = getattr(event, "type", "")
        payload = getattr(event, "b64_json", None)
        if kind == "image_generation.partial_image" and payload:
            try:
                partial_callback(base64.b64decode(payload), getattr(event, "partial_image_index", 0))
            except Exception:
                logger.debug("partial image callback failed", exc_info=True)
        elif kind == "image_generation.completed":
            final = event
            b64_image = payload
    return final, b64_image


async def generate_gpt_image(
    prompt: str, partial_callback=None, references: list[bytes] | None = None
) -> Optional[BytesIO]:
    """Render `prompt` with gpt-image.

    With `references`, the pictures the user supplied are the subject, so the
    request goes to the edits endpoint where the model can see them. The
    text-only generate endpoint has nowhere to put them, and a prompt like
    "imagine this in an art museum" then reaches the model with nothing to
    resolve "this" against.
    """
    try:
        background_type = "transparent" if "transparent background" in prompt.lower() else "auto"
        result = None
        b64_image = None
        if partial_callback is not None:
            try:
                result, b64_image = await _generate_gpt_image_streaming(
                    prompt, background_type, partial_callback, references=references
                )
            except ImageModerationError:
                raise
            except Exception as stream_error:
                if "moderation_blocked" in str(stream_error):
                    raise
                logger.info("gpt-image streaming unavailable, falling back: %s", stream_error)
                result = None
                b64_image = None
        if b64_image:
            _record_image_usage(result, label="image_generation")
            logger.info(
                "gpt-image streamed %d bytes (prompt=%.80r)", len(b64_image) * 3 // 4, prompt
            )
            return BytesIO(base64.b64decode(b64_image))
        if references:
            result = await get_openai_image_client().images.edit(
                model=OPENAI_IMAGE_MODEL,
                image=_reference_streams(references),
                prompt=prompt,
                size="auto",
                quality="high",
                input_fidelity="high",
            )
        else:
            result = await get_openai_image_client().images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size="auto",
                background=background_type,
                quality="high",
                moderation="low",
                n=1,
            )
        _record_image_usage(result, label="image_generation")
        b64_image = result.data[0].b64_json if result and result.data else None
        if not b64_image:
            logger.warning("gpt-image returned no image data (prompt=%.80r, result=%r)", prompt, result)
            return None
        logger.info("gpt-image generated %d bytes (prompt=%.80r)", len(b64_image) * 3 // 4, prompt)
        return BytesIO(base64.b64decode(b64_image))
    except Exception as e:
        if "moderation_blocked" in str(e):
            logger.warning("gpt-image moderation block (prompt=%.80r)", prompt)
            raise ImageModerationError("OpenAI's safety system rejected this image prompt.") from e
        logger.exception("Error generating GPT image")
        return None


def _record_image_usage(result, *, label: str) -> None:
    """Ledger entry for a gpt-image call: token-based when the API reports
    usage, else a flat per-image estimate (OPENAI_IMAGE_COST_USD, default 0.06)."""
    try:
        import os

        from services import usage_costs

        if getattr(result, "usage", None) is not None:
            usage_costs.record_response(OPENAI_IMAGE_MODEL, result, label=label)
        else:
            usage_costs.record(
                OPENAI_IMAGE_MODEL,
                None,
                float(os.getenv("OPENAI_IMAGE_COST_USD", "0.06")),
                label=label,
            )
    except Exception:
        logger.warning("image usage recording failed", exc_info=True)


# "gemini imagine X", "gemini generate an image of X", "gemini draw a picture
# of Y", ... — match the routing prefix so it can be stripped from the actual
# image prompt.
_GEMINI_IMAGE_PREFIX_RE = re.compile(
    r"^gemini\s+(?:imagine|generate|create|draw|paint|make)\b"
    r"(?:\s+(?:an|a|the|me|us)\b)*"
    r"(?:\s+(?:image|picture|photo|pic|artwork|art|drawing|portrait|wallpaper|logo)\b)?"
    r"(?:\s+of\b|\s*:)?\s*",
    re.IGNORECASE,
)


async def handle_image_generation(
    message,
    prompt: str,
    reply_msg=None,
    retry_context: str = "",
    use_gemini: bool | None = None,
    provider_state: dict | None = None,
    partial_callback=None,
    reference_urls: list[str] | None = None,
) -> Optional[BytesIO]:
    """Render an image request, using whatever pictures came with it.

    `reference_urls` is what the Discord layer collected for this message. Both
    providers read the same resolved set -- OpenAI through the edits endpoint,
    Gemini through reference contents -- so an attached picture informs the
    render on either backend instead of only on Gemini.
    """

    def _set_provider(name: str, model: str) -> None:
        if provider_state is not None:
            provider_state["provider"] = name
            provider_state["model"] = model

    # Resolved once. Both providers read this same set, and a mid-flight
    # fallback must not download the pictures a second time.
    references: list[bytes] = []

    def _compose(text: str) -> str:
        composed = _compose_reply_aware_image_prompt(text, reply_msg, retry_context)
        return _with_reference_note(composed) if references else composed

    async def _gemini_render(image_prompt: str, width: int, height: int) -> Optional[BytesIO]:
        if references:
            img = await asyncio.to_thread(
                generate_gemini_with_references, image_prompt, _reference_streams(references)
            )
            if img:
                return img
        return await asyncio.to_thread(generate_gemini_image, image_prompt, width, height)

    async def _openai_then_gemini(image_prompt: str, width: int, height: int) -> Optional[BytesIO]:
        """Try OpenAI once, then always try Gemini for any OpenAI failure."""
        _set_provider("OpenAI", IMG_MODEL_OPENAI)
        try:
            img = await generate_gpt_image(
                image_prompt, partial_callback=partial_callback, references=references
            )
        except ImageModerationError:
            img = None
        if img:
            return img

        if message:
            await message.channel.send("⚠️ OpenAI image generation failed — trying **Gemini** instead…")
        _set_provider("Gemini", IMG_MODEL_GEMINI)
        img = await _gemini_render(image_prompt, width, height)
        if img:
            return img
        if message:
            await message.channel.send("❌ Gemini image generation failed too. Try again or rephrase the prompt.")
        return None

    try:
        references = await collect_reference_images(
            message=message,
            reply_msg=reply_msg,
            prompt=prompt,
            reference_urls=reference_urls,
        )
        prompt_with_reply_context = _compose(prompt)
        width, height = extract_width_height_from_prompt(prompt_with_reply_context)
        if prompt.lower().startswith("stable imagine"):
            image_prompt = _compose(prompt[15:].strip())
            # Stability takes no reference image. When the user supplied one,
            # only the OpenAI/Gemini path can honor it, so skip Stability
            # rather than silently render without the picture.
            if STABILITY_AVAILABLE and not references:
                _set_provider("Stability", IMG_MODEL_STABILITY)
                img = await generate_stability_image(image_prompt, width, height)
                if img:
                    return img
            return await _openai_then_gemini(image_prompt, width, height)

        # Provider selection is passed in by the dispatcher (the user said
        # "gemini" somewhere); the prefix regex is only used to strip routing
        # words like "gemini generate an image of" from the actual prompt.
        gemini_prefix = _GEMINI_IMAGE_PREFIX_RE.match(prompt)
        if use_gemini is None:
            use_gemini = bool(gemini_prefix)
        if use_gemini:
            if gemini_prefix:
                core_prompt = prompt[gemini_prefix.end():].strip()
            else:
                core_prompt = re.sub(r"^gemini[\s,:]*", "", prompt, flags=re.IGNORECASE).strip() or prompt
            image_prompt = _compose(core_prompt)
            _set_provider("Gemini", IMG_MODEL_GEMINI)
            img = await _gemini_render(image_prompt, width, height)
            if img:
                return img
            if message:
                await message.channel.send("⚠️ **Gemini generation failed** (likely rate limit or error). Falling back to OpenAI... 🧠")
            _set_provider("OpenAI", IMG_MODEL_OPENAI)
            try:
                return await generate_gpt_image(
                    image_prompt, partial_callback=partial_callback, references=references
                )
            except ImageModerationError:
                if message:
                    await message.channel.send("🚫 OpenAI's safety system also rejected this prompt. Try rephrasing.")
                return None

        return await _openai_then_gemini(prompt_with_reply_context, width, height)
    except Exception:
        logger.exception("Error in handle_image_generation")
        return None


async def edit_image_with_prompt(image_input: str | list[str], prompt: str) -> Optional[BytesIO]:
    try:
        urls = [image_input] if isinstance(image_input, str) else image_input
        if not urls:
            return None

        async def decode_img(u):
            if u.startswith("data:image/"):
                _, b64 = u.split(",", 1)
                if len(b64) > (DEFAULT_MEDIA_BYTES * 4 // 3) + 16:
                    raise ValueError("image input exceeds the allowed size")
                decoded = base64.b64decode(b64, validate=True)
                if len(decoded) > DEFAULT_MEDIA_BYTES:
                    raise ValueError("image input exceeds the allowed size")
                return BytesIO(decoded)
            if u.startswith("http"):
                fetched = await fetch_url_bytes_async(
                    u,
                    timeout=30,
                    max_bytes=DEFAULT_MEDIA_BYTES,
                    allowed_content_types=("image/",),
                )
                return BytesIO(fetched.body)
            if len(u) > (DEFAULT_MEDIA_BYTES * 4 // 3) + 16:
                raise ValueError("image input exceeds the allowed size")
            decoded = base64.b64decode(u, validate=True)
            if len(decoded) > DEFAULT_MEDIA_BYTES:
                raise ValueError("image input exceeds the allowed size")
            return BytesIO(decoded)

        if prompt.lower().startswith("gemini edit"):
            base_img = await decode_img(urls[0])
            return edit_gemini_image(base_img, prompt[11:].strip())

        base_img = await decode_img(urls[0])
        result = await get_openai_image_client().images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=base_img,
            prompt=prompt,
            size="auto",
        )
        _record_image_usage(result, label="image_edit")
        b64_image = result.data[0].b64_json if result and result.data else None
        if not b64_image:
            return None
        return BytesIO(base64.b64decode(b64_image))
    except Exception:
        logger.exception("Error editing image")
        return None
