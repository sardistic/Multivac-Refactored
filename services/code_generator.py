"""Bounded, owner-invoked patch generation for code proposals."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import PurePosixPath

from providers.claude_utils import CLAUDE_MODEL, generate_claude_response
from providers.openai_client import OPENAI_CODE_MODEL, OPENAI_LIGHT_MODEL
from providers.openai_messages import generate_openai_messages_response_with_tools
from services.code_changes import MAX_PATCH_BYTES, REPO_PATH, inspect_patch, is_protected_path

MAX_CONTEXT_CHARS = 70_000
MAX_FILES = 14
CODE_MODEL = OPENAI_CODE_MODEL
IDEA_CODE_MODEL = os.getenv("OPENAI_IDEA_CODE_MODEL", OPENAI_LIGHT_MODEL)
_CLAUDE_PROVIDER_RE = re.compile(
    r"^\s*(?:(?:hey\s+)?(?:claude|fable)\b|"
    r"(?:use|using|with|via)\s+(?:claude|fable)\b|"
    r"(?:ask|have|let)\s+(?:claude|fable)\b)",
    re.IGNORECASE,
)


def select_code_generation_provider(request: str) -> str:
    """Honor an explicit Claude/Fable directive; otherwise use the OpenAI code model."""
    return "claude" if _CLAUDE_PROVIDER_RE.search(request or "") else "openai"


def code_generation_model(request: str) -> str:
    return CLAUDE_MODEL if select_code_generation_provider(request) == "claude" else CODE_MODEL


def _request_without_provider_directive(request: str) -> str:
    text = request or ""
    patterns = (
        r"^\s*(?:hey\s+)?(?:claude|fable)\s*[:,]?\s*",
        r"^\s*(?:use|using|with|via)\s+(?:claude|fable)\s+(?:to\s+)?",
        r"^\s*(?:ask|have|let)\s+(?:claude|fable)\s+(?:to\s+)?",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        if cleaned != text:
            return cleaned.strip() or text.strip()
    return text.strip()


async def _generate_code_response(
    messages: list[dict], *, provider: str, max_tokens: int, model_override: str | None = None
) -> tuple[str, str]:
    if provider == "claude":
        if model_override:
            raise ValueError("OpenAI model override cannot be used with Claude")
        response = await generate_claude_response(
            messages,
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        if response.startswith("❌"):
            raise RuntimeError(response)
        return response, CLAUDE_MODEL
    selected_model = model_override or CODE_MODEL
    response = await generate_openai_messages_response_with_tools(
        messages,
        tools=[],
        model=selected_model,
        max_tokens=max_tokens,
        temperature=0.1,
        reasoning_effort="high",
    )
    if response.startswith("⚠️ OpenAI"):
        raise RuntimeError(response)
    return response, selected_model


def _git_at(baseline: str, *args: str, max_chars: int = 100_000) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_PATH, capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "Git context lookup failed").strip()[:500])
    return result.stdout[:max_chars]


def _request_terms(request: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z][a-z0-9_]{2,}", request.lower())
        if token not in {"the", "and", "that", "with", "from", "this", "make", "change", "code", "bot"}
    }


def _candidate_paths(baseline: str) -> list[str]:
    tracked = _git_at(baseline, "ls-tree", "-r", "--name-only", baseline).splitlines()
    return [
        path for path in tracked
        if PurePosixPath(path).suffix in {".py", ".md", ".toml", ".yml", ".yaml"}
        and not is_protected_path(path)
        and not path.startswith(("archive/", "dev/"))
    ]


def select_code_context(request: str, baseline: str) -> list[tuple[str, str]]:
    """Lexical fallback used only when model-driven repository planning fails."""
    candidates = _candidate_paths(baseline)
    terms = _request_terms(request)
    ranked: list[tuple[int, str, str]] = []
    for path in candidates:
        try:
            content = _git_at(baseline, "show", f"{baseline}:{path}", max_chars=25_000)
        except RuntimeError:
            continue
        low = f"{path}\n{content}".lower()
        score = sum((8 if term in path.lower() else 1) * low.count(term) for term in terms)
        if path in {"readme.md", "discord_bot.py"}:
            score += 2
        ranked.append((score, path, content))
    ranked.sort(key=lambda item: (-item[0], len(item[2]), item[1]))

    selected: list[tuple[str, str]] = []
    used = 0
    for _score, path, content in ranked:
        block_size = len(path) + len(content)
        if selected and (len(selected) >= MAX_FILES or used + block_size > MAX_CONTEXT_CHARS):
            continue
        selected.append((path, content))
        used += block_size
        if len(selected) >= MAX_FILES:
            break
    return selected


def _architecture_index(baseline: str, candidates: list[str]) -> str:
    """Build a compact, request-independent map of ownership and call-path clues."""
    blocks = []
    used = 0
    structural = re.compile(
        r"^\s*(?:class |def |async def |from |import )|"
        r"system|prompt|persona|personality|context|message|handler|dispatch|provider",
        re.IGNORECASE,
    )
    for path in candidates:
        try:
            content = _git_at(baseline, "show", f"{baseline}:{path}", max_chars=25_000)
        except RuntimeError:
            continue
        clues = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(content.splitlines(), 1)
            if structural.search(line)
        ][:35]
        block = f"[{path}]\n" + "\n".join(clues)
        if used + len(block) > 45_000:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


async def plan_code_context(
    request: str, baseline: str, *, provider: str = "openai"
) -> tuple[list[tuple[str, str]], dict]:
    candidates = _candidate_paths(baseline)
    repo_map = "\n".join(candidates)
    architecture = _architecture_index(baseline, candidates)
    messages = [
        {
            "role": "system",
            "content": (
                "You are planning repository scope for a code change. Reason about architecture, "
                "ownership, call paths, and the user's intended breadth; do not select files merely "
                "because they repeat request words. Distinguish a global behavior owner from a local "
                "handler that happens to contain similar prose. Return only JSON with keys files "
                "(an ordered array of 1-14 exact paths), rationale, and scope_check. Include the source "
                "owners and their important consumers when the request is cross-cutting."
            ),
        },
        {
            "role": "user",
            "content": (
                f"REQUEST: {request}\nBASELINE: {baseline}\n\n"
                f"ALLOWED REPOSITORY MAP:\n{repo_map}\n\n"
                f"ARCHITECTURE INDEX:\n{architecture}"
            ),
        },
    ]
    response, planning_model = await _generate_code_response(
        messages, provider=provider, max_tokens=2500
    )
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
    plan = json.loads(cleaned)
    requested_paths = plan.get("files")
    if not isinstance(requested_paths, list):
        raise ValueError("Repository planner returned no file list")
    allowed = set(candidates)
    selected = []
    used = 0
    for raw_path in requested_paths[:MAX_FILES]:
        path = str(raw_path)
        if path not in allowed or any(existing == path for existing, _ in selected):
            continue
        content = _git_at(baseline, "show", f"{baseline}:{path}", max_chars=25_000)
        if selected and used + len(content) + len(path) > MAX_CONTEXT_CHARS:
            continue
        selected.append((path, content))
        used += len(content) + len(path)
    if not selected:
        raise ValueError("Repository planner selected no policy-allowed files")
    return selected, {
        "selection": "model_planner",
        "planning_model": planning_model,
        "rationale": str(plan.get("rationale") or "")[:1000],
        "scope_check": str(plan.get("scope_check") or "")[:1000],
    }


def extract_unified_diff(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("diff --git ")
    if start < 0:
        raise ValueError("Model did not return a unified Git diff")
    patch = text[start:].strip() + "\n"
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ValueError("Generated patch exceeds the patch-size limit")
    report = inspect_patch(patch)
    if not report["ok"]:
        raise ValueError("Generated patch violated policy: " + "; ".join(report["errors"][:4]))
    return patch


async def generate_code_patch(request: str, baseline: str) -> tuple[str, dict]:
    provider = select_code_generation_provider(request)
    generation_request = _request_without_provider_directive(request)
    try:
        context, planning = await plan_code_context(
            generation_request, baseline, provider=provider
        )
    except (RuntimeError, ValueError, json.JSONDecodeError):
        context = select_code_context(generation_request, baseline)
        planning = {"selection": "lexical_fallback"}
    if not context:
        raise RuntimeError("No policy-allowed source context was found")
    source = "\n\n".join(
        f"===== FILE: {path} =====\n{content}" for path, content in context
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are generating a minimal code change for a reviewed deployment pipeline. "
                "Return ONLY one valid unified Git diff beginning with 'diff --git'. Do not use "
                "markdown fences or commentary. Change only files supplied below. Never modify "
                "tests, ops, secrets, configuration, dependency manifests, startup files, audit "
                "storage, or self-modification policy. Do not add dependencies. Preserve existing "
                "behavior outside the request."
            ),
        },
        {
            "role": "user",
            "content": (
                f"BASELINE SHA: {baseline}\nREQUEST: {generation_request}\n"
                f"REPOSITORY SCOPE PLAN: {planning}\n\n"
                f"AVAILABLE SOURCE FILES:\n{source}"
            ),
        },
    ]
    response, generation_model = await _generate_code_response(
        messages,
        provider=provider,
        max_tokens=8000,
    )
    patch = extract_unified_diff(response)
    return patch, {
        "provider": provider,
        "model": generation_model,
        "context_files": [path for path, _ in context],
        "context_chars": sum(len(content) for _, content in context),
        **planning,
    }


async def generate_planned_idea_patch(
    request: str,
    baseline: str,
    *,
    context_paths: list[str] | None = None,
) -> tuple[str, dict]:
    """Generate an owner-prompted idea patch with the low-cost coding tier.

    Expensive planning has already happened in the background. This step may
    draft code, but it still passes the ordinary proposal policy, validation,
    owner approval, and deployment gates.
    """
    candidates = set(_candidate_paths(baseline))
    context: list[tuple[str, str]] = []
    used = 0
    for path in list(context_paths or [])[:MAX_FILES]:
        if path not in candidates:
            continue
        content = _git_at(baseline, "show", f"{baseline}:{path}", max_chars=25_000)
        if context and used + len(path) + len(content) > MAX_CONTEXT_CHARS:
            continue
        context.append((path, content))
        used += len(path) + len(content)
    if not context:
        context = select_code_context(request, baseline)[:6]
    if not context:
        raise RuntimeError("No policy-allowed source context was found")

    source = "\n\n".join(
        f"===== FILE: {path} =====\n{content}" for path, content in context
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are implementing an already-reviewed architectural plan with a small, low-cost "
                "coding pass. Return ONLY one valid unified Git diff beginning with 'diff --git'. "
                "Do not use markdown fences or commentary. Prefer the supplied source files. A new "
                "standalone live_tools, live_commands, or live_components Python file is allowed only "
                "when the request explicitly selects that hotload kind. Never modify tests, ops, "
                "secrets, configuration, dependency manifests, startup files, audit storage, or "
                "self-modification policy. Do not add dependencies. Preserve unrelated behavior."
            ),
        },
        {
            "role": "user",
            "content": (
                f"BASELINE SHA: {baseline}\nREVIEWED IDEA: {request}\n\n"
                f"AVAILABLE SOURCE CONTEXT:\n{source}"
            ),
        },
    ]
    response, model = await _generate_code_response(
        messages,
        provider="openai",
        model_override=IDEA_CODE_MODEL,
        max_tokens=6000,
    )
    patch = extract_unified_diff(response)
    return patch, {
        "provider": "openai",
        "model": model,
        "selection": "reflection_plan",
        "context_files": [path for path, _ in context],
        "context_chars": sum(len(content) for _, content in context),
    }
