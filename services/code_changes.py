"""Policy and static validation for owner-reviewed local code proposals.

Patches are never applied to the live checkout. Validation uses a temporary
archive of the recorded baseline commit and never imports or executes proposed
Python code.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from services.git_utils import REPO_PATH

MAX_PATCH_BYTES = 256_000
MAX_CHANGED_FILES = 20

PROTECTED_PATTERNS = (
    ".env",
    ".env.*",
    ".git/*",
    ".github/*",
    "*.db",
    "*.sqlite",
    "*.key",
    "*.pem",
    "*secret*",
    "*credential*",
    "config.py",
    "main.py",
    "pyproject.toml",
    "poetry.lock",
    "services/code_changes.py",
    "services/sqlite_store.py",
    "services/database_utils.py",
    "services/git_utils.py",
    "services/tool_runtime.py",
    "services/tool_control.py",
    "services/tools_registry.py",
    "services/tool_dispatch.py",
    "services/command_runtime.py",
    "services/command_control.py",
    "services/behavior_runtime.py",
    "services/behavior_registry.py",
    "services/behavior_control.py",
    "dev/validate_tool_modules.py",
    "dev/validate_command_modules.py",
    "dev/validate_behavior_modules.py",
    "ops/*",
    "tests/*",
    "dashboard/*",
    "website/*",
    "services/portal.py",
    "services/portal_store.py",
    "services/portal_usage.py",
    "bot/invite_approval.py",
)


def is_protected_path(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    return any(fnmatch.fnmatch(low, pattern.lower()) for pattern in PROTECTED_PATTERNS)


def get_baseline_sha() -> str:
    branch = os.environ.get("MULTIVAC_CANONICAL_BRANCH", "main").strip()
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch)
        or ".." in branch
        or branch.endswith(("/", "."))
    ):
        raise RuntimeError("MULTIVAC_CANONICAL_BRANCH is not a valid branch name")
    canonical_ref = f"refs/heads/{branch}^{{commit}}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", canonical_ref],
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or f"Unable to resolve canonical Git baseline {branch}").strip()
        )
    sha = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("Git returned an invalid baseline SHA")
    return sha


def _normalize_patch_path(raw: str) -> str | None:
    raw = raw.strip().split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    raw = raw.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or ".." in path.parts:
        raise ValueError(f"Unsafe patch path: {raw!r}")
    return str(path)


def inspect_patch(patch: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    paths: set[str] = set()
    size = len(patch.encode("utf-8", errors="replace"))

    if not patch.strip():
        errors.append("Patch is empty")
    if size > MAX_PATCH_BYTES:
        errors.append(f"Patch exceeds {MAX_PATCH_BYTES:,} bytes")
    if "GIT binary patch" in patch or "Binary files " in patch:
        errors.append("Binary patches are not accepted")
    if re.search(r"^(?:new|old) file mode (?:120000|160000)$", patch, re.MULTILINE):
        errors.append("Symlink and submodule changes are not accepted")

    for line in patch.splitlines():
        raw = None
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+) b/(.+)$", line)
            if not match:
                errors.append(f"Malformed diff header: {line[:120]}")
                continue
            for candidate in match.groups():
                try:
                    normalized = _normalize_patch_path(candidate)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if normalized:
                    paths.add(normalized)
        elif line.startswith(("+++ ", "--- ")):
            raw = line[4:]
        if raw is not None:
            try:
                normalized = _normalize_patch_path(raw)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if normalized:
                paths.add(normalized)

    if not paths and patch.strip():
        errors.append("No changed files could be read from the unified diff")
    if len(paths) > MAX_CHANGED_FILES:
        errors.append(f"Patch changes {len(paths)} files; maximum is {MAX_CHANGED_FILES}")

    for path in sorted(paths):
        low = path.lower()
        if is_protected_path(path):
            errors.append(f"Protected path cannot be changed: {path}")
        if low.startswith(("archive/", "scripts/")):
            warnings.append(f"Review executable or archived path carefully: {path}")

    live_paths = [
        path
        for path in paths
        if path.lower().startswith(("live_tools/", "live_commands/", "live_components/"))
    ]
    if live_paths and len(live_paths) != len(paths):
        errors.append("Live modules must be proposed separately from all other changes")
    live_roots = {PurePosixPath(path).parts[0].lower() for path in live_paths}
    if len(live_roots) > 1:
        errors.append("Live tool, command, and behavior modules require separate proposals")
    for path in live_paths:
        normalized = PurePosixPath(path)
        if normalized.suffix.lower() != ".py" or normalized.name == "__init__.py":
            errors.append(f"Live artifact must be one standalone .py module: {path}")

    return {
        "ok": not errors,
        "files": sorted(paths),
        "errors": errors,
        "warnings": warnings,
        "patch_bytes": size,
    }


def validate_patch(baseline_sha: str, patch: str) -> dict[str, Any]:
    report = inspect_patch(patch)
    report["baseline_sha"] = baseline_sha
    report["syntax_checked"] = []
    if not report["ok"]:
        return report
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_sha or "", re.IGNORECASE):
        report["errors"].append("Invalid baseline SHA")
        report["ok"] = False
        return report

    with tempfile.TemporaryDirectory(prefix="multivac-proposal-") as tmp:
        root = Path(tmp)
        archive = root / "baseline.tar"
        snapshot = root / "snapshot"
        snapshot.mkdir()
        patch_path = root / "proposal.patch"
        # Preserve LF bytes on Windows; newline translation makes Git treat
        # every added line's CR as trailing whitespace.
        patch_path.write_bytes(patch.replace("\r\n", "\n").encode("utf-8"))

        archived = subprocess.run(
            ["git", "archive", "--format=tar", "-o", str(archive), baseline_sha],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if archived.returncode != 0:
            report["errors"].append((archived.stderr or "Could not archive baseline").strip())
            report["ok"] = False
            return report
        with tarfile.open(archive, "r") as bundle:
            bundle.extractall(snapshot, filter="data")

        applied = subprocess.run(
            # LLMs occasionally miscount the old/new line totals in an otherwise
            # valid hunk header. Git can safely recompute those totals while still
            # rejecting bad context, malformed patches, and whitespace errors.
            ["git", "apply", "--recount", "--whitespace=error", str(patch_path)],
            cwd=snapshot,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if applied.returncode != 0:
            report["errors"].append((applied.stderr or "Patch does not apply cleanly").strip()[:1000])
            report["ok"] = False
            return report

        for relative in report["files"]:
            if not relative.endswith(".py"):
                continue
            source_path = snapshot / relative
            if not source_path.is_file():
                continue
            try:
                source = source_path.read_text(encoding="utf-8")
                ast.parse(source, filename=relative)
                report["syntax_checked"].append(relative)
            except (OSError, UnicodeError, SyntaxError) as exc:
                report["errors"].append(f"Python syntax failed for {relative}: {exc}")

    report["ok"] = not report["errors"]
    return report
