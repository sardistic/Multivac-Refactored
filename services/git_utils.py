"""
Git utilities for bot self-awareness.
Allows the bot to answer questions about its own code and recent commits.

Security features:
- File blocklist to prevent reading sensitive files
- Content redaction to scrub API key patterns
- Output truncation to prevent huge dumps
"""

import subprocess
import re
import os
from typing import List, Dict, Any, Optional
from fnmatch import fnmatch

# Repository root
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- Security: File Blocklist ----
BLOCKED_PATTERNS = [
    ".env", ".env.*", "*.pem", "*.key", "*.crt",
    "secrets.*", "*secret*", "*credential*",
    "*.sqlite", "*.db",
]

def _is_blocked_file(path: str) -> bool:
    """Check if file matches any blocked pattern."""
    basename = os.path.basename(path)
    for pattern in BLOCKED_PATTERNS:
        if fnmatch(basename.lower(), pattern.lower()):
            return True
    return False


def _is_internal_tool_file(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith("services/git_utils.py") or normalized == "git_utils.py"

# ---- Security: Content Redaction ----
REDACT_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_OPENAI_KEY]"),
    (r"sk-proj-[a-zA-Z0-9_-]{80,}", "[REDACTED_OPENAI_KEY]"),
    (r"AIza[a-zA-Z0-9_-]{35}", "[REDACTED_GOOGLE_KEY]"),
    (r"ghp_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),
    (r"ghu_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "[REDACTED_SLACK_TOKEN]"),
    (r"[a-f0-9]{64}", "[REDACTED_HEX_TOKEN]"),  # Generic 64-char hex
    (r"Bearer [a-zA-Z0-9_-]{20,}", "[REDACTED_BEARER]"),
    (r"token[\"']?\s*[:=]\s*[\"'][^\"']{20,}[\"']", "[REDACTED_TOKEN_ASSIGNMENT]"),
]

API_CALL_PATTERNS = {
    "openai_responses": r"responses\.create",
    "openai_chat": r"chat\.completions\.create",
    "openai_http": r"api\.openai\.com|/v1/(chat/completions|responses|images|video|videos|sora)",
    "gemini_client": r"genai\.Client|google\.genai",
    "anthropic_client": r"AsyncAnthropic|anthropic\.",
    "sora_http": r"sora_headers|/videos?/|/sora/",
    "stability_sdk": r"stability_client\.StabilityInference|stability_sdk",
    "requests_http": r"requests\.(get|post|put|delete|patch)",
}

def _redact_secrets(text: str) -> str:
    """Scrub potential API keys and tokens from text."""
    for pattern, replacement in REDACT_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

# ---- Git Commands ----

def _run_git(*args, max_output: int = 8000) -> str:
    """Run a git command and return output (truncated if needed)."""
    try:
        # Use -c safe.directory=* to avoid ownership errors in automated environments
        cmd = ["git", "-c", "safe.directory=*"] + list(args)
        result = subprocess.run(
            cmd,
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout or result.stderr or ""
        if result.returncode != 0:
            return f"[error: git command failed with code {result.returncode}: {output.strip()}]"
            
        if len(output) > max_output:
            output = output[:max_output] + f"\n... [truncated, {len(output)} total chars]"
        return _redact_secrets(output)
    except subprocess.TimeoutExpired:
        return "[error: git command timed out]"
    except Exception as e:
        return f"[error: {e}]"


def get_recent_commits(n: int = 10) -> List[Dict[str, str]]:
    """Get recent commits with SHA, message, author, and relative date."""
    n = min(n, 50)  # Cap at 50
    output = _run_git("log", f"-n{n}", "--pretty=format:%h|%s|%an|%ar")
    
    if output.startswith("[error:"):
        return [{"sha": "error", "message": output, "author": "system", "date": "now"}]
        
    commits = []
    for line in output.strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "sha": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
    return commits


def get_commit_diff(sha: str) -> str:
    """Get the diff for a specific commit. SHA is validated."""
    # Validate SHA format (only alphanumeric)
    if not re.match(r"^[a-f0-9]{4,40}$", sha, re.IGNORECASE):
        return "[error: invalid SHA format]"
    
    output = _run_git("show", "--stat", sha, max_output=6000)
    # Get the actual patch using a command that works across older Git builds.
    diff = _run_git("diff", f"{sha}^!", "--", max_output=4000)
    
    return f"{output}\n\n--- Diff ---\n{diff}"


def get_file_content(path: str, max_lines: int = 200) -> str:
    """Read a file from the repo. Blocked files return error."""
    # Security check
    if _is_blocked_file(path):
        return f"[error: access to '{path}' is blocked for security]"
    
    # Normalize path (prevent directory traversal)
    path = path.lstrip("/").lstrip("\\")
    if ".." in path:
        return "[error: invalid path]"
    
    full_path = os.path.join(REPO_PATH, path)
    if not os.path.isfile(full_path):
        return f"[error: file '{path}' not found]"
    
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            content = "".join(lines[:max_lines])
            content += f"\n... [truncated, {len(lines)} total lines]"
        else:
            content = "".join(lines)
        
        return _redact_secrets(content)
    except Exception as e:
        return f"[error reading file: {e}]"


def search_code(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search code using git grep. Returns matching lines with context."""
    if not query or len(query) < 2:
        return [{"error": "query too short"}]
    
    # Escape special regex chars for literal search
    escaped = re.escape(query)
    
    output = _run_git("grep", "-n", "-i", "--", escaped, max_output=10000)
    
    results = []
    for line in output.strip().split("\n")[:max_results]:
        if ":" in line:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                file_path = parts[0]
                if _is_blocked_file(file_path) or _is_internal_tool_file(file_path):
                    continue  # Skip blocked files
                results.append({
                    "file": file_path,
                    "line": parts[1],
                    "content": _redact_secrets(parts[2][:200]),
                })
    
    return results


def search_history(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search git history for a regex/pattern and return matching commits/files."""
    if not query or len(query) < 2:
        return [{"error": "query too short"}]

    max_results = max(1, min(max_results, 20))
    output = _run_git(
        "log",
        "--all",
        f"-G{query}",
        "--pretty=format:__COMMIT__%H|%s|%an|%ar",
        "--name-only",
        max_output=12000,
    )

    if output.startswith("[error:"):
        return [{"error": output}]

    results: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("__COMMIT__"):
            if current:
                results.append(current)
                if len(results) >= max_results:
                    break
            payload = line[len("__COMMIT__") :]
            parts = payload.split("|", 3)
            if len(parts) == 4:
                current = {
                    "sha": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                    "files": [],
                }
            else:
                current = None
            continue
        if current is not None and not _is_blocked_file(line):
            current["files"].append(line)

    if current and len(results) < max_results:
        results.append(current)

    return results


def _grep_regex(pattern: str, max_results: int = 20) -> List[Dict[str, Any]]:
    output = _run_git("grep", "-n", "-i", "-E", "--", pattern, max_output=12000)
    if output.startswith("[error:"):
        return [{"error": output}]

    results: List[Dict[str, Any]] = []
    for line in output.strip().split("\n")[:max_results]:
        if ":" not in line:
            continue
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path = parts[0]
        if _is_blocked_file(file_path) or _is_internal_tool_file(file_path):
            continue
        results.append(
            {
                "file": file_path,
                "line": parts[1],
                "content": _redact_secrets(parts[2][:200]),
            }
        )
    return results


def find_api_calls(provider: str | None = None, max_results: int = 20) -> Dict[str, Any]:
    """Find API/SDK call sites in current code and relevant history."""
    provider = (provider or "").strip().lower()
    patterns = {
        name: regex
        for name, regex in API_CALL_PATTERNS.items()
        if not provider or provider in name
    }
    if not patterns:
        patterns = API_CALL_PATTERNS

    current_matches: List[Dict[str, Any]] = []
    history_matches: List[Dict[str, Any]] = []

    for name, pattern in patterns.items():
        for match in _grep_regex(pattern, max_results=max_results):
            if "error" in match:
                continue
            current_matches.append({"pattern": name, **match})
            if len(current_matches) >= max_results:
                break
        if len(current_matches) >= max_results:
            break

    seen_shas = set()
    per_pattern_history = max(1, min(3, max_results // max(1, len(patterns))))
    for name, pattern in patterns.items():
        for hit in search_history(pattern, max_results=per_pattern_history):
            sha = hit.get("sha")
            if not sha or sha in seen_shas or "error" in hit:
                continue
            seen_shas.add(sha)
            history_matches.append({"pattern": name, **hit})
            if len(history_matches) >= max_results:
                break
        if len(history_matches) >= max_results:
            break

    return {
        "ok": True,
        "provider": provider or "all",
        "current_matches": current_matches[:max_results],
        "history_matches": history_matches[:max_results],
    }


def get_file_list() -> List[str]:
    """Get list of all tracked files in the repo."""
    output = _run_git("ls-files")
    files = [f for f in output.strip().split("\n") if f and not _is_blocked_file(f)]
    return files


def get_repo_info() -> Dict[str, str]:
    """Get basic repo info: current branch, remote URL, last commit."""
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD").strip()
    remote = _run_git("remote", "get-url", "origin").strip()
    last_commit = _run_git("log", "-1", "--pretty=format:%h - %s (%ar)").strip()
    
    return {
        "branch": branch,
        "remote": remote,
        "last_commit": last_commit,
    }
