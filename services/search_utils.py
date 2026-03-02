# search_utils.py
# Web search utilities using Google Programmable Search (CSE),
# with credentials resolved from:
#   1) ENV
#   2) GCE instance metadata (instance/attributes/*)
#   3) GCE project metadata (project/attributes/*)
# Discovered values are mirrored back into os.environ so other modules see them.

from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Any, Optional

import httpx

log = logging.getLogger("search_utils")

# -----------------------------------------------------------------------------
# Metadata helpers (works even if config.py isn't imported anywhere else)
# -----------------------------------------------------------------------------

_MD_BASE = "http://metadata.google.internal/computeMetadata/v1"
_MD_HEADERS = {"Metadata-Flavor": "Google"}

def _md_fetch(path: str, timeout: float = 0.8) -> Optional[str]:
    try:
        with httpx.Client(timeout=timeout, headers=_MD_HEADERS) as s:
            r = s.get(f"{_MD_BASE}/{path}")
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except Exception:
        pass
    return None

def _get_from_metadata(key: str) -> Optional[str]:
    # instance attribute first
    v = _md_fetch(f"instance/attributes/{key}")
    if v:
        return v
    # project attribute next
    v = _md_fetch(f"project/attributes/{key}")
    if v:
        return v
    return None

def _mirror_env(k: str, v: Optional[str]) -> Optional[str]:
    if v and not os.getenv(k):
        os.environ[k] = v
    return v

def _resolve_credential(*names: str) -> Optional[str]:
    """
    Try ENV first; if missing, try GCE metadata for each name; when found, mirror into env.
    Accepts multiple names to support both GOOGLE_API_KEY and GOOGLE_SEARCH_API_KEY, etc.
    """
    # 1) ENV
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v

    # 2) metadata (instance → project)
    for n in names:
        v = _get_from_metadata(n)
        if v:
            return _mirror_env(n, v)

    return None

# -----------------------------------------------------------------------------
# Config (now backed by metadata resolution)
# -----------------------------------------------------------------------------

# Support both naming schemes:
# - GOOGLE_API_KEY / GOOGLE_CSE_ID  (as in config.py)
# - GOOGLE_SEARCH_API_KEY / GOOGLE_SEARCH_CX  (as in your previous file)
_API_KEY = _resolve_credential("GOOGLE_SEARCH_API_KEY", "GOOGLE_API_KEY")
_CX      = _resolve_credential("GOOGLE_SEARCH_CX", "GOOGLE_CSE_ID")

_SAFE = (os.getenv("GOOGLE_SEARCH_SAFE") or "off").strip().lower()  # "off" | "active"
_DEFAULT_NUM = max(1, min(int(os.getenv("GOOGLE_SEARCH_NUM", "4")), 10))
_HTTP_TIMEOUT = float(os.getenv("SEARCH_HTTP_TIMEOUT", "8.0"))
_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
_UA = "DiscordBot/1.0 (+google-cse)"

if not _API_KEY or not _CX:
    # Log once at import — makes misconfig obvious in journalctl
    log.warning(
        "Google CSE credentials not fully resolved "
        "(API_KEY=%s, CX=%s). Will return empty results until set.",
        bool(_API_KEY), bool(_CX)
    )

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _normalize_items(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for it in items or []:
        title = (it.get("title") or "").strip()
        link = (it.get("link") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        if title and link:
            out.append({"title": title, "url": link, "snippet": snippet})
    return out

def _google_cse(
    query: str,
    *,
    site: Optional[str],
    num: int,
    gl: Optional[str],
    lr: Optional[str],
    safe: str,
    api_key: Optional[str],
    cx: Optional[str],
    retries: int = 2,
    backoff_base: float = 0.6,
) -> List[Dict[str, str]]:
    """
    Call Google CSE with minimal fields.
    Retries a couple times on 429/5xx. Returns a normalized list of results.
    """
    key = (api_key or _API_KEY)
    search_cx = (cx or _CX)

    if not key or not search_cx:
        # Be explicit in logs so this is easy to diagnose
        if not key:
            log.error("web_search: missing GOOGLE_API_KEY/GOOGLE_SEARCH_API_KEY (metadata/env)")
        if not search_cx:
            log.error("web_search: missing GOOGLE_CSE_ID/GOOGLE_SEARCH_CX (metadata/env)")
        return []

    params = {
        "key": key,
        "cx": search_cx,
        "q": f"site:{site} {query}" if site else query,
        "num": max(1, min(int(num or _DEFAULT_NUM), 10)),
        "safe": "active" if safe == "active" else "off",
        "fields": "items(title,link,snippet)",
    }
    if gl:
        params["gl"] = gl  # country bias (e.g., "us", "fr")
    if lr:
        params["lr"] = lr  # language restrict (e.g., "lang_en")

    headers = {
        "User-Agent": _UA,
        "Accept": "application/json",
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            with httpx.Client(timeout=_HTTP_TIMEOUT, headers=headers, follow_redirects=True) as s:
                r = s.get(_GOOGLE_CSE_URL, params=params)

            # Retry on transient issues
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt <= retries:
                    sleep_for = backoff_base * (2 ** (attempt - 1))
                    time.sleep(sleep_for)
                    continue

            if r.status_code != 200:
                try:
                    body = r.json()
                except Exception:
                    body = {"raw": r.text[:400]}
                log.error("web_search: CSE HTTP %s: %s", r.status_code, body)
                return []

            data = r.json()
            return _normalize_items(data.get("items") or [])

        except httpx.HTTPStatusError as e:
            log.error("web_search: HTTPStatusError %s", e)
            return []
        except Exception as e:
            log.exception("web_search: request failed: %s", e)
            return []

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def web_search(
    query: str,
    *,
    site: Optional[str] = None,
    max_results: Optional[int] = None,
    gl: Optional[str] = None,              # Country bias, e.g. "us", "fr"
    lr: Optional[str] = None,              # Language restrict, e.g. "lang_en"
    safe: Optional[str] = None,            # "off" | "active"
    api_key: Optional[str] = None,         # Per-call override
    cx: Optional[str] = None,              # Per-call override
) -> List[Dict[str, str]]:
    """
    Search the web using Google CSE.

    Returns a list of dicts: {title, url, snippet}
    """
    if not (query or "").strip():
        return []
    num = max_results or _DEFAULT_NUM
    return _google_cse(
        query.strip(),
        site=site,
        num=num,
        gl=gl,
        lr=lr,
        safe=(safe or _SAFE),
        api_key=api_key,
        cx=cx,
    )
