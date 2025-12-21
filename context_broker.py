# context_broker.py
# Consent-free context broker:
# - store_message(...)  -> writes conversation lines to OpenSearch
# - fetch_context(...)  -> hybrid retrieval (user + channel + recent), ranked with time decay
#
# Uses config.py vars: ELASTIC_URL, ELASTIC_USERNAME, ELASTIC_PASSWORD

from __future__ import annotations
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests.auth import HTTPBasicAuth
    from config import ELASTIC_URL, ELASTIC_USERNAME, ELASTIC_PASSWORD
    OS_AUTH = HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD) if (ELASTIC_USERNAME or ELASTIC_PASSWORD) else None
    OS_CLIENT = OpenSearch(
        hosts=[ELASTIC_URL],
        http_auth=OS_AUTH,
        verify_certs=False,  # self-signed in your dev env
        connection_class=RequestsHttpConnection,
        timeout=10,
        max_retries=2,
        retry_on_timeout=True,
    )
except Exception as e:
    logging.warning(f"[context_broker] OpenSearch unavailable: {e}")
    OS_CLIENT = None

CHAT_INDEX = "discord_chat_memory"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            p = ""
        h.update(p.encode("utf-8", "ignore"))
    return h.hexdigest()

def _safe_os() -> Optional["OpenSearch"]:
    return OS_CLIENT

def _time_decay(ts: str, half_life_hours: float = 72.0) -> float:
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 1.0
    age_h = max(0.0, (time.time() - t) / 3600.0)
    return 0.5 ** (age_h / max(0.1, half_life_hours))

def _role_weight(role: str) -> float:
    return {"user": 1.0, "assistant": 0.9, "system": 0.7}.get((role or "user").lower(), 0.8)

def _score(hit: Dict[str, Any]) -> float:
    src  = hit.get("_source", {})
    ts   = src.get("timestamp") or _now_iso()
    role = src.get("role", "user")
    bm25 = float(hit.get("_score", 1.0))
    return bm25 * _time_decay(ts) * _role_weight(role)

def _dedupe(lines: List[str], max_len: int = 4000) -> str:
    seen = set()
    out  = []
    total = 0
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ln = len(line) + 1
        if total + ln > max_len:
            break
        out.append(line)
        total += ln
    return "\n".join(out)

def store_message(
    user_id: str,
    guild_id: str,
    channel_id: str,
    role: str,
    content: str,
    reply_to: Optional[str] = None,
    timestamp_iso: Optional[str] = None,
) -> None:
    """Consent-free storage (always store)."""
    if not content or not content.strip():
        return
    cli = _safe_os()
    if not cli:
        return
    try:
        doc_id = _hash_id(user_id, guild_id or "", channel_id or "", role or "", content[:256], reply_to or "")
        body = {
            "user_id": str(user_id),
            "guild_id": str(guild_id or ""),
            "channel_id": str(channel_id or ""),
            "role": role or "user",
            "content": content.strip(),
            "reply_to": reply_to,
            "timestamp": timestamp_iso or _now_iso(),
        }
        cli.index(index=CHAT_INDEX, id=doc_id, body=body)
    except Exception as e:
        logging.warning(f"[context_broker] store failed: {e}")

def fetch_context(
    user_id: str,
    guild_id: str,
    channel_id: str,
    prompt: str,
    *,
    k_user: int = 12,
    k_channel: int = 16,
    k_recent: int = 12,
    max_chars: int = 4000,
) -> str:
    """Hybrid retrieval: user + channel + recent continuity."""
    cli = _safe_os()
    if not cli:
        return ""

    query_text = (prompt or "").strip() or "*"
    queries: List[Tuple[Dict[str, Any], int]] = []

    user_q = {
        "size": k_user,
        "query": {
            "bool": {
                "must": [
                    {"term": {"user_id.keyword": str(user_id)}},
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["content^2", "role", "reply_to"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    },
                ]
            }
        }
    }
    queries.append((user_q, k_user))

    chan_q = {
        "size": k_channel,
        "query": {
            "bool": {
                "must": [
                    {"term": {"guild_id.keyword": str(guild_id)}},
                    {"term": {"channel_id.keyword": str(channel_id)}},
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["content^2", "role", "reply_to"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    },
                ]
            }
        }
    }
    queries.append((chan_q, k_channel))

    recent_q = {
        "size": k_recent,
        "sort": [{"timestamp": {"order": "desc"}}],
        "query": {
            "bool": {
                "must": [
                    {"term": {"guild_id.keyword": str(guild_id)}},
                    {"term": {"channel_id.keyword": str(channel_id)}},
                ]
            }
        }
    }
    queries.append((recent_q, k_recent))

    hits: List[Dict[str, Any]] = []
    try:
        for q, _ in queries:
            res = cli.search(index=CHAT_INDEX, body=q)
            hits.extend(res.get("hits", {}).get("hits", []))
    except Exception as e:
        logging.warning(f"[context_broker] search failed: {e}")
        return ""

    scored = [(_score(h), h) for h in hits]
    scored.sort(key=lambda x: x[0], reverse=True)

    lines: List[str] = []
    for _, h in scored:
        src = h.get("_source", {})
        role = (src.get("role") or "user").lower()
        text = (src.get("content") or "").strip()
        if not text:
            continue
        prefix = {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, "User")
        lines.append(f"{prefix}: {text}")

    return _dedupe(lines, max_len=max_chars)
