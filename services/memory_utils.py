# memory_utils.py
# Elasticsearch-backed memory helpers for the Discord bot.
# - Uses message_id as the ES _id (dedupe)
# - Conversation scoping via conversation_key = "{guild}:{channel}:{user}"
# - Context windows are taken from newest -> oldest (then reversed to chronological)
# - Helpers to query newest/oldest indexed ids to steer backfill behavior
# - NEW: recent-first paging via search_after, and recent-biased keyword search
#
# NOTE: This version uses the official 'elasticsearch' client exclusively.
#       Env vars keep their OPENSEARCH_* names for backward-compat.

from __future__ import annotations

import os
import sys
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import Elasticsearch  # type: ignore
from elasticsearch.exceptions import AuthenticationException, ApiError  # type: ignore

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
def _get_bool(env: str, default: bool) -> bool:
    v = os.environ.get(env)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

# Optional config module
try:
    from config import (
        OPENSEARCH_HOST as _CFG_HOST,
        OPENSEARCH_USER as _CFG_USER,
        OPENSEARCH_PASS as _CFG_PASS,
        OPENSEARCH_VERIFY_CERTS as _CFG_VERIFY,
        OPENSEARCH_INDEX as _CFG_INDEX,
    )
except Exception:
    _CFG_HOST = _CFG_USER = _CFG_PASS = _CFG_VERIFY = _CFG_INDEX = None

# Back-compat: keep OPENSEARCH_* names, but talk to Elasticsearch
OS_HOST = _CFG_HOST or os.environ.get("OPENSEARCH_HOST") or os.environ.get("OPENSEARCH_URL") or "https://localhost:9200"
OS_USER = _CFG_USER or os.environ.get("OPENSEARCH_USER") or os.environ.get("ELASTIC_USERNAME") or "elastic"
OS_PASS = _CFG_PASS or os.environ.get("OPENSEARCH_PASS") or os.environ.get("ELASTIC_PASSWORD") or ""
OS_VERIFY = _get_bool("OPENSEARCH_VERIFY_CERTS", bool(_CFG_VERIFY) if _CFG_VERIFY is not None else False)
OPENSEARCH_INDEX = _CFG_INDEX or os.environ.get("OPENSEARCH_INDEX", "discord_chat_memory")
OPENSEARCH_ENABLED = _get_bool("OPENSEARCH_ENABLED", True)

# ------------------------------------------------------------
# Module-global client handle
# ------------------------------------------------------------
es: Optional[Elasticsearch] = None
_es_disabled = not OPENSEARCH_ENABLED

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

def _now_iso() -> str:
    return _now_utc().isoformat()

def conversation_key(guild_id: str | int, channel_id: str | int, user_id: str | int) -> str:
    return f"{guild_id}:{channel_id}:{user_id}"


def _disable_es(reason: str) -> None:
    global es, _es_disabled
    es = None
    _es_disabled = True
    logger.warning("[ES] disabling ES-backed memory: %s", reason)

# ------------------------------------------------------------
# Client init & index management
# ------------------------------------------------------------
def init_es_client(force: bool = False) -> Optional[Elasticsearch]:
    """Create global client if missing. Safe to call often."""
    global es, _es_disabled
    if _es_disabled and not force:
        return None
    if es is not None and not force:
        return es

    host = OS_HOST
    user = OS_USER
    pwd = OS_PASS
    verify = OS_VERIFY

    try:
        es = Elasticsearch(
            hosts=[host],
            basic_auth=(user, pwd) if (user or pwd) else None,
            verify_certs=verify,
            ssl_show_warn=not verify,
            request_timeout=10,
        )
        logger.info(
            "[ES] build host=%s user=%s pass_len=%s verify=%s py=%s",
            host, user, len(pwd or ""), verify, sys.executable
        )
    except Exception as e:
        logger.warning("[ES] Failed to initialize client: %r", e)
        _disable_es(f"client_init_failed: {e!r}")
        return None

    # Try an immediate ping/auth so we fail fast if the server isn't reachable.
    try:
        if not es.ping():
            _disable_es("ping_failed")
            return None
        who = es.security.authenticate()
        logger.info(
            "[ES] startup auth OK as %s (realms %s/%s)",
            who.get("username"),
            who.get("authentication_realm", {}).get("name"),
            who.get("lookup_realm", {}).get("name"),
        )
    except Exception as e:
        logger.error("[ES] startup auth FAILED: %r", e)
        _disable_es(f"startup_auth_failed: {e!r}")
        return None

    # Best-effort ensure index exists
    try:
        ensure_index()
    except AuthenticationException as e:
        logger.warning("[ES] ensure_index auth failed: %r", e)
        _disable_es(f"ensure_index_auth_failed: {e!r}")
        return None
    except Exception as e:
        logger.warning("[ES] ensure_index failed: %r", e)
        _disable_es(f"ensure_index_failed: {e!r}")
        return None

    return es

def ensure_index():
    """Create index and mapping if it doesn't exist."""
    client = es or init_es_client()
    if client is None:
        return

    # HEAD/exists
    if client.indices.exists(index=OPENSEARCH_INDEX):  # type: ignore[attr-defined]
        return

    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "message_id": {"type": "keyword"},
                "guild_id": {"type": "keyword"},
                "channel_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "conversation_key": {"type": "keyword"},
                "reply_to_id": {"type": "keyword", "null_value": "NULL"},
                "role": {"type": "keyword"},
                "model": {"type": "keyword"}, # New field for model tagging
                "content": {"type": "text"},
                "timestamp": {"type": "date"},
                "ts": {"type": "date"},
            },
        },
    }
    client.indices.create(index=OPENSEARCH_INDEX, body=body)  # type: ignore[attr-defined]
    logger.info("[ES] created index %s", OPENSEARCH_INDEX)

    # Attempt to update mapping for existing index (best-effort)
    try:
        client.indices.put_mapping(
            index=OPENSEARCH_INDEX,
            body={"properties": {"model": {"type": "keyword"}}}
        )
    except Exception:
        pass # Ignore if already exists or fails

# ------------------------------------------------------------
# Indexing
# ------------------------------------------------------------
def index_message(
    *,
    message_id: str,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
    reply_to_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Index a single Discord message. Uses message_id as the ES _id to dedupe.
    """
    client = es or init_es_client()
    if client is None:
        return None

    ts = timestamp or _now_iso()
    ckey = conversation_key(guild_id, channel_id, user_id)

    doc = {
        "message_id": str(message_id),
        "guild_id": str(guild_id),
        "channel_id": str(channel_id),
        "user_id": str(user_id),
        "conversation_key": ckey,
        "reply_to_id": str(reply_to_id) if reply_to_id else None,
        "role": str(role),
        "model": str(model) if model else None,
        "content": content or "",
        "timestamp": ts,
        "ts": ts,
    }

    try:
        # ES 8.x prefers 'document=' (body still works but is deprecated)
        resp = client.index(index=OPENSEARCH_INDEX, id=str(message_id), document=doc, refresh=False)
        _id = resp.get("_id")
        logger.debug("[ES] indexed id=%s ckey=%s role=%s model=%s", _id, ckey, role, model)
        return _id
    except ApiError as e:
        logger.warning("[ES] index_message API error: %r", e)
        return None
    except Exception as e:
        logger.warning("[ES] index_message error: %r", e)
        _disable_es(f"index_message_failed: {e!r}")
        return None

# ------------------------------------------------------------
# Low-level search
# ------------------------------------------------------------
def _search_raw(query: Dict[str, Any], *, index: Optional[str] = None, size: int = 24, source: Optional[List[str]] = None, sort=None) -> Dict[str, Any]:
    client = es or init_es_client()
    if client is None:
        return {"hits": {"total": {"value": 0}, "hits": []}}

    target_index = index or OPENSEARCH_INDEX
    body: Dict[str, Any] = {"query": query, "size": int(size)}
    if sort is not None:
        body["sort"] = sort
    if source:
        body["_source"] = source

    try:
        logger.debug("es.search [%s] > %r", target_index, body)
    except Exception:
        pass

    # ES 8.x allows body=; also supports query= but we keep body for parity with existing code.
    try:
        resp = client.search(index=target_index, body=body)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("[ES] search error: %r", e)
        _disable_es(f"search_failed: {e!r}")
        return {"hits": {"total": {"value": 0}, "hits": []}}

    try:
        logger.debug("es.search [%s] < %r", target_index, resp)
    except Exception:
        pass

    return resp

# ------------------------------------------------------------
# Conversation windows (context for chat)
# ------------------------------------------------------------
def build_message_window(
    *,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    limit_msgs: int = 24,
) -> List[Dict[str, str]]:
    """
    Return a list of {role, content} for this (guild, channel, user) conversation.
    We take the most recent N messages (desc), then reverse to oldest->newest for the model.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    resp = _search_raw(
        {"bool": {"filter": [{"term": {"conversation_key.keyword": ckey}}]}},
        size=limit_msgs,
        source=["role", "content", "timestamp", "user_id", "message_id"],
        sort=[{"timestamp": {"order": "desc"}}],
    )

    hits = [h.get("_source", {}) for h in resp.get("hits", {}).get("hits", [])]

    # reverse to chronological order for chat APIs
    hits.reverse()
    msgs = [{"role": (h.get("role") or "user"), "content": (h.get("content") or "")} for h in hits]
    return msgs

# ------------------------------------------------------------
# Timeline block for system prompt (no timebox, no redaction)
# ------------------------------------------------------------
def _quantize_ago(dt: datetime, now: Optional[datetime] = None) -> str:
    now = now or _now_utc()
    delta = now - dt
    sec = int(delta.total_seconds())
    if sec < 45:
        return "just now"
    if sec < 90:
        return "1m ago"
    minutes = sec // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 18:
        return f"{months}mo ago"
    years = days // 365
    return f"{years}y ago"

def fetch_recent_timeline(
    *,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    max_items: int = 12,
) -> List[Tuple[str, str]]:
    """
    Return [(ago, "role: content")] newest->oldest. No timeboxing.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    resp = _search_raw(
        {"bool": {"filter": [{"term": {"conversation_key.keyword": ckey}}]}},
        size=max_items,
        source=["role", "content", "timestamp"],
        sort=[{"timestamp": {"order": "desc"}}],
    )

    now = _now_utc()
    items: List[Tuple[str, str]] = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        ts = src.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            dt = now
        ago = _quantize_ago(dt, now)
        who = src.get("role", "user")
        content = (src.get("content") or "").strip()
        if len(content) > 140:
            content = content[:137].rstrip() + "…"
        items.append((ago, f"{who}: {content}"))
    return items

def build_timeline_prompt_block(
    *,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    max_items: int = 12,
) -> str:
    now_iso = _now_iso()
    lines = [
        "Use temporal context when helpful. For example, if asked 'when did I say that?', reference how long ago an item occurred using the recent timeline below.",
        f"Current UTC time: {now_iso}",
        "",
    ]
    timeline = fetch_recent_timeline(
        guild_id=guild_id, channel_id=channel_id, user_id=user_id, max_items=max_items
    )
    if not timeline:
        lines.append("Recent timeline: (no recent messages found)")
        return "\n".join(lines)

    lines.append("Recent timeline (newest first):")
    for ago, pretty in timeline:
        lines.append(f"- [{ago}] {pretty}")
    return "\n".join(lines)

# ------------------------------------------------------------
# Helpers for smarter backfill (avoid pulling the very first messages)
# ------------------------------------------------------------
def get_newest_indexed_message_id(
    *, guild_id: str | int, channel_id: str | int, user_id: str | int
) -> Optional[str]:
    """
    Return the newest indexed message_id for this conversation (or None).
    Use this to backfill with Discord's channel.history(after=...) so you only add newer messages.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    resp = _search_raw(
        {"bool": {"filter": [{"term": {"conversation_key.keyword": ckey}}]}},
        size=1,
        source=["message_id", "timestamp"],
        sort=[{"timestamp": {"order": "desc"}}],
    )
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return None
    return hits[0].get("_source", {}).get("message_id")

def get_oldest_indexed_message_id(
    *, guild_id: str | int, channel_id: str | int, user_id: str | int
) -> Optional[str]:
    """
    Return the oldest indexed message_id for this conversation (or None).
    (Useful if you still want a traditional 'older history' backfill path.)
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    resp = _search_raw(
        {"bool": {"filter": [{"term": {"conversation_key.keyword": ckey}}]}},
        size=1,
        source=["message_id", "timestamp"],
        sort=[{"timestamp": {"order": "asc"}}],
    )
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return None
    return hits[0].get("_source", {}).get("message_id")

# ------------------------------------------------------------
# NEW: recent-first paging + recent-biased matches (optional helpers)
# ------------------------------------------------------------
# Stable tie-break with message_id ensures deterministic paging
_SORT_RECENT = [{"timestamp": {"order": "desc"}}, {"message_id": {"order": "desc"}}]

def fetch_recent_page(
    *,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    size: int = 24,
    after: List[Any] | None = None,
    source: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Any] | None]:
    """
    Newest-first page of docs using sort desc + search_after.
    Returns (docs, next_after). `after` should be the previous page's last `sort` array.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    body: Dict[str, Any] = {
        "query": {"bool": {"filter": [{"term": {"conversation_key.keyword": ckey}}]}},
        "size": int(size),
        "sort": _SORT_RECENT,
    }
    if after:
        body["search_after"] = after
    if source:
        body["_source"] = source

    client = es or init_es_client()
    if client is None:
        return ([], None)

    resp = client.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore[attr-defined]
    hits = resp.get("hits", {}).get("hits", [])
    docs = [h.get("_source", {}) for h in hits]
    next_after = hits[-1].get("sort") if hits else None
    return (docs, next_after)

def _parse_relative_time_search(query: str) -> Optional[Dict[str, Any]]:
    """
    Parse queries like "2 months ago", "yesterday", "last week" into an ES range query.
    Returns a dict like {"gte": "now-2d", "lte": "now-1d"} or similar.
    """
    q = query.lower()
    now = datetime.now(timezone.utc)
    
    # regex for "X [unit] ago"
    match = re.search(r'(\d+)\s+(sec|second|min|minute|hour|hr|day|week|month|year)s?\s+ago', q)
    if match:
        val = int(match.group(1))
        unit = match.group(2)
        
        # Determine delta
        if "sec" in unit: delta = timedelta(seconds=val)
        elif "min" in unit: delta = timedelta(minutes=val)
        elif "hour" in unit or "hr" in unit: delta = timedelta(hours=val)
        elif "day" in unit: delta = timedelta(days=val)
        elif "week" in unit: delta = timedelta(weeks=val)
        elif "month" in unit: delta = timedelta(days=val * 30) # approx
        elif "year" in unit: delta = timedelta(days=val * 365) # approx
        else: return None
        
        target_time = now - delta
        
        # Define a window. if unit is large, window is larger.
        # "2 months ago" -> probably means around that time, maybe +/- 3 days?
        # "2 minutes ago" -> +/- 30 seconds?
        
        if "month" in unit:
            window = timedelta(days=10)
        elif "year" in unit:
            window = timedelta(days=30)
        elif "week" in unit:
             window = timedelta(days=4)
        elif "day" in unit:
             window = timedelta(days=1)
        else:
             window = timedelta(minutes=15)
             
        start_ts = (target_time - window).isoformat()
        end_ts = (target_time + window).isoformat()
        
        return {"gte": start_ts, "lte": end_ts}

    return None

def search_history_for_context(
    guild_id: str | int, 
    channel_id: str | int, 
    user_id: str | int, 
    query_text: str,
    limit: int = 5,
    oldest_first: bool = False
) -> str:
    """
    Search memory for context injection (RAG).
    Returns a formatted string of found messages.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    
    # Check for "random" intent first
    if "random" in query_text.lower():
        # Use function_score with random_score
        query = {
            "function_score": {
                "query": { 
                    "bool": { 
                        "filter": [{"term": {"conversation_key.keyword": ckey}}]
                    } 
                },
                "functions": [
                    {"random_score": {}}
                ],
                "boost_mode": "replace"
            }
        }
        # No sort order (uses score)
        resp = _search_raw(
            query,
            size=limit,
            source=["role", "content", "timestamp"],
            sort=None 
        )
    else:
        # Check for relative time queries (e.g. "2 months ago")
        time_range = _parse_relative_time_search(query_text)
        
        # Standard logic
        query = {
            "bool": {
                "filter": [{"term": {"conversation_key.keyword": ckey}}],
                "must": []
            }
        }
        
        if time_range:
             # Apply range filter
             query["bool"]["filter"].append({"range": {"timestamp": time_range}})
             # If time is specified, we likely don't need content text match unless specifically asked.
             # But "entry for 2 months ago" -> "entry" is generic.
             # We should probably clear the content match if it was just time-related words.
             # For now, let's just NOT add the content match if we found a time range, 
             # unless there's OTHER text left.
             
             # Remove time phrases from query to see if there's other text
             # (Simple approach: just rely on the range and maybe loose match)
             pass
        else:
            skip_content_filter = any(k in query_text.lower() for k in ["first message", "first thing", "history", "beginning", "start"])
            
            if not skip_content_filter and query_text.strip():
                # Clean query text of trigger words
                cleaned = query_text.replace("search", "").replace("history", "").strip()
                if cleaned:
                     query["bool"]["must"].append({"match": {"content": cleaned}})
        
        sort_order = "asc" if oldest_first else "desc"
        resp = _search_raw(
            query,
            size=limit,
            source=["role", "content", "timestamp"],
            sort=[{"timestamp": {"order": sort_order}}]
        )
    
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return ""
    
    lines = []
    for h in hits:
        src = h.get("_source", {})
        role = src.get("role", "unknown")
        content = src.get("content", "").strip()
        ts = src.get("timestamp", "")
        if content:
            lines.append(f"[{ts}] {role}: {content}")
            
    return "\n".join(lines)

def fetch_matches_recent(
    *,
    guild_id: str | int,
    channel_id: str | int,
    user_id: str | int,
    query: str,
    size: int = 16,
    source: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Recent-biased match search within the conversation scope.
    """
    ckey = conversation_key(guild_id, channel_id, user_id)
    body: Dict[str, Any] = {
        "query": {
            "bool": {
                "must": [{"match": {"content": {"query": query, "operator": "and"}}}],
                "filter": [{"term": {"conversation_key.keyword": ckey}}],
            }
        },
        "size": int(size),
        "sort": _SORT_RECENT,
    }
    if source:
        body["_source"] = source

    client = es or init_es_client()
    if client is None:
        return []

    resp = client.search(index=OPENSEARCH_INDEX, body=body)  # type: ignore[attr-defined]
    hits = resp.get("hits", {}).get("hits", [])
    return [h.get("_source", {}) for h in hits]

def build_timeline_from_docs(docs: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Helper to turn docs (usually from fetch_recent_page) into [(ago, 'role: content')] rows.
    """
    now = _now_utc()
    items: List[Tuple[str, str]] = []
    for src in docs:
        ts = src.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            dt = now
        ago = _quantize_ago(dt, now)
        who = src.get("role", "user")
        content = (src.get("content") or "").strip()
        if len(content) > 140:
            content = content[:137].rstrip() + "…"
        items.append((ago, f"{who}: {content}"))
    return items

# ------------------------------------------------------------
# Exports
# ------------------------------------------------------------
__all__ = [
    "init_es_client",
    "ensure_index",
    "index_message",
    "build_message_window",
    "fetch_recent_timeline",
    "build_timeline_prompt_block",
    "get_newest_indexed_message_id",
    "get_oldest_indexed_message_id",
    "fetch_recent_page",
    "fetch_matches_recent",
    "build_timeline_from_docs",
    "conversation_key",
    "OPENSEARCH_INDEX",
    "OS_HOST",
    "OS_USER",
    "OS_PASS",
    "OS_VERIFY",
]
