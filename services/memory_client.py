from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch  # type: ignore
from elasticsearch.exceptions import ApiError, AuthenticationException  # type: ignore

logger = logging.getLogger(__name__)


def _get_bool(env: str, default: bool) -> bool:
    v = os.environ.get(env)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


try:
    from config import (
        OPENSEARCH_HOST as _CFG_HOST,
        OPENSEARCH_INDEX as _CFG_INDEX,
        OPENSEARCH_PASS as _CFG_PASS,
        OPENSEARCH_USER as _CFG_USER,
        OPENSEARCH_VERIFY_CERTS as _CFG_VERIFY,
    )
except Exception:
    _CFG_HOST = _CFG_USER = _CFG_PASS = _CFG_VERIFY = _CFG_INDEX = None


OS_HOST = _CFG_HOST or os.environ.get("OPENSEARCH_HOST") or os.environ.get("OPENSEARCH_URL") or "https://localhost:9200"
OS_USER = _CFG_USER or os.environ.get("OPENSEARCH_USER") or os.environ.get("ELASTIC_USERNAME") or "elastic"
OS_PASS = _CFG_PASS or os.environ.get("OPENSEARCH_PASS") or os.environ.get("ELASTIC_PASSWORD") or ""
OS_VERIFY = _get_bool("OPENSEARCH_VERIFY_CERTS", bool(_CFG_VERIFY) if _CFG_VERIFY is not None else False)
OPENSEARCH_INDEX = _CFG_INDEX or os.environ.get("OPENSEARCH_INDEX", "discord_chat_memory")
OPENSEARCH_ENABLED = _get_bool("OPENSEARCH_ENABLED", True)


@dataclass
class MemoryRuntime:
    client: Optional[Elasticsearch] = None
    disabled: bool = not OPENSEARCH_ENABLED


runtime = MemoryRuntime()


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def conversation_key(guild_id: str | int, channel_id: str | int, user_id: str | int) -> str:
    return f"{guild_id}:{channel_id}:{user_id}"


def _disable_es(reason: str) -> None:
    runtime.client = None
    runtime.disabled = True
    logger.warning("[ES] disabling ES-backed memory: %s", reason)


def init_es_client(force: bool = False) -> Optional[Elasticsearch]:
    if runtime.disabled and not force:
        return None
    if runtime.client is not None and not force:
        return runtime.client

    try:
        runtime.client = Elasticsearch(
            hosts=[OS_HOST],
            basic_auth=(OS_USER, OS_PASS) if (OS_USER or OS_PASS) else None,
            verify_certs=OS_VERIFY,
            ssl_show_warn=not OS_VERIFY,
            request_timeout=10,
        )
        logger.info(
            "[ES] build host=%s user=%s pass_len=%s verify=%s py=%s",
            OS_HOST,
            OS_USER,
            len(OS_PASS or ""),
            OS_VERIFY,
            sys.executable,
        )
    except Exception as e:
        logger.warning("[ES] Failed to initialize client: %r", e)
        _disable_es(f"client_init_failed: {e!r}")
        return None

    try:
        if not runtime.client.ping():
            _disable_es("ping_failed")
            return None
        who = runtime.client.security.authenticate()
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

    return runtime.client


def ensure_index():
    client = runtime.client or init_es_client()
    if client is None:
        return

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
                "model": {"type": "keyword"},
                "content": {"type": "text"},
                "timestamp": {"type": "date"},
                "ts": {"type": "date"},
            },
        },
    }
    client.indices.create(index=OPENSEARCH_INDEX, body=body)  # type: ignore[attr-defined]
    logger.info("[ES] created index %s", OPENSEARCH_INDEX)

    try:
        client.indices.put_mapping(index=OPENSEARCH_INDEX, body={"properties": {"model": {"type": "keyword"}}})
    except Exception:
        pass


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
    client = runtime.client or init_es_client()
    if client is None:
        return None

    ts = timestamp or _now_iso()
    doc = {
        "message_id": str(message_id),
        "guild_id": str(guild_id),
        "channel_id": str(channel_id),
        "user_id": str(user_id),
        "conversation_key": conversation_key(guild_id, channel_id, user_id),
        "reply_to_id": str(reply_to_id) if reply_to_id else None,
        "role": str(role),
        "model": str(model) if model else None,
        "content": content or "",
        "timestamp": ts,
        "ts": ts,
    }

    try:
        resp = client.index(index=OPENSEARCH_INDEX, id=str(message_id), document=doc, refresh=False)
        return resp.get("_id")
    except ApiError as e:
        logger.warning("[ES] index_message API error: %r", e)
        return None
    except Exception as e:
        logger.warning("[ES] index_message error: %r", e)
        _disable_es(f"index_message_failed: {e!r}")
        return None


def search_raw(
    query: Dict[str, Any],
    *,
    index: Optional[str] = None,
    size: int = 24,
    source: Optional[List[str]] = None,
    sort=None,
) -> Dict[str, Any]:
    client = runtime.client or init_es_client()
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

