# usage_costs.py
# Lightweight SQLite accounting for OpenAI usage & cost.
# Public API:
#   record(model, usage, cost, *, label=None, meta=None) -> None
#   last() -> dict
#   window_minutes(minutes=60) -> dict
#   today() -> dict
#   month_to_date() -> dict
#
# Env:
#   USAGE_DB_PATH=/absolute/path/usage_costs.db   (default: ./usage_costs.db)
#   USAGE_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR       (optional)

from __future__ import annotations

import json
import os
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# ----------------------------
# Config & Logging
# ----------------------------
DB_PATH = os.getenv("USAGE_DB_PATH", "./usage_costs.db")

_log_level = os.getenv("USAGE_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.WARNING))
logger = logging.getLogger("usage_costs")

# ----------------------------
# Schema
# ----------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_logs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc            TEXT NOT NULL,            -- ISO8601 in UTC
  model             TEXT NOT NULL,
  label             TEXT,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  total_tokens      INTEGER NOT NULL DEFAULT 0,
  cost_usd          REAL NOT NULL DEFAULT 0.0,
  meta_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_logs (ts_utc);
"""

# ----------------------------
# DB helpers
# ----------------------------
@contextmanager
def _conn_rw():
    # ensure directory exists (if DB_PATH has a directory portion)
    dirpart = os.path.dirname(DB_PATH)
    if dirpart:
        os.makedirs(dirpart, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _coerce_int(x: Any) -> int:
    try:
        return int(x or 0)
    except Exception:
        return 0

def _coerce_float(x: Any) -> float:
    try:
        return float(x or 0.0)
    except Exception:
        return 0.0

def _usage_fields(usage: Dict[str, Any] | None) -> Dict[str, int]:
    """
    Normalize usage across Chat Completions and Responses APIs.
    Accepts keys:
      - prompt_tokens / completion_tokens / total_tokens
      - input_tokens  / output_tokens
    Returns dict with prompt_tokens, completion_tokens, total_tokens.
    """
    u = usage or {}
    prompt = u.get("prompt_tokens")
    if prompt is None:
        prompt = u.get("input_tokens", 0)
    completion = u.get("completion_tokens")
    if completion is None:
        completion = u.get("output_tokens", 0)
    total = u.get("total_tokens")
    if total is None:
        total = (prompt or 0) + (completion or 0)
    return {
        "prompt_tokens": _coerce_int(prompt),
        "completion_tokens": _coerce_int(completion),
        "total_tokens": _coerce_int(total),
    }

# ----------------------------
# Public API
# ----------------------------
def record(
    model: str,
    usage: Dict[str, Any] | None,
    cost: float,
    *,
    label: Optional[str] = None,
    meta: Dict[str, Any] | None = None
) -> None:
    """
    Insert a single usage entry. Call this right after each OpenAI API call.
    """
    fields = _usage_fields(usage)
    payload = {
        "usage": usage or {},
        "label": label,
        **(meta or {})
    }
    meta_json = json.dumps(payload, ensure_ascii=False)

    logger.debug(
        "record(): model=%s label=%s prompt=%s completion=%s total=%s cost=%s",
        model, label, fields["prompt_tokens"], fields["completion_tokens"],
        fields["total_tokens"], cost
    )

    with _conn_rw() as c:
        c.execute(
            """
            INSERT INTO usage_logs
              (ts_utc, model, label, prompt_tokens, completion_tokens, total_tokens, cost_usd, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_utc_iso(),
                model,
                label,
                fields["prompt_tokens"],
                fields["completion_tokens"],
                fields["total_tokens"],
                _coerce_float(cost),
                meta_json,
            ),
        )

def last() -> Dict[str, Any]:
    """
    Return the most recent single exchange. If none exist, {}.
    """
    with _conn_rw() as c:
        cur = c.execute(
            """
            SELECT ts_utc, model, label, prompt_tokens, completion_tokens, total_tokens, cost_usd, meta_json
            FROM usage_logs
            ORDER BY id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            logger.debug("last(): no rows")
            return {}
    out = {
        "ts_utc": row[0],
        "model": row[1],
        "label": row[2],
        "prompt_tokens": row[3],
        "completion_tokens": row[4],
        "total_tokens": row[5],
        "cost": float(row[6]),
        "meta": json.loads(row[7] or "{}"),
    }
    logger.debug("last(): %s", out)
    return out

def _aggregate_where(where_sql: str, args: tuple) -> Dict[str, Any]:
    sql = f"""
    SELECT
      COUNT(*)                               AS calls,
      COALESCE(SUM(prompt_tokens), 0)        AS prompt_tokens,
      COALESCE(SUM(completion_tokens), 0)    AS completion_tokens,
      COALESCE(SUM(total_tokens), 0)         AS total_tokens,
      COALESCE(SUM(cost_usd), 0.0)           AS cost
    FROM usage_logs
    WHERE {where_sql}
    """
    with _conn_rw() as c:
        cur = c.execute(sql, args)
        row = cur.fetchone() or (0, 0, 0, 0, 0.0)
    out = {
        "calls": row[0],
        "prompt_tokens": row[1],
        "completion_tokens": row[2],
        "total_tokens": row[3],
        "cost": float(row[4]),
    }
    logger.debug("_aggregate_where(%s, %s): %s", where_sql, args, out)
    return out

def window_minutes(minutes: int = 60) -> Dict[str, Any]:
    minutes = max(1, int(minutes or 60))
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return _aggregate_where("ts_utc >= ?", (since.replace(microsecond=0).isoformat(),))

def today() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return _aggregate_where("ts_utc >= ?", (start.isoformat(),))

def month_to_date() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return _aggregate_where("ts_utc >= ?", (start.isoformat(),))

# ----------------------------
# CLI self-test (optional)
# ----------------------------
if __name__ == "__main__":
    # Run: python usage_costs.py
    print(f"DB_PATH={DB_PATH}")
    dummy_usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    record("selftest-model", dummy_usage, 0.00123, label="selftest", meta={"note": "hello"})
    print("last():", last())
    print("window_minutes(60):", window_minutes(60))
    print("today():", today())
    print("month_to_date():", month_to_date())
