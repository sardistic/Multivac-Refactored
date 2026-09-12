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
#   USAGE_METRICS_PATH=/absolute/path/usage_metrics.json (optional public aggregates)
#   USAGE_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR       (optional)
#   USAGE_TZ=America/New_York                       (report day/month boundary tz)

from __future__ import annotations

import contextvars
import json
import os
import sqlite3
import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - stdlib since 3.9
    ZoneInfo = None

# ----------------------------
# Config & Logging
# ----------------------------
DB_PATH = os.getenv("USAGE_DB_PATH", "./usage_costs.db")
METRICS_PATH = os.getenv("USAGE_METRICS_PATH", "").strip()

_log_level = os.getenv("USAGE_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.WARNING))
logger = logging.getLogger("usage_costs")

# Reporting timezone for "today" / "month-to-date" boundaries. Rows are always
# STORED in UTC; only the report windows roll over on the local calendar day so
# an evening report (e.g. 10pm ET) doesn't read 0 just because it's already the
# next day in UTC. Defaults to US Eastern; override with USAGE_TZ.
_REPORT_TZ_NAME = os.getenv("USAGE_TZ", "America/New_York")
if ZoneInfo is not None:
    try:
        REPORT_TZ: timezone | Any = ZoneInfo(_REPORT_TZ_NAME)
    except Exception:
        logger.warning("Unknown USAGE_TZ=%r; falling back to UTC", _REPORT_TZ_NAME)
        REPORT_TZ = timezone.utc
else:
    REPORT_TZ = timezone.utc


def _day_start_utc(now_utc: datetime, tz) -> datetime:
    """Start of the current local (tz) day, expressed in UTC."""
    local_midnight = now_utc.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc)


def _month_start_utc(now_utc: datetime, tz) -> datetime:
    """Start of the current local (tz) month, expressed in UTC."""
    local_first = now_utc.astimezone(tz).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return local_first.astimezone(timezone.utc)


def _week_start_utc(now_utc: datetime, tz) -> datetime:
    """Monday start of the current local week, expressed in UTC."""
    local_now = now_utc.astimezone(tz)
    local_monday = (local_now - timedelta(days=local_now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return local_monday.astimezone(timezone.utc)


def _year_start_utc(now_utc: datetime, tz) -> datetime:
    """Start of the current local (tz) year, expressed in UTC."""
    local_first = now_utc.astimezone(tz).replace(
        month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    )
    return local_first.astimezone(timezone.utc)


def _report_day_start_iso() -> str:
    return _day_start_utc(datetime.now(timezone.utc), REPORT_TZ).isoformat()


def _report_month_start_iso() -> str:
    return _month_start_utc(datetime.now(timezone.utc), REPORT_TZ).isoformat()


def _report_year_start_iso() -> str:
    return _year_start_utc(datetime.now(timezone.utc), REPORT_TZ).isoformat()

# ----------------------------
# Schema
# ----------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_logs (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc            TEXT NOT NULL,            -- ISO8601 in UTC
  model             TEXT NOT NULL,
  label             TEXT,
  user_id           TEXT,
  prompt_tokens     INTEGER NOT NULL DEFAULT 0,
  cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
  cache_write_tokens INTEGER NOT NULL DEFAULT 0,
  completion_tokens INTEGER NOT NULL DEFAULT 0,
  total_tokens      INTEGER NOT NULL DEFAULT 0,
  cost_usd          REAL NOT NULL DEFAULT 0.0,
  meta_json         TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_logs (ts_utc);
"""

# ----------------------------
# Per-request attribution context
# ----------------------------
# Set once per incoming Discord message (task-local, safe under asyncio):
#   usage_costs.set_request_context(user_id="123", intent="chat")
# Every record() during that task picks it up automatically.
_request_ctx: contextvars.ContextVar[Dict[str, Any] | None] = contextvars.ContextVar(
    "usage_request_ctx", default=None
)

def set_request_context(**fields: Any) -> None:
    _request_ctx.set({k: v for k, v in fields.items() if v is not None})

def get_request_context() -> Dict[str, Any]:
    return dict(_request_ctx.get() or {})

# Running totals for the request currently in flight, so a completion card can
# report what the run actually cost without re-querying the ledger.
_request_totals: contextvars.ContextVar[Dict[str, float] | None] = contextvars.ContextVar(
    "usage_request_totals", default=None
)

def reset_request_totals() -> None:
    _request_totals.set({"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0})

def get_request_totals() -> Dict[str, float]:
    return dict(_request_totals.get() or {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0})

def _accumulate_request_totals(tokens: Any, cost: Any) -> None:
    running = _request_totals.get()
    if running is None:
        return
    running["calls"] += 1
    running["tokens"] += _coerce_float(tokens)
    running["cost_usd"] += _coerce_float(cost)

# ----------------------------
# Pricing (USD per 1M tokens). Tuples are input, output, cached input,
# cache-write. Two-value operator overrides remain supported.
# Override via env: OPENAI_PRICE_JSON='{"gpt-5.6-terra": [2,12,.2,2.5]}'
# ----------------------------
_DEFAULT_PRICING: Dict[str, tuple] = {
    # prefix: (input_per_1m, output_per_1m, cached_input_per_1m, cache_write_per_1m)
    "gpt-5.6-sol": (5.00, 30.00, 0.50, 6.25),
    "gpt-5.6-terra": (2.00, 12.00, 0.20, 2.50),
    "gpt-5.6-luna": (0.20, 1.20, 0.02, 0.25),
    "gpt-5.6": (5.00, 30.00, 0.50, 6.25),
    "gpt-5.5": (1.25, 10.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5": (1.25, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    # GPT Image 2.5 Sunburst/Flare: $5/M text input, $30/M image output.
    "gpt-image": (5.00, 30.00),
    "claude-fable-5": (10.00, 50.00, 1.00, 12.50),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude": (3.00, 15.00),
    # gemini-3.7-flash promo rates run through 2026-12-31; they double to
    # (1.50, 7.50) on 2027-01-01. Needs its own row because estimate_cost
    # matches by longest startswith prefix and "gemini-3.7-flash" does not
    # start with "gemini-3-flash".
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3-flash": (0.30, 2.50),
    "gemini": (0.30, 2.50),
}

def _pricing() -> Dict[str, tuple]:
    raw = os.getenv("OPENAI_PRICE_JSON")
    if not raw:
        return _DEFAULT_PRICING
    try:
        parsed = json.loads(raw)
        return {k: tuple(v) for k, v in parsed.items()}
    except Exception:
        logger.warning("Bad OPENAI_PRICE_JSON; using defaults")
        return _DEFAULT_PRICING

def estimate_cost(model: str, usage: Dict[str, Any] | None) -> float:
    fields = _usage_fields(usage)
    m = (model or "").lower()
    best = None
    for prefix, rates in _pricing().items():
        if m.startswith(prefix.lower()) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, rates)
    if not best:
        return 0.0
    rates = best[1]
    in_rate, out_rate = rates[:2]
    cached_rate = rates[2] if len(rates) >= 3 else in_rate
    write_rate = rates[3] if len(rates) >= 4 else in_rate
    cached = min(fields["cached_prompt_tokens"], fields["prompt_tokens"])
    cache_write = min(
        fields["cache_write_tokens"],
        max(0, fields["prompt_tokens"] - cached),
    )
    uncached = max(0, fields["prompt_tokens"] - cached - cache_write)
    return (
        uncached * in_rate
        + cached * cached_rate
        + cache_write * write_rate
        + fields["completion_tokens"] * out_rate
    ) / 1_000_000.0

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
        # Migrate pre-user_id databases in place. Must run before creating the
        # user_id index, or the index DDL fails on old tables.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(usage_logs)")}
        if "user_id" not in cols:
            conn.execute("ALTER TABLE usage_logs ADD COLUMN user_id TEXT")
        if "cached_prompt_tokens" not in cols:
            conn.execute(
                "ALTER TABLE usage_logs ADD COLUMN cached_prompt_tokens INTEGER NOT NULL DEFAULT 0"
            )
        if "cache_write_tokens" not in cols:
            conn.execute(
                "ALTER TABLE usage_logs ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_logs (user_id)")
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
    Returns prompt/completion/total plus cached-read and cache-write tokens.
    """
    u = usage or {}
    details = (
        u.get("prompt_tokens_details")
        or u.get("input_tokens_details")
        or {}
    )
    if not isinstance(details, dict):
        details = {}
    cache_read = u.get("cached_prompt_tokens")
    if cache_read is None:
        cache_read = u.get("cache_read_input_tokens")
    if cache_read is None:
        cache_read = details.get("cached_tokens", 0)
    cache_write = u.get("cache_write_tokens")
    if cache_write is None:
        cache_write = u.get("cache_creation_input_tokens")
    if cache_write is None:
        cache_write = details.get("cache_write_tokens", 0)

    prompt = u.get("prompt_tokens")
    if prompt is None:
        prompt = u.get("input_tokens", 0)
        # Anthropic reports uncached input and cache read/write as separate
        # top-level fields. Responses/OpenAI include cached reads in input.
        if "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:
            prompt = (prompt or 0) + (cache_read or 0) + (cache_write or 0)
    completion = u.get("completion_tokens")
    if completion is None:
        completion = u.get("output_tokens", 0)
    total = u.get("total_tokens")
    if total is None:
        total = (prompt or 0) + (completion or 0)
    return {
        "prompt_tokens": _coerce_int(prompt),
        "cached_prompt_tokens": _coerce_int(cache_read),
        "cache_write_tokens": _coerce_int(cache_write),
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
    ctx = get_request_context()
    user_id = (meta or {}).get("user_id") or ctx.get("user_id")
    label = label or ctx.get("intent")
    payload = {
        "usage": usage or {},
        "label": label,
        **ctx,
        **(meta or {})
    }
    meta_json = json.dumps(payload, ensure_ascii=False)
    _accumulate_request_totals(fields["total_tokens"], cost)

    logger.debug(
        "record(): model=%s label=%s user=%s prompt=%s cached=%s cache_write=%s completion=%s total=%s cost=%s",
        model, label, user_id, fields["prompt_tokens"],
        fields["cached_prompt_tokens"], fields["cache_write_tokens"],
        fields["completion_tokens"], fields["total_tokens"], cost
    )

    with _conn_rw() as c:
        c.execute(
            """
            INSERT INTO usage_logs
              (ts_utc, model, label, user_id, prompt_tokens, cached_prompt_tokens,
               cache_write_tokens, completion_tokens, total_tokens, cost_usd, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_utc_iso(),
                model,
                label,
                str(user_id) if user_id else None,
                fields["prompt_tokens"],
                fields["cached_prompt_tokens"],
                fields["cache_write_tokens"],
                fields["completion_tokens"],
                fields["total_tokens"],
                _coerce_float(cost),
                meta_json,
            ),
        )
    publish_metrics_snapshot()


def record_response(model: str, response: Any, *, label: Optional[str] = None) -> None:
    """Record usage straight from an OpenAI SDK response object (chat or
    responses API). Cost is estimated from the pricing table. Never raises."""
    try:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return
        if hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = usage_obj
        else:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", 0),
            }
        actual_model = getattr(response, "model", None) or model
        record(actual_model, usage, estimate_cost(actual_model, usage), label=label)
    except Exception:
        logger.warning("record_response failed", exc_info=True)


def record_metered(
    service: str,
    cost_usd: float,
    *,
    label: Optional[str] = None,
    meta: Dict[str, Any] | None = None,
) -> None:
    """Record a per-call provider charge that is not token-metered."""
    record(
        service,
        {},
        _coerce_float(cost_usd),
        label=label or "metered_tool",
        meta={"metered_call": True, **(meta or {})},
    )

def last() -> Dict[str, Any]:
    """
    Return the most recent single exchange. If none exist, {}.
    """
    with _conn_rw() as c:
        cur = c.execute(
            """
            SELECT ts_utc, model, label, prompt_tokens, cached_prompt_tokens,
                   cache_write_tokens, completion_tokens, total_tokens, cost_usd, meta_json
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
        "cached_prompt_tokens": row[4],
        "cache_write_tokens": row[5],
        "completion_tokens": row[6],
        "total_tokens": row[7],
        "cost": float(row[8]),
        "meta": json.loads(row[9] or "{}"),
    }
    logger.debug("last(): %s", out)
    return out

def _aggregate_where(where_sql: str, args: tuple) -> Dict[str, Any]:
    sql = f"""
    SELECT
      COUNT(*)                               AS calls,
      COALESCE(SUM(prompt_tokens), 0)        AS prompt_tokens,
      COALESCE(SUM(cached_prompt_tokens), 0) AS cached_prompt_tokens,
      COALESCE(SUM(cache_write_tokens), 0)   AS cache_write_tokens,
      COALESCE(SUM(completion_tokens), 0)    AS completion_tokens,
      COALESCE(SUM(total_tokens), 0)         AS total_tokens,
      COALESCE(SUM(cost_usd), 0.0)           AS cost
    FROM usage_logs
    WHERE {where_sql}
    """
    with _conn_rw() as c:
        cur = c.execute(sql, args)
        row = cur.fetchone() or (0, 0, 0, 0, 0, 0, 0.0)
    out = {
        "calls": row[0],
        "prompt_tokens": row[1],
        "cached_prompt_tokens": row[2],
        "cache_write_tokens": row[3],
        "completion_tokens": row[4],
        "total_tokens": row[5],
        "cost": float(row[6]),
    }
    logger.debug("_aggregate_where(%s, %s): %s", where_sql, args, out)
    return out

def window_minutes(minutes: int = 60) -> Dict[str, Any]:
    minutes = max(1, int(minutes or 60))
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return _aggregate_where("ts_utc >= ?", (since.replace(microsecond=0).isoformat(),))

def today() -> Dict[str, Any]:
    return _aggregate_where("ts_utc >= ?", (_report_day_start_iso(),))

def month_to_date() -> Dict[str, Any]:
    return _aggregate_where("ts_utc >= ?", (_report_month_start_iso(),))

def year_to_date() -> Dict[str, Any]:
    return _aggregate_where("ts_utc >= ?", (_report_year_start_iso(),))

def all_time() -> Dict[str, Any]:
    """All usage ever recorded in this database."""
    return _aggregate_where("1=1", ())


def _public_metrics_snapshot(now_utc: datetime | None = None) -> Dict[str, Any]:
    """Build a public aggregate without identity, prompts, labels, or models."""
    now = now_utc or datetime.now(timezone.utc)
    boundaries = (
        _day_start_utc(now, REPORT_TZ).isoformat(),
        _week_start_utc(now, REPORT_TZ).isoformat(),
        _month_start_utc(now, REPORT_TZ).isoformat(),
    )
    local_today = now.astimezone(REPORT_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    series_start_local = local_today - timedelta(days=14)
    with _conn_rw() as c:
        rows = c.execute(
            """
            SELECT
              MAX(ts_utc),
              COUNT(*),
              COALESCE(SUM(prompt_tokens), 0),
              COALESCE(SUM(cached_prompt_tokens), 0),
              COALESCE(SUM(cache_write_tokens), 0),
              COALESCE(SUM(completion_tokens), 0),
              COALESCE(SUM(total_tokens), 0),
              COUNT(CASE WHEN ts_utc >= ? THEN 1 END),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cached_prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cache_write_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN completion_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN total_tokens ELSE 0 END), 0),
              COUNT(CASE WHEN ts_utc >= ? THEN 1 END),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cached_prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cache_write_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN completion_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN total_tokens ELSE 0 END), 0),
              COUNT(CASE WHEN ts_utc >= ? THEN 1 END),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cached_prompt_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN cache_write_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN completion_tokens ELSE 0 END), 0),
              COALESCE(SUM(CASE WHEN ts_utc >= ? THEN total_tokens ELSE 0 END), 0)
            FROM usage_logs
            """,
            (boundaries[0],) * 6 + (boundaries[1],) * 6 + (boundaries[2],) * 6,
        ).fetchone()
        daily_rows = c.execute(
            "SELECT ts_utc, total_tokens FROM usage_logs WHERE ts_utc >= ? ORDER BY ts_utc",
            (series_start_local.astimezone(timezone.utc).isoformat(),),
        ).fetchall()

    daily_totals = {
        (series_start_local + timedelta(days=offset)).date().isoformat(): 0
        for offset in range(14)
    }
    for ts_utc, total_tokens in daily_rows:
        try:
            day = datetime.fromisoformat(ts_utc).astimezone(REPORT_TZ).date().isoformat()
        except (TypeError, ValueError):
            continue
        if day in daily_totals:
            daily_totals[day] += int(total_tokens or 0)

    def window(offset: int) -> Dict[str, int]:
        return {
            "calls": int(rows[offset] or 0),
            "promptTokens": int(rows[offset + 1] or 0),
            "cachedPromptTokens": int(rows[offset + 2] or 0),
            "cacheWriteTokens": int(rows[offset + 3] or 0),
            "completionTokens": int(rows[offset + 4] or 0),
            "totalTokens": int(rows[offset + 5] or 0),
        }

    return {
        "schema": 1,
        "generatedAt": now.replace(microsecond=0).isoformat(),
        "timezone": _REPORT_TZ_NAME,
        "lastActivityAt": rows[0],
        "daily": {
            "from": next(iter(daily_totals)),
            "to": next(reversed(daily_totals)),
            "days": list(daily_totals.values()),
        },
        "windows": {
            "today": window(7),
            "thisWeek": window(13),
            "monthToDate": window(19),
            "allTime": window(1),
        },
    }


def publish_metrics_snapshot() -> bool:
    """Atomically publish privacy-safe usage aggregates when configured."""
    if not METRICS_PATH:
        return False
    temporary = None
    try:
        destination = os.path.abspath(METRICS_PATH)
        directory = os.path.dirname(destination)
        os.makedirs(directory, exist_ok=True)
        payload = _public_metrics_snapshot()
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".usage-metrics-", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        return True
    except Exception:
        logger.warning("usage metrics snapshot publish failed", exc_info=True)
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        return False

def today_breakdown(user_id: Optional[str] = None, limit: int = 12) -> list:
    """Per model+label rollup for today, most expensive first.
    user_id=None aggregates everyone."""
    where = "ts_utc >= ?"
    args: tuple = (_report_day_start_iso(),)
    if user_id is not None:
        where += " AND user_id = ?"
        args += (str(user_id),)
    with _conn_rw() as c:
        rows = c.execute(
            f"""
            SELECT model, COALESCE(label, 'other'), COUNT(*),
                   COALESCE(SUM(total_tokens), 0), COALESCE(SUM(cost_usd), 0.0)
            FROM usage_logs
            WHERE {where}
            GROUP BY model, COALESCE(label, 'other')
            ORDER BY SUM(cost_usd) DESC, COUNT(*) DESC
            LIMIT ?
            """,
            args + (int(limit),),
        ).fetchall()
    return [
        {"model": r[0], "label": r[1], "calls": r[2], "total_tokens": r[3], "cost": float(r[4])}
        for r in rows
    ]


def today_for_user(user_id: str) -> Dict[str, Any]:
    return _aggregate_where("ts_utc >= ? AND user_id = ?", (_report_day_start_iso(), str(user_id)))

def _top_users_since(since_iso: str, limit: int) -> list:
    with _conn_rw() as c:
        rows = c.execute(
            """
            SELECT user_id, COUNT(*), COALESCE(SUM(total_tokens),0), COALESCE(SUM(cost_usd),0.0)
            FROM usage_logs
            WHERE ts_utc >= ? AND user_id IS NOT NULL
            GROUP BY user_id ORDER BY SUM(cost_usd) DESC LIMIT ?
            """,
            (since_iso, int(limit)),
        ).fetchall()
    return [
        {"user_id": r[0], "calls": r[1], "total_tokens": r[2], "cost": float(r[3])}
        for r in rows
    ]


def top_users_today(limit: int = 5) -> list:
    return _top_users_since(_report_day_start_iso(), limit)


def top_users_month(limit: int = 5) -> list:
    """Highest-spending users this local month (mod view)."""
    return _top_users_since(_report_month_start_iso(), limit)

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
