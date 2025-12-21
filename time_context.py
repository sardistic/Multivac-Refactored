# time_context.py
from __future__ import annotations
from datetime import datetime, timezone, timedelta

def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" + ("" if n == 1 else "s")

def time_ago_str(ts_iso: str, *, now: datetime | None = None, max_units: int = 2) -> str:
    dt = _parse_iso(ts_iso)
    if now is None:
        now = datetime.now(timezone.utc)
    delta = now - dt
    if delta < timedelta(seconds=1):
        return "just now"

    seconds = int(delta.total_seconds())
    minutes, s = divmod(seconds, 60)
    hours, m = divmod(minutes, 60)
    days, h = divmod(hours, 24)
    weeks, d = divmod(days, 7)
    months, w = divmod(weeks, 4)
    years, mo = divmod(months, 12)

    parts = []
    if years:  parts.append(_plural(years, "year"))
    if mo:     parts.append(_plural(mo, "month"))
    if not parts:
        if weeks:  parts.append(_plural(weeks, "week"))
        if d:      parts.append(_plural(d, "day"))
    if not parts:
        if hours:  parts.append(_plural(hours, "hour"))
        if m:      parts.append(_plural(m, "minute"))
    if not parts:
        parts.append(_plural(s, "second"))
    return " ".join(parts[:max_units])

def abs_time_str(ts_iso: str) -> str:
    dt = _parse_iso(ts_iso).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")

def describe_when(ts_iso: str) -> str:
    return f"{time_ago_str(ts_iso)} ago ({abs_time_str(ts_iso)})"
