"""Read only, aggregate reporting. Never expose usage metadata or user IDs."""

import sqlite3
import json
import os
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def overview(path, days=30, tz_name='America/New_York'):
    if days not in (1, 7, 30, 90):
        raise ValueError('Choose 1, 7, 30, or 90 days.')
    tz = ZoneInfo(tz_name)
    now = datetime.now(timezone.utc)
    start = (now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
             - timedelta(days=days - 1)).astimezone(timezone.utc).isoformat()
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        # Identifiers here are fixed application expressions, never request text.
        fields = """COUNT(*) AS calls, COALESCE(SUM(prompt_tokens),0) AS input_tokens,
            COALESCE(SUM(cached_prompt_tokens),0) AS cached_tokens,
            COALESCE(SUM(cache_write_tokens),0) AS cache_write_tokens,
            COALESCE(SUM(completion_tokens),0) AS output_tokens,
            COALESCE(SUM(total_tokens),0) AS tokens,
            COALESCE(SUM(cost_usd),0) AS estimated_usd"""
        total = dict(conn.execute(f'SELECT {fields} FROM usage_logs WHERE ts_utc>=?', (start,)).fetchone())
        groups = {}
        for key, expr in (
            ('servers', "COALESCE(CAST(json_extract(CASE WHEN json_valid(meta_json) THEN meta_json ELSE '{}' END,'$.guild_id') AS TEXT),'unattributed')"),
            ('models', "COALESCE(model,'unknown')"),
            ('daily', 'substr(ts_utc,1,10)'),
        ):
            groups[key] = [dict(row) for row in conn.execute(
                f'SELECT {expr} AS key,{fields} FROM usage_logs WHERE ts_utc>=? GROUP BY 1 ORDER BY tokens DESC', (start,))]
        groups['daily'].sort(key=lambda row: row['key'])
        last = conn.execute('SELECT MAX(ts_utc) FROM usage_logs').fetchone()[0]
    return dict(total=total, **groups, days=days, since=start, timezone=tz_name,
                daily_timezone='UTC', last_activity_at=last, generated_at=now.isoformat())


def publish_reports(usage_path, report_path, tz_name='America/New_York'):
    """The bot reads its WAL ledger; the web service receives only aggregates."""
    payload = {str(days): overview(usage_path, days, tz_name) for days in (1, 7, 30, 90)}
    target = Path(report_path)
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf8', dir=target.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_report(path, days):
    if days not in (1, 7, 30, 90):
        raise ValueError('Choose 1, 7, 30, or 90 days.')
    with Path(path).open(encoding='utf8') as handle:
        return json.load(handle)[str(days)]
