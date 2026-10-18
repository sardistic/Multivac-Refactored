"""Private invite requests and audit trail, shared with the bot admission gate."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def admission_allowed(path, guild_id):
    """Fail closed without creating a missing database or altering its schema."""
    if guild_id is None or str(guild_id) == 'DM' or not path:
        return True
    try:
        conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
        try:
            row = conn.execute('SELECT status FROM requests WHERE guild_id=?', (str(guild_id),)).fetchone()
            return bool(row and row[0] == 'approved')
        finally:
            conn.close()
    except sqlite3.Error:
        return False


class PortalStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    guild_id TEXT PRIMARY KEY, guild_name TEXT NOT NULL,
                    user_id TEXT NOT NULL, user_name TEXT NOT NULL,
                    reason TEXT NOT NULL, status TEXT NOT NULL,
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                    review_note TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY, at INTEGER NOT NULL,
                    actor_id TEXT NOT NULL, guild_id TEXT NOT NULL,
                    action TEXT NOT NULL, note TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS kofi (
                    message_id TEXT PRIMARY KEY, at INTEGER NOT NULL,
                    kind TEXT NOT NULL, amount_cents INTEGER NOT NULL, currency TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def seed_existing(self, guilds, owner_id):
        """One explicit operator bootstrap; never auto-approve on bot startup."""
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM settings WHERE key='seeded'").fetchone():
                raise ValueError("Existing servers have already been seeded")
            now = int(time.time())
            for guild in guilds:
                conn.execute("INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?,?,?,?,?)", (
                    str(guild['id']), guild['name'], str(owner_id), 'Existing installation',
                    'Installed before invite review was enabled', 'approved', now, now, '',
                ))
                self._audit(conn, owner_id, guild['id'], 'grandfathered', '')
            conn.execute("INSERT INTO settings VALUES('seeded', ?)", (json.dumps(now),))

    @staticmethod
    def _audit(conn, actor, guild, action, note):
        conn.execute("INSERT INTO audit(at,actor_id,guild_id,action,note) VALUES(?,?,?,?,?)",
                     (int(time.time()), str(actor), str(guild), action, note))

    def allowed(self, guild_id):
        with self.connection() as conn:
            row = conn.execute("SELECT status FROM requests WHERE guild_id=?", (str(guild_id),)).fetchone()
        return bool(row and row['status'] == 'approved')

    def request(self, guild, user, reason):
        reason = str(reason).strip()
        if not 10 <= len(reason) <= 2000:
            raise ValueError("Describe your intended use in 10–2000 characters.")
        now = int(time.time())
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT status,updated_at FROM requests WHERE guild_id=?", (guild['id'],)).fetchone()
            if old and old['status'] in ('pending', 'approved', 'revoked'):
                raise ValueError("This server already has a request or requires owner review.")
            if old and now - old['updated_at'] < 86400:
                raise ValueError("Please wait 24 hours before requesting another review.")
            conn.execute("INSERT OR REPLACE INTO requests VALUES(?,?,?,?,?,?,?,?,?)", (
                guild['id'], guild['name'][:200], user['id'], user['username'][:100],
                reason, 'pending', now, now, '',
            ))
            self._audit(conn, user['id'], guild['id'], 'requested', reason)

    def review(self, guild_id, actor_id, status, note):
        if status not in ('approved', 'rejected', 'revoked'):
            raise ValueError("Choose approve, reject, or revoke.")
        note = str(note).strip()
        if len(note) > 1000:
            raise ValueError("Review notes must be at most 1000 characters.")
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM requests WHERE guild_id=?", (guild_id,)).fetchone()
            if not row:
                raise ValueError("Request not found.")
            if row['status'] == status:
                return
            if status == 'rejected' and row['status'] != 'pending':
                raise ValueError("Only pending requests can be rejected; revoke approved access instead.")
            conn.execute("UPDATE requests SET status=?,review_note=?,updated_at=? WHERE guild_id=?",
                         (status, note, int(time.time()), guild_id))
            self._audit(conn, actor_id, guild_id, status, note)

    def requests(self):
        with self.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM requests ORDER BY updated_at DESC")]

    def audit(self):
        with self.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 100")]

    def record_kofi(self, message_id, at, kind, amount_cents, currency):
        """One Ko-fi payment: when, how much and what kind. Never who, or what they wrote."""
        with self.connection() as conn:
            conn.execute("INSERT OR IGNORE INTO kofi VALUES(?,?,?,?,?)",
                         (str(message_id)[:80], int(at), str(kind)[:40], int(amount_cents), str(currency)[:8].upper()))

    def kofi_payments(self):
        with self.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT at,kind,amount_cents,currency FROM kofi ORDER BY at")]
