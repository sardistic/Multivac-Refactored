from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterator


@dataclass(frozen=True)
class DatabasePaths:
    logs_db: Path
    locations_db: Path


class SQLiteStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or Path(__file__).resolve().parent.parent
        self.paths = DatabasePaths(
            logs_db=root / "conversation_history.db",
            locations_db=root / "user_locations.db",
        )
        self._lock = RLock()
        self._initialize()

    @contextmanager
    def logs_conn(self) -> Iterator[sqlite3.Connection]:
        with self._connect(self.paths.logs_db) as conn:
            yield conn

    @contextmanager
    def locations_conn(self) -> Iterator[sqlite3.Connection]:
        with self._connect(self.paths.locations_db) as conn:
            yield conn

    def log_message(self, conversation_id, user_id, user_msg, bot_msg) -> None:
        timestamp = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT INTO logs (conversation_id, user_id, user_message, bot_response, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, str(user_id), user_msg, bot_msg, timestamp),
            )
            conn.commit()

    def fetch_conversation(self, conversation_id):
        with self.logs_conn() as conn:
            rows = conn.execute(
                "SELECT user_message, bot_response FROM logs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchall()
        return rows

    def insert_or_update_user_location(self, user_id, location) -> None:
        with self.locations_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_locations (user_id, location)
                VALUES (?, ?)
                """,
                (user_id, location),
            )
            conn.commit()

    def fetch_user_location(self, user_id):
        with self.locations_conn() as conn:
            row = conn.execute(
                "SELECT location FROM user_locations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row[0] if row else None

    def save_message_expansion(self, message_id: int, full_text: str, expanded: bool = False) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO message_expansions (message_id, full_text, expanded)
                VALUES (?, ?, ?)
                """,
                (str(message_id), full_text, 1 if expanded else 0),
            )
            conn.commit()

    def get_message_expansion(self, message_id: int):
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT full_text, expanded FROM message_expansions WHERE message_id = ?",
                (str(message_id),),
            ).fetchone()
        return {"full_text": row[0], "expanded": bool(row[1])} if row else None

    def set_message_expanded(self, message_id: int, expanded: bool) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                "UPDATE message_expansions SET expanded=? WHERE message_id=?",
                (1 if expanded else 0, str(message_id)),
            )
            conn.commit()

    def set_user_instruction(self, user_id: str, instruction: str) -> None:
        with self.logs_conn() as conn:
            if not instruction:
                conn.execute("DELETE FROM user_instructions WHERE user_id=?", (str(user_id),))
            else:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_instructions (user_id, instruction, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (str(user_id), instruction, datetime.utcnow().isoformat()),
                )
            conn.commit()

    def get_user_instruction(self, user_id: str) -> str | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT instruction FROM user_instructions WHERE user_id=?",
                (str(user_id),),
            ).fetchone()
        return row[0] if row else None

    def set_memory_consent(self, user_id, consent: bool) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_memory_consent (user_id, opted_in)
                VALUES (?, ?)
                """,
                (str(user_id), int(consent)),
            )
            conn.commit()

    def has_opted_in_memory(self, user_id):
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT opted_in FROM user_memory_consent WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return bool(row and row[0] == 1)

    def log_sora_usage(self, user_id: str, video_id: str | None = None) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                "INSERT INTO sora_usage (user_id, video_id, timestamp) VALUES (?, ?, ?)",
                (str(user_id), str(video_id) if video_id else None, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_last_sora_video_id(self, user_id: str) -> str | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                """
                SELECT video_id
                FROM sora_usage
                WHERE user_id = ? AND video_id IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
        return row[0] if row else None

    def check_sora_limit(self, user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
        whitelist = {"54277066459193344", "54280542740287488"}
        if str(user_id) in whitelist:
            return True

        with self.logs_conn() as conn:
            rows = conn.execute(
                "SELECT timestamp FROM sora_usage WHERE user_id = ?",
                (str(user_id),),
            ).fetchall()

        now = datetime.utcnow()
        count = 0
        for (ts_str,) in rows:
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            if (now - ts).total_seconds() < window_seconds:
                count += 1
        return count < limit

    def _initialize(self) -> None:
        with self._lock:
            self._initialize_logs_db()
            self._initialize_locations_db()

    def _initialize_logs_db(self) -> None:
        with self.logs_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    user_id TEXT,
                    user_message TEXT,
                    bot_response TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS user_memory_consent (
                    user_id TEXT PRIMARY KEY,
                    opted_in INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS message_expansions (
                    message_id TEXT PRIMARY KEY,
                    full_text  TEXT NOT NULL,
                    expanded   INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS user_instructions (
                    user_id TEXT PRIMARY KEY,
                    instruction TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS sora_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    video_id TEXT,
                    timestamp TEXT
                );
                """
            )
            conn.commit()

    def _initialize_locations_db(self) -> None:
        with self.locations_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_locations (
                    user_id INTEGER PRIMARY KEY,
                    location TEXT
                )
                """
            )
            conn.commit()

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        return sqlite3.connect(path, check_same_thread=False)

