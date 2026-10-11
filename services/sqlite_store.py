from __future__ import annotations

import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Iterator


_PROPOSAL_ADJECTIVES = (
    "amber", "brisk", "cobalt", "coral", "gentle", "golden", "indigo", "lively",
    "lunar", "quiet", "silver", "solar", "velvet", "violet", "warm", "wild",
)
_PROPOSAL_NOUNS = (
    "badger", "birch", "cedar", "comet", "falcon", "fern", "fox", "harbor",
    "heron", "juniper", "kiwi", "lantern", "maple", "mango", "otter", "peach",
    "pepper", "plum", "raven", "river", "robin", "saffron", "spruce", "willow",
    "wren", "zephyr",
)
_PROPOSAL_MOTIONS = (
    "bloom", "drift", "echo", "glide", "orbit", "ripple", "roam", "spark",
    "trail", "turn", "wake", "wander",
)


@dataclass(frozen=True)
class DatabasePaths:
    logs_db: Path
    locations_db: Path


class SQLiteStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or Path(
            os.environ.get("MULTIVAC_STATE_DIR", Path(__file__).resolve().parent.parent)
        )
        root.mkdir(parents=True, exist_ok=True)
        self.paths = DatabasePaths(
            logs_db=root / "conversation_history.db",
            locations_db=root / "user_locations.db",
        )
        self._lock = RLock()
        self._initialize()

    @contextmanager
    def logs_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.paths.logs_db)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def locations_conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(self.paths.locations_db)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

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
        change_id = self.propose_behavior_change(
            user_id,
            instruction,
            created_by=user_id,
            source="behavior_tool",
        )
        self.activate_behavior_change(user_id, change_id)

    def _set_active_user_instruction(self, conn, user_id: str, instruction: str) -> None:
        """Update the compatibility projection used by the chat hot path."""
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

    def propose_behavior_change(
        self,
        user_id: str,
        instruction: str,
        *,
        created_by: str | None = None,
        source: str = "discord_command",
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            active = conn.execute(
                """
                SELECT id FROM behavior_changes
                WHERE scope_type='user' AND scope_id=? AND status='active'
                ORDER BY id DESC LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
            cur = conn.execute(
                """
                INSERT INTO behavior_changes
                    (scope_type, scope_id, instruction, status, parent_id,
                     created_by, source, created_at)
                VALUES ('user', ?, ?, 'draft', ?, ?, ?, ?)
                """,
                (
                    str(user_id),
                    instruction.strip(),
                    active[0] if active else None,
                    str(created_by or user_id),
                    source,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def activate_behavior_change(self, user_id: str, change_id: int) -> dict:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            row = conn.execute(
                """
                SELECT id, instruction, status FROM behavior_changes
                WHERE id=? AND scope_type='user' AND scope_id=?
                """,
                (int(change_id), str(user_id)),
            ).fetchone()
            if not row:
                raise ValueError("Behavior change not found for this user")
            if row[2] == "rolled_back":
                raise ValueError("A rolled-back change cannot be reactivated")

            conn.execute(
                """
                UPDATE behavior_changes SET status='superseded'
                WHERE scope_type='user' AND scope_id=? AND status='active' AND id<>?
                """,
                (str(user_id), int(change_id)),
            )
            conn.execute(
                "UPDATE behavior_changes SET status='active', activated_at=? WHERE id=?",
                (now, int(change_id)),
            )
            self._set_active_user_instruction(conn, str(user_id), row[1])
            conn.commit()
        return self.get_behavior_change(user_id, change_id)

    def rollback_behavior_change(self, user_id: str) -> dict | None:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            current = conn.execute(
                """
                SELECT id, parent_id FROM behavior_changes
                WHERE scope_type='user' AND scope_id=? AND status='active'
                ORDER BY id DESC LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()
            if not current:
                return None

            conn.execute(
                "UPDATE behavior_changes SET status='rolled_back', rolled_back_at=? WHERE id=?",
                (now, current[0]),
            )
            restored = None
            if current[1] is not None:
                restored = conn.execute(
                    """
                    SELECT id, instruction FROM behavior_changes
                    WHERE id=? AND scope_type='user' AND scope_id=?
                    """,
                    (current[1], str(user_id)),
                ).fetchone()
            if restored:
                conn.execute(
                    "UPDATE behavior_changes SET status='active', activated_at=? WHERE id=?",
                    (now, restored[0]),
                )
                self._set_active_user_instruction(conn, str(user_id), restored[1])
            else:
                self._set_active_user_instruction(conn, str(user_id), "")
            conn.commit()

        return self.get_behavior_change(user_id, restored[0]) if restored else None

    def get_behavior_change(self, user_id: str, change_id: int) -> dict | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                """
                SELECT id, instruction, status, parent_id, created_by, source,
                       created_at, activated_at, rolled_back_at
                FROM behavior_changes
                WHERE id=? AND scope_type='user' AND scope_id=?
                """,
                (int(change_id), str(user_id)),
            ).fetchone()
        if not row:
            return None
        keys = (
            "id", "instruction", "status", "parent_id", "created_by", "source",
            "created_at", "activated_at", "rolled_back_at",
        )
        return dict(zip(keys, row))

    def list_behavior_changes(self, user_id: str, limit: int = 10) -> list[dict]:
        limit = max(1, min(int(limit), 50))
        with self.logs_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, instruction, status, parent_id, created_by, source,
                       created_at, activated_at, rolled_back_at
                FROM behavior_changes
                WHERE scope_type='user' AND scope_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
        keys = (
            "id", "instruction", "status", "parent_id", "created_by", "source",
            "created_at", "activated_at", "rolled_back_at",
        )
        return [dict(zip(keys, row)) for row in rows]

    # ---- Owner-reviewed executable code proposals ----

    def _new_proposal_public_id(self, conn: sqlite3.Connection) -> str:
        for _ in range(100):
            adjective = secrets.choice(_PROPOSAL_ADJECTIVES)
            noun = secrets.choice(_PROPOSAL_NOUNS)
            motion = secrets.choice(_PROPOSAL_MOTIONS)
            other_noun = secrets.choice(_PROPOSAL_NOUNS)
            if other_noun == noun:
                other_noun = _PROPOSAL_NOUNS[(_PROPOSAL_NOUNS.index(noun) + 1) % len(_PROPOSAL_NOUNS)]
            patterns = (
                f"{adjective}-{noun}",
                f"{noun}-{motion}",
                f"{noun}-{other_noun}",
                f"{adjective}-{noun}-{motion}",
            )
            value = secrets.choice(patterns)
            if not conn.execute(
                "SELECT 1 FROM code_proposals WHERE public_id=?", (value,)
            ).fetchone():
                return value
        raise RuntimeError("Could not allocate a unique proposal passphrase")

    def create_code_proposal(self, owner_id: str, request: str, baseline_sha: str) -> int:
        request = request.strip()
        if not request:
            raise ValueError("Code-change request cannot be empty")
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            public_id = self._new_proposal_public_id(conn)
            cur = conn.execute(
                """
                INSERT INTO code_proposals
                    (owner_id, public_id, request, baseline_sha, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'draft', ?, ?)
                """,
                (str(owner_id), public_id, request, baseline_sha, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def set_code_proposal_patch(self, owner_id: str, proposal_id: int, patch: str) -> dict:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            cur = conn.execute(
                """
                UPDATE code_proposals
                SET patch=?, status='patch_uploaded', validation_json=NULL,
                    reviewed_by=NULL, reviewed_at=NULL, updated_at=?
                WHERE id=? AND owner_id=? AND status NOT IN ('approved', 'rejected')
                """,
                (patch, now, int(proposal_id), str(owner_id)),
            )
            if cur.rowcount != 1:
                raise ValueError("Editable code proposal not found")
            conn.commit()
        return self.get_code_proposal(owner_id, proposal_id)

    def set_code_proposal_validation(
        self, owner_id: str, proposal_id: int, report: dict
    ) -> dict:
        now = datetime.utcnow().isoformat()
        status = "reviewable" if report.get("ok") else "validation_failed"
        with self.logs_conn() as conn:
            cur = conn.execute(
                """
                UPDATE code_proposals
                SET validation_json=?, status=?, updated_at=?
                WHERE id=? AND owner_id=? AND status NOT IN ('approved', 'rejected')
                """,
                (json.dumps(report, sort_keys=True), status, now, int(proposal_id), str(owner_id)),
            )
            if cur.rowcount != 1:
                raise ValueError("Code proposal not found or already reviewed")
            conn.commit()
        return self.get_code_proposal(owner_id, proposal_id)

    def review_code_proposal(
        self, owner_id: str, proposal_id: int, decision: str, *, reviewer_id: str
    ) -> dict:
        if decision not in {"approved", "rejected"}:
            raise ValueError("Decision must be approved or rejected")
        now = datetime.utcnow().isoformat()
        required_status = "reviewable" if decision == "approved" else None
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT status FROM code_proposals WHERE id=? AND owner_id=?",
                (int(proposal_id), str(owner_id)),
            ).fetchone()
            if not row:
                raise ValueError("Code proposal not found")
            if required_status and row[0] != required_status:
                raise ValueError("Only a successfully validated proposal can be approved")
            if row[0] in {"approved", "rejected"}:
                raise ValueError("Code proposal has already been reviewed")
            conn.execute(
                """
                UPDATE code_proposals
                SET status=?, reviewed_by=?, reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (decision, str(reviewer_id), now, now, int(proposal_id)),
            )
            conn.commit()
        return self.get_code_proposal(owner_id, proposal_id)

    def review_any_code_proposal(
        self, proposal_id: int, decision: str, *, reviewer_id: str
    ) -> dict:
        if decision not in {"approved", "rejected"}:
            raise ValueError("Decision must be approved or rejected")
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT owner_id, status FROM code_proposals WHERE id=?",
                (int(proposal_id),),
            ).fetchone()
            if not row:
                raise ValueError("Code proposal not found")
            if decision == "approved" and row[1] != "reviewable":
                raise ValueError("Only a successfully validated proposal can be approved")
            if row[1] in {"approved", "rejected"}:
                raise ValueError("Code proposal has already been reviewed")
            conn.execute(
                """
                UPDATE code_proposals
                SET status=?, reviewed_by=?, reviewed_at=?, updated_at=? WHERE id=?
                """,
                (decision, str(reviewer_id), now, now, int(proposal_id)),
            )
            conn.commit()
        return self.get_code_proposal(row[0], proposal_id)

    def set_code_proposal_approval_message(
        self, proposal_id: int, channel_id: str, message_id: str
    ) -> None:
        with self.logs_conn() as conn:
            cur = conn.execute(
                """
                UPDATE code_proposals
                SET approval_channel_id=?, approval_message_id=?
                WHERE id=? AND status='approved'
                """,
                (str(channel_id), str(message_id), int(proposal_id)),
            )
            if cur.rowcount != 1:
                raise ValueError("Approved code proposal not found")
            conn.commit()

    def get_code_proposal(self, owner_id: str, proposal_id: int) -> dict | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                """
                SELECT id, public_id, owner_id, request, baseline_sha, patch, status,
                       validation_json, created_at, updated_at, reviewed_by, reviewed_at
                FROM code_proposals WHERE id=? AND owner_id=?
                """,
                (int(proposal_id), str(owner_id)),
            ).fetchone()
        if not row:
            return None
        keys = (
            "id", "public_id", "owner_id", "request", "baseline_sha", "patch", "status",
            "validation", "created_at", "updated_at", "reviewed_by", "reviewed_at",
        )
        result = dict(zip(keys, row))
        result["validation"] = json.loads(result["validation"]) if result["validation"] else None
        return result

    def get_any_code_proposal(self, proposal_id: int) -> dict | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT owner_id FROM code_proposals WHERE id=?", (int(proposal_id),)
            ).fetchone()
        return self.get_code_proposal(row[0], proposal_id) if row else None

    def list_code_proposals(self, owner_id: str, limit: int = 10) -> list[dict]:
        limit = max(1, min(int(limit), 50))
        with self.logs_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, public_id, request, baseline_sha, status, created_at, updated_at
                FROM code_proposals WHERE owner_id=? ORDER BY id DESC LIMIT ?
                """,
                (str(owner_id), limit),
            ).fetchall()
        keys = ("id", "public_id", "request", "baseline_sha", "status", "created_at", "updated_at")
        return [dict(zip(keys, row)) for row in rows]

    def list_all_code_proposals(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self.logs_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, public_id, owner_id, request, baseline_sha, status, created_at, updated_at
                FROM code_proposals ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = ("id", "public_id", "owner_id", "request", "baseline_sha", "status", "created_at", "updated_at")
        return [dict(zip(keys, row)) for row in rows]

    def get_code_deployment(self, owner_id: str, proposal_id: int) -> dict | None:
        with self.logs_conn() as conn:
            try:
                row = conn.execute(
                    """
                    SELECT d.proposal_id, d.status, d.release_path, d.patch_sha256,
                           d.previous_release, d.created_at, d.activated_at,
                           d.finished_at, d.detail
                    FROM code_deployments d
                    JOIN code_proposals p ON p.id=d.proposal_id
                    WHERE d.proposal_id=? AND p.owner_id=?
                    """,
                    (int(proposal_id), str(owner_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        if not row:
            return None
        keys = (
            "proposal_id", "status", "release_path", "patch_sha256",
            "previous_release", "created_at", "activated_at", "finished_at", "detail",
        )
        return dict(zip(keys, row))

    def request_code_rollback(self, owner_id: str, proposal_id: int) -> int:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            deployment = conn.execute(
                """
                SELECT d.status FROM code_deployments d
                JOIN code_proposals p ON p.id=d.proposal_id
                WHERE d.proposal_id=? AND p.owner_id=?
                """,
                (int(proposal_id), str(owner_id)),
            ).fetchone()
            if not deployment or deployment[0] != "active":
                raise ValueError("Only your currently active deployment can be rolled back")
            cur = conn.execute(
                """
                INSERT INTO code_control_requests
                    (proposal_id, owner_id, action, status, created_at)
                VALUES (?, ?, 'rollback', 'pending', ?)
                """,
                (int(proposal_id), str(owner_id), now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def request_any_code_rollback(self, reviewer_id: str, proposal_id: int) -> int:
        now = datetime.utcnow().isoformat()
        with self.logs_conn() as conn:
            deployment = conn.execute(
                "SELECT status FROM code_deployments WHERE proposal_id=?",
                (int(proposal_id),),
            ).fetchone()
            if not deployment or deployment[0] != "active":
                raise ValueError("Only the currently active deployment can be rolled back")
            cur = conn.execute(
                """
                INSERT INTO code_control_requests
                    (proposal_id, owner_id, action, status, created_at)
                VALUES (?, ?, 'rollback', 'pending', ?)
                """,
                (int(proposal_id), str(reviewer_id), now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_user_instruction(self, user_id: str) -> str | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT instruction FROM user_instructions WHERE user_id=?",
                (str(user_id),),
            ).fetchone()
        return row[0] if row else None

    def get_conversation_persona_enabled(self, scope_key: str) -> bool:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT enabled FROM conversation_personas WHERE scope_key=?",
                (str(scope_key),),
            ).fetchone()
        return True if row is None else bool(row[0])

    def get_model_preferences(self, user_id: str | int) -> dict[str, str]:
        with self.logs_conn() as conn:
            return dict(conn.execute("SELECT workload,tier FROM user_model_preferences WHERE user_id=?", (str(user_id),)))

    def set_model_preferences(self, user_id: str | int, preferences: dict[str, str]) -> None:
        with self.logs_conn() as conn:
            for workload, tier in preferences.items():
                if tier == "auto":
                    conn.execute("DELETE FROM user_model_preferences WHERE user_id=? AND workload=?", (str(user_id), workload))
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO user_model_preferences VALUES(?,?,?,?)",
                        (str(user_id), workload, tier, datetime.utcnow().isoformat()),
                    )
            conn.commit()

    def set_conversation_persona_enabled(self, scope_key: str, enabled: bool) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversation_personas
                    (scope_key, persona, enabled, updated_at)
                VALUES (?, 'mistake_not', ?, ?)
                """,
                (
                    str(scope_key),
                    1 if enabled else 0,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

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

    # ---- Per-user long-term facts (model-invoked via remember_fact tool) ----

    def add_user_fact(self, user_id: str, fact: str, category: str | None = None) -> int:
        with self.logs_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO user_facts (user_id, fact, category, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(user_id), fact, category, datetime.utcnow().isoformat()),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_user_facts(self, user_id: str, limit: int = 50) -> list[dict]:
        with self.logs_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, fact, category, created_at FROM user_facts
                WHERE user_id = ? ORDER BY id DESC LIMIT ?
                """,
                (str(user_id), int(limit)),
            ).fetchall()
        return [
            {"id": r[0], "fact": r[1], "category": r[2], "created_at": r[3]}
            for r in rows
        ]

    def delete_user_fact(self, user_id: str, fact_id: int) -> bool:
        with self.logs_conn() as conn:
            cur = conn.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND id = ?",
                (str(user_id), int(fact_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_user_facts_matching(self, user_id: str, query: str) -> int:
        with self.logs_conn() as conn:
            cur = conn.execute(
                "DELETE FROM user_facts WHERE user_id = ? AND fact LIKE ?",
                (str(user_id), f"%{query}%"),
            )
            conn.commit()
            return cur.rowcount

    # ---- Distilled per-user profile (background summarization) ----

    def get_user_profile(self, user_id: str) -> dict | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT profile, updated_at FROM user_profiles WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return {"profile": row[0], "updated_at": row[1]} if row else None

    def set_user_profile(self, user_id: str, profile: str) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_profiles (user_id, profile, updated_at)
                VALUES (?, ?, ?)
                """,
                (str(user_id), profile, datetime.utcnow().isoformat()),
            )
            conn.commit()

    # ---- Condensed YouTube transcript cache (one condensation cost per video) ----

    def get_cached_transcript_summary(self, video_id: str) -> str | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT summary FROM yt_transcript_cache WHERE video_id = ?",
                (str(video_id),),
            ).fetchone()
        return row[0] if row else None

    def set_cached_transcript_summary(self, video_id: str, summary: str) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO yt_transcript_cache (video_id, summary, created_at)
                VALUES (?, ?, ?)
                """,
                (str(video_id), summary, datetime.utcnow().isoformat()),
            )
            conn.commit()

    # ---- Last-interaction tracking (time-passage awareness, intent continuity) ----

    def get_user_seen(self, user_id: str) -> dict | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT last_seen_at, last_intent, last_prompt FROM user_seen WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return (
            {"last_seen_at": row[0], "last_intent": row[1], "last_prompt": row[2]}
            if row
            else None
        )

    def set_user_seen(self, user_id: str, *, intent: str | None = None, prompt: str | None = None) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_seen (user_id, last_seen_at, last_intent, last_prompt)
                VALUES (?, ?, ?, ?)
                """,
                (str(user_id), datetime.utcnow().isoformat(), intent, (prompt or "")[:500]),
            )
            conn.commit()

    def get_channel_last_seen(self, key: str) -> str | None:
        with self.logs_conn() as conn:
            row = conn.execute(
                "SELECT last_seen_id FROM channel_state WHERE key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def set_channel_last_seen(self, key: str, last_seen_id: str) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO channel_state (key, last_seen_id, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, str(last_seen_id), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def log_sora_usage(self, user_id: str, video_id: str | None = None) -> None:
        self._log_video_usage("sora_usage", user_id, video_id=video_id)

    def log_veo_usage(self, user_id: str, video_id: str | None = None) -> None:
        self._log_video_usage("veo_usage", user_id, video_id=video_id)

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
        return self._check_usage_limit("sora_usage", user_id, limit=limit, window_seconds=window_seconds)

    def check_veo_limit(self, user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
        return self._check_usage_limit("veo_usage", user_id, limit=limit, window_seconds=window_seconds)

    def sora_limit_status(self, user_id: str, limit: int = 2, window_seconds: int = 3600) -> dict:
        return self._usage_limit_status("sora_usage", user_id, limit=limit, window_seconds=window_seconds)

    def veo_limit_status(self, user_id: str, limit: int = 2, window_seconds: int = 3600) -> dict:
        return self._usage_limit_status("veo_usage", user_id, limit=limit, window_seconds=window_seconds)

    def _usage_limit_status(self, table_name: str, user_id: str, limit: int, window_seconds: int) -> dict:
        """Like _check_usage_limit but also reports remaining uses and when the
        oldest in-window use expires (so refusals can say 'resets in 23m')."""
        whitelist = {"54277066459193344", "54280542740287488"}
        if str(user_id) in whitelist:
            return {"allowed": True, "remaining": limit, "resets_in_seconds": 0}

        with self.logs_conn() as conn:
            rows = conn.execute(
                f"SELECT timestamp FROM {table_name} WHERE user_id = ?",
                (str(user_id),),
            ).fetchall()

        now = datetime.utcnow()
        in_window: list[datetime] = []
        for (ts_str,) in rows:
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            if (now - ts).total_seconds() < window_seconds:
                in_window.append(ts)

        remaining = max(0, limit - len(in_window))
        resets_in = 0
        if in_window and remaining == 0:
            oldest = min(in_window)
            resets_in = max(0, int(window_seconds - (now - oldest).total_seconds()))
        return {"allowed": remaining > 0, "remaining": remaining, "resets_in_seconds": resets_in}

    def _log_video_usage(self, table_name: str, user_id: str, video_id: str | None = None) -> None:
        with self.logs_conn() as conn:
            conn.execute(
                f"INSERT INTO {table_name} (user_id, video_id, timestamp) VALUES (?, ?, ?)",
                (str(user_id), str(video_id) if video_id else None, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def _check_usage_limit(self, table_name: str, user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
        whitelist = {"54277066459193344", "54280542740287488"}
        if str(user_id) in whitelist:
            return True

        with self.logs_conn() as conn:
            rows = conn.execute(
                f"SELECT timestamp FROM {table_name} WHERE user_id = ?",
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

                CREATE TABLE IF NOT EXISTS behavior_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope_type TEXT NOT NULL CHECK(scope_type IN ('user', 'guild', 'global')),
                    scope_id TEXT NOT NULL,
                    instruction TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN ('draft', 'active', 'superseded', 'rolled_back')),
                    parent_id INTEGER,
                    created_by TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    rolled_back_at TEXT,
                    FOREIGN KEY(parent_id) REFERENCES behavior_changes(id)
                );
                CREATE INDEX IF NOT EXISTS idx_behavior_changes_scope
                    ON behavior_changes (scope_type, scope_id, id DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_behavior_changes_one_active
                    ON behavior_changes (scope_type, scope_id) WHERE status='active';

                INSERT INTO behavior_changes
                    (scope_type, scope_id, instruction, status, parent_id,
                     created_by, source, created_at, activated_at)
                SELECT 'user', ui.user_id, ui.instruction, 'active', NULL,
                       ui.user_id, 'legacy_migration',
                       COALESCE(ui.updated_at, datetime('now')),
                       COALESCE(ui.updated_at, datetime('now'))
                FROM user_instructions AS ui
                WHERE COALESCE(ui.instruction, '') <> ''
                  AND NOT EXISTS (
                      SELECT 1 FROM behavior_changes AS bc
                      WHERE bc.scope_type='user' AND bc.scope_id=ui.user_id
                  );

                CREATE TABLE IF NOT EXISTS code_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    public_id TEXT UNIQUE,
                    owner_id TEXT NOT NULL,
                    request TEXT NOT NULL,
                    baseline_sha TEXT NOT NULL,
                    patch TEXT,
                    status TEXT NOT NULL CHECK(status IN (
                        'draft', 'patch_uploaded', 'validation_failed',
                        'reviewable', 'approved', 'rejected'
                    )),
                    validation_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    approval_channel_id TEXT,
                    approval_message_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_code_proposals_owner
                    ON code_proposals (owner_id, id DESC);

                CREATE TABLE IF NOT EXISTS code_control_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id INTEGER NOT NULL,
                    owner_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('rollback')),
                    status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'failed')),
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    detail TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_code_control_pending
                    ON code_control_requests (proposal_id, action) WHERE status='pending';

                CREATE TABLE IF NOT EXISTS sora_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    video_id TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS veo_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    video_id TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS channel_state (
                    key TEXT PRIMARY KEY,
                    last_seen_id TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS conversation_personas (
                    scope_key TEXT PRIMARY KEY,
                    persona TEXT NOT NULL DEFAULT 'mistake_not',
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_model_preferences (
                    user_id TEXT NOT NULL,
                    workload TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, workload)
                );

                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    category TEXT,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_user_facts_user ON user_facts (user_id);

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS user_seen (
                    user_id TEXT PRIMARY KEY,
                    last_seen_at TEXT,
                    last_intent TEXT,
                    last_prompt TEXT
                );

                CREATE TABLE IF NOT EXISTS yt_transcript_cache (
                    video_id TEXT PRIMARY KEY,
                    summary TEXT,
                    created_at TEXT
                );
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(code_proposals)")}
            if "public_id" not in columns:
                conn.execute("ALTER TABLE code_proposals ADD COLUMN public_id TEXT")
            if "approval_channel_id" not in columns:
                conn.execute("ALTER TABLE code_proposals ADD COLUMN approval_channel_id TEXT")
            if "approval_message_id" not in columns:
                conn.execute("ALTER TABLE code_proposals ADD COLUMN approval_message_id TEXT")
            for row in conn.execute("SELECT id FROM code_proposals WHERE public_id IS NULL"):
                conn.execute(
                    "UPDATE code_proposals SET public_id=? WHERE id=?",
                    (self._new_proposal_public_id(conn), row[0]),
                )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_code_proposals_public_id ON code_proposals(public_id)"
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
