import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.sqlite_store import SQLiteStore
from services import usage_costs


class UserMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: sqlite3's context manager commits but doesn't
        # close, so Windows may still hold a lock on the db file at teardown.
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = SQLiteStore(base_dir=Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_fact_crud(self):
        fid = self.store.add_user_fact("u1", "Has a dog named Kevin", "relationship")
        self.store.add_user_fact("u1", "Prefers concise answers", "preference")
        self.store.add_user_fact("u2", "Other user's fact")

        facts = self.store.list_user_facts("u1")
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[-1]["id"], fid)

        deleted = self.store.delete_user_facts_matching("u1", "kevin")
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.store.list_user_facts("u1")), 1)
        # Other users untouched
        self.assertEqual(len(self.store.list_user_facts("u2")), 1)

    def test_profile_roundtrip(self):
        self.assertIsNone(self.store.get_user_profile("u1"))
        self.store.set_user_profile("u1", "- likes trains")
        prof = self.store.get_user_profile("u1")
        self.assertEqual(prof["profile"], "- likes trains")
        self.assertTrue(prof["updated_at"])

    def test_user_seen_roundtrip(self):
        self.assertIsNone(self.store.get_user_seen("u1"))
        self.store.set_user_seen("u1", intent="chat", prompt="hello there")
        seen = self.store.get_user_seen("u1")
        self.assertEqual(seen["last_intent"], "chat")
        self.assertEqual(seen["last_prompt"], "hello there")

    def test_behavior_changes_are_drafts_until_activated(self):
        change_id = self.store.propose_behavior_change("u1", "Always be concise")
        change = self.store.get_behavior_change("u1", change_id)
        self.assertEqual(change["status"], "draft")
        self.assertIsNone(self.store.get_user_instruction("u1"))

        active = self.store.activate_behavior_change("u1", change_id)
        self.assertEqual(active["status"], "active")
        self.assertEqual(self.store.get_user_instruction("u1"), "Always be concise")

    def test_behavior_activation_and_rollback_restore_parent(self):
        first = self.store.propose_behavior_change("u1", "Speak like a pirate")
        self.store.activate_behavior_change("u1", first)
        second = self.store.propose_behavior_change("u1", "Use terse technical prose")
        self.store.activate_behavior_change("u1", second)

        history = self.store.list_behavior_changes("u1")
        self.assertEqual(history[0]["status"], "active")
        self.assertEqual(history[1]["status"], "superseded")

        restored = self.store.rollback_behavior_change("u1")
        self.assertEqual(restored["id"], first)
        self.assertEqual(restored["status"], "active")
        self.assertEqual(self.store.get_user_instruction("u1"), "Speak like a pirate")
        self.assertEqual(self.store.get_behavior_change("u1", second)["status"], "rolled_back")

    def test_behavior_clear_version_rolls_back_to_prior_instruction(self):
        self.store.set_user_instruction("u1", "Be cheerful")
        self.store.set_user_instruction("u1", "")
        self.assertIsNone(self.store.get_user_instruction("u1"))

        restored = self.store.rollback_behavior_change("u1")
        self.assertEqual(restored["instruction"], "Be cheerful")
        self.assertEqual(self.store.get_user_instruction("u1"), "Be cheerful")

    def test_behavior_changes_are_user_scoped(self):
        change_id = self.store.propose_behavior_change("u1", "Private behavior")
        self.assertIsNone(self.store.get_behavior_change("u2", change_id))
        with self.assertRaises(ValueError):
            self.store.activate_behavior_change("u2", change_id)

    def test_existing_instruction_is_imported_as_active_version(self):
        import sqlite3

        legacy_dir = Path(self._tmp.name) / "legacy"
        legacy_dir.mkdir()
        conn = sqlite3.connect(legacy_dir / "conversation_history.db")
        conn.execute(
            "CREATE TABLE user_instructions (user_id TEXT PRIMARY KEY, instruction TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO user_instructions VALUES (?, ?, ?)",
            ("legacy-user", "Keep the old preference", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        migrated = SQLiteStore(base_dir=legacy_dir)
        history = migrated.list_behavior_changes("legacy-user")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "active")
        self.assertEqual(history[0]["source"], "legacy_migration")

    def test_code_proposal_review_lifecycle(self):
        proposal_id = self.store.create_code_proposal(
            "owner-1", "Add a harmless feature", "a" * 40
        )
        proposal = self.store.get_code_proposal("owner-1", proposal_id)
        self.assertEqual(proposal["status"], "draft")
        self.assertRegex(proposal["public_id"], r"^[a-z]+(?:-[a-z]+){1,2}$")
        self.assertIsNone(self.store.get_code_proposal("owner-2", proposal_id))

        proposal = self.store.set_code_proposal_patch(
            "owner-1", proposal_id, "diff --git a/a.py b/a.py"
        )
        self.assertEqual(proposal["status"], "patch_uploaded")
        with self.assertRaises(ValueError):
            self.store.review_code_proposal(
                "owner-1", proposal_id, "approved", reviewer_id="owner-1"
            )

        proposal = self.store.set_code_proposal_validation(
            "owner-1", proposal_id, {"ok": True, "files": ["a.py"], "errors": []}
        )
        self.assertEqual(proposal["status"], "reviewable")
        proposal = self.store.review_code_proposal(
            "owner-1", proposal_id, "approved", reviewer_id="owner-1"
        )
        self.assertEqual(proposal["status"], "approved")
        self.assertEqual(proposal["reviewed_by"], "owner-1")

    def test_failed_code_proposal_can_be_rejected(self):
        proposal_id = self.store.create_code_proposal("owner-1", "Bad patch", "b" * 40)
        self.store.set_code_proposal_patch("owner-1", proposal_id, "broken")
        proposal = self.store.set_code_proposal_validation(
            "owner-1", proposal_id, {"ok": False, "files": [], "errors": ["bad"]}
        )
        self.assertEqual(proposal["status"], "validation_failed")
        rejected = self.store.review_code_proposal(
            "owner-1", proposal_id, "rejected", reviewer_id="owner-1"
        )
        self.assertEqual(rejected["status"], "rejected")

    def test_privileged_reviewer_can_approve_another_users_proposal(self):
        proposal_id = self.store.create_code_proposal("user-2", "Community change", "c" * 40)
        self.store.set_code_proposal_patch("user-2", proposal_id, "diff --git a/a.py b/a.py")
        self.store.set_code_proposal_validation(
            "user-2", proposal_id, {"ok": True, "files": ["a.py"], "errors": []}
        )
        approved = self.store.review_any_code_proposal(
            proposal_id, "approved", reviewer_id="app-owner"
        )
        self.assertEqual(approved["owner_id"], "user-2")
        self.assertEqual(approved["reviewed_by"], "app-owner")
        self.assertEqual(approved["status"], "approved")
        self.store.set_code_proposal_approval_message(proposal_id, "channel-1", "message-1")
        with self.store.logs_conn() as conn:
            target = conn.execute(
                "SELECT approval_channel_id, approval_message_id FROM code_proposals WHERE id=?",
                (proposal_id,),
            ).fetchone()
        self.assertEqual(target, ("channel-1", "message-1"))

    def test_global_proposal_list_keeps_requester_identity(self):
        self.store.create_code_proposal("user-a", "First", "a" * 40)
        newest = self.store.create_code_proposal("user-b", "Second", "b" * 40)
        rows = self.store.list_all_code_proposals()
        self.assertEqual(rows[0]["id"], newest)
        self.assertEqual(rows[0]["owner_id"], "user-b")

    def test_transcript_cache_roundtrip(self):
        self.assertIsNone(self.store.get_cached_transcript_summary("vid123"))
        self.store.set_cached_transcript_summary("vid123", "condensed notes")
        self.assertEqual(self.store.get_cached_transcript_summary("vid123"), "condensed notes")

    def test_condense_short_text_passthrough_costs_nothing(self):
        import asyncio
        from services.condense import condense_long_text
        text = "short transcript"
        # Under target: returned untouched without any model call.
        self.assertEqual(asyncio.run(condense_long_text(text, target_chars=9000)), text)

    def test_sora_limit_status_reports_reset(self):
        status = self.store.sora_limit_status("u9", limit=2)
        self.assertTrue(status["allowed"])
        self.assertEqual(status["remaining"], 2)

        self.store.log_sora_usage("u9", video_id="v1")
        self.store.log_sora_usage("u9", video_id="v2")
        status = self.store.sora_limit_status("u9", limit=2)
        self.assertFalse(status["allowed"])
        self.assertEqual(status["remaining"], 0)
        self.assertGreater(status["resets_in_seconds"], 0)
        self.assertLessEqual(status["resets_in_seconds"], 3600)


class UsageCostsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_db = usage_costs.DB_PATH
        self._old_metrics = usage_costs.METRICS_PATH
        usage_costs.DB_PATH = str(Path(self._tmp.name) / "usage.db")
        usage_costs.METRICS_PATH = ""

    def tearDown(self):
        usage_costs.DB_PATH = self._old_db
        usage_costs.METRICS_PATH = self._old_metrics
        self._tmp.cleanup()

    def test_public_metrics_snapshot_contains_only_aggregates(self):
        usage_costs.METRICS_PATH = str(Path(self._tmp.name) / "usage_metrics.json")
        usage_costs.set_request_context(user_id="private-user", intent="private-label")
        usage_costs.record(
            "private-model",
            {
                "prompt_tokens": 120,
                "cached_prompt_tokens": 40,
                "cache_write_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
            99.0,
            meta={"prompt": "must not leak"},
        )
        prior_local = (datetime.now(timezone.utc).astimezone(usage_costs.REPORT_TZ) - timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
        with usage_costs._conn_rw() as conn:
            conn.execute(
                "UPDATE usage_logs SET ts_utc = ?",
                (prior_local.astimezone(timezone.utc).isoformat(),),
            )
        self.assertTrue(usage_costs.publish_metrics_snapshot())

        snapshot = json.loads(Path(usage_costs.METRICS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(snapshot["schema"], 1)
        self.assertEqual(snapshot["windows"]["today"]["calls"], 0)
        self.assertEqual(snapshot["windows"]["allTime"]["totalTokens"], 150)
        self.assertEqual(len(snapshot["daily"]["days"]), 14)
        self.assertEqual(snapshot["daily"]["days"][-1], 150)
        self.assertNotEqual(snapshot["daily"]["to"], datetime.now(usage_costs.REPORT_TZ).date().isoformat())
        serialized = json.dumps(snapshot)
        for private_value in ("private-user", "private-label", "private-model", "must not leak", "99.0"):
            self.assertNotIn(private_value, serialized)

    def test_record_with_request_context_attributes_user(self):
        usage_costs.set_request_context(user_id="42", intent="chat")
        usage_costs.record(
            "gpt-5.5",
            {"prompt_tokens": 1000, "completion_tokens": 500},
            usage_costs.estimate_cost("gpt-5.5", {"prompt_tokens": 1000, "completion_tokens": 500}),
        )
        mine = usage_costs.today_for_user("42")
        self.assertEqual(mine["calls"], 1)
        self.assertEqual(mine["total_tokens"], 1500)
        self.assertGreater(mine["cost"], 0)

        other = usage_costs.today_for_user("99")
        self.assertEqual(other["calls"], 0)

        top = usage_costs.top_users_today()
        self.assertEqual(top[0]["user_id"], "42")

        # Month view ranks the same users (fresh records are within the month).
        top_month = usage_costs.top_users_month()
        self.assertEqual(top_month[0]["user_id"], "42")
        self.assertEqual(top_month[0]["calls"], 1)

        # YTD and all-time include the single fresh record too.
        self.assertEqual(usage_costs.year_to_date()["calls"], 1)
        self.assertEqual(usage_costs.all_time()["calls"], 1)
        self.assertEqual(usage_costs.all_time()["total_tokens"], 1500)

    def test_today_breakdown_groups_by_model_and_label(self):
        usage_costs.set_request_context(user_id="42")
        usage_costs.record("gpt-5.5", {"prompt_tokens": 100, "completion_tokens": 50}, 0.02, label="chat")
        usage_costs.record("gpt-5.5", {"prompt_tokens": 200, "completion_tokens": 80}, 0.03, label="chat")
        usage_costs.record("gpt-5.4-nano", {"prompt_tokens": 50, "completion_tokens": 5}, 0.0001, label="intent_classify")
        usage_costs.record("gpt-image-2.5-sunburst", None, 0.06, label="image_generation")

        rows = usage_costs.today_breakdown("42")
        self.assertEqual(len(rows), 3)
        # Most expensive first
        self.assertEqual(rows[0]["model"], "gpt-image-2.5-sunburst")
        chat_row = next(r for r in rows if r["label"] == "chat")
        self.assertEqual(chat_row["calls"], 2)
        self.assertEqual(chat_row["total_tokens"], 430)
        # Other users excluded
        self.assertEqual(usage_costs.today_breakdown("99"), [])
        # None aggregates everyone
        self.assertEqual(len(usage_costs.today_breakdown(None)), 3)

    def test_estimate_cost_known_and_unknown_models(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0}
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.5", usage), 1.25)
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.4-mini", usage), 0.75)
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.4-nano", usage), 0.20)
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.6-terra", usage), 2.00)
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.6-sol", usage), 5.00)
        self.assertAlmostEqual(usage_costs.estimate_cost("gpt-5.6-luna", usage), 0.20)
        self.assertAlmostEqual(usage_costs.estimate_cost("claude-fable-5", usage), 10.00)
        self.assertEqual(usage_costs.estimate_cost("mystery-model", usage), 0.0)

    def test_gpt_image_output_priced_at_30_per_million(self):
        # GPT Image 2.5 image-output tokens bill at $30/M. A ~6,600-token
        # high-quality image should cost about $0.20.
        usage = {"prompt_tokens": 10, "completion_tokens": 6_600}
        self.assertAlmostEqual(
            usage_costs.estimate_cost("gpt-image-2.5-sunburst", usage),
            0.19805,
            places=4,
        )

    def test_migrates_old_schema_without_user_id(self):
        # Simulate a database created before the user_id column existed.
        import sqlite3
        conn = sqlite3.connect(usage_costs.DB_PATH)
        conn.executescript(
            """
            CREATE TABLE usage_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts_utc TEXT NOT NULL,
              model TEXT NOT NULL,
              label TEXT,
              prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              total_tokens INTEGER NOT NULL DEFAULT 0,
              cost_usd REAL NOT NULL DEFAULT 0.0,
              meta_json TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        usage_costs.set_request_context(user_id="5")
        usage_costs.record("gpt-5.5", {"prompt_tokens": 10, "completion_tokens": 5}, 0.0)
        mine = usage_costs.today_for_user("5")
        self.assertEqual(mine["calls"], 1)

    def test_responses_api_token_keys_normalized(self):
        usage_costs.set_request_context(user_id="7")
        usage_costs.record("gpt-5.5", {"input_tokens": 200, "output_tokens": 100}, 0.0)
        mine = usage_costs.today_for_user("7")
        self.assertEqual(mine["prompt_tokens"], 200)
        self.assertEqual(mine["completion_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
