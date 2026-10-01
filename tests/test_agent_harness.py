import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from providers import claude_utils, openai_messages
from services.agent_execution import consume_tool_call_budget, execute_agent_tool
from services.agent_runs import AgentRunStore
from services.tool_handlers import handle_get_agent_run_status
from services import usage_costs


def _spec(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class AgentRunStoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_trace_keeps_evidence_but_not_prompt_profile_or_image_bytes(self):
        store = AgentRunStore(self.path)
        run_id = store.start(
            provider="openai",
            model="gpt-5.6-sol",
            context={
                "guild_id": "g",
                "channel_id": "c",
                "user_id": "u",
                "intent": "chat_reverse_image",
                "request_text": "private reverse-search wording",
                "image_urls": ["data:image/png;base64,SECRETBYTES"],
                "profile": "private profile",
            },
            max_steps=8,
        )
        store.add_step(
            run_id,
            step_index=1,
            phase="tool",
            tool_name="reverse_image_search",
            status="completed",
            args={"query": "private manga title", "image_index": 0},
            result={
                "ok": True,
                "provider_chain": [
                    {"provider": "google_cloud_vision_web_detection", "status": "completed"},
                    {"provider": "serpapi_google_lens", "status": "completed"},
                ],
                "pages_with_matches": [{"url": "https://source.test/page"}],
                "visually_similar_images": [{"url": "https://similar.test/panel"}],
            },
        )
        store.finish(run_id, "completed")

        raw = self.path.read_bytes()
        self.assertNotIn(b"private reverse-search wording", raw)
        self.assertNotIn(b"SECRETBYTES", raw)
        self.assertNotIn(b"private profile", raw)
        self.assertNotIn(b"private manga title", raw)
        step = store.steps(run_id)[0]
        self.assertIn("https://source.test/page", step["result_json"])
        self.assertIn("https://similar.test/panel", step["result_json"])
        self.assertIn("serpapi_google_lens", step["result_json"])
        metadata = json.loads(store.recent(limit=1)[0]["metadata_json"])
        self.assertIn("request_sha256", metadata)

    async def test_mutating_tool_requires_current_explicit_request(self):
        snapshot = SimpleNamespace()
        executor = AsyncMock(return_value={"ok": True})
        blocked = await execute_agent_tool(
            "remember_fact",
            {"fact": "x"},
            context={"request_text": "hello"},
            snapshot=snapshot,
            recorder=None,
            trace=[],
            step_index=1,
            executor=executor,
        )
        self.assertEqual(blocked["error"], "approval_required")
        executor.assert_not_awaited()

        allowed = await execute_agent_tool(
            "remember_fact",
            {"fact": "x"},
            context={"request_text": "remember this for me"},
            snapshot=snapshot,
            recorder=None,
            trace=[],
            step_index=1,
            executor=executor,
        )
        self.assertTrue(allowed["ok"])
        executor.assert_awaited_once()

    async def test_read_only_transient_failure_retries_once(self):
        recorder = MagicMock()
        executor = AsyncMock(
            side_effect=[
                {"ok": False, "error": "temporary 503 unavailable"},
                {"ok": True, "results": []},
            ]
        )
        trace = []
        result = await execute_agent_tool(
            "web_search",
            {"q": "current result"},
            context={},
            snapshot=SimpleNamespace(),
            recorder=recorder,
            trace=trace,
            step_index=1,
            executor=executor,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(executor.await_count, 2)
        recorder.retry.assert_called_once()
        self.assertEqual([item["status"] for item in trace], ["failed", "completed"])

    def test_per_tool_call_budget_blocks_only_after_allowed_calls(self):
        counts = {}
        limits = {"web_search": 2}
        self.assertIsNone(consume_tool_call_budget("web_search", limits, counts))
        self.assertIsNone(consume_tool_call_budget("web_search", limits, counts))
        blocked = consume_tool_call_budget("web_search", limits, counts)
        self.assertEqual(blocked["error"], "tool_call_limit_reached")
        self.assertEqual(blocked["limit"], 2)
        self.assertIsNone(consume_tool_call_budget("summarize_url", limits, counts))

    async def test_status_tool_is_scoped_to_current_user_and_conversation(self):
        with patch("services.agent_runs._state_db_path", return_value=self.path):
            store = AgentRunStore()
            own = store.start(
                provider="openai",
                model="gpt-5.6-sol",
                context={"guild_id": "g", "channel_id": "c", "user_id": "u"},
                max_steps=4,
            )
            store.finish(own, "completed")
            other = store.start(
                provider="openai",
                model="gpt-5.6-sol",
                context={"guild_id": "g", "channel_id": "c", "user_id": "other"},
                max_steps=4,
            )
            store.finish(other, "completed")
            result = await handle_get_agent_run_status(
                {"_context": {"guild_id": "g", "channel_id": "c", "user_id": "u"}}
            )
        self.assertEqual([run["run_id"] for run in result["runs"]], [own])


class ProviderHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_responses_loop_enforces_per_tool_call_budget(self):
        calls = [
            SimpleNamespace(
                type="function_call",
                call_id=f"call-{index}",
                name="web_search",
                arguments='{"q":"candidate"}',
            )
            for index in range(3)
        ]
        first = SimpleNamespace(output=calls)
        final = SimpleNamespace(output=[], output_text="done")
        trace = []
        recorder = MagicMock()
        with patch.object(
            openai_messages,
            "_responses_create",
            new=AsyncMock(return_value=final),
        ), patch.object(
            openai_messages,
            "_exec_tool",
            new=AsyncMock(return_value='{"ok":true}'),
        ) as execute:
            _response, current_input = await openai_messages._responses_tool_loop(
                first,
                [{"role": "user", "content": "find it"}],
                active_tools=[],
                tool_snapshot=SimpleNamespace(),
                recorder=recorder,
                tool_trace=trace,
                tool_call_limits={"web_search": 2},
            )

        self.assertEqual(execute.await_count, 2)
        self.assertEqual(trace[-1], {"name": "web_search", "status": "call_limit_reached"})
        self.assertIn("tool_call_limit_reached", current_input[-1]["output"])
        recorder.step.assert_called_once()

    async def test_fable_uses_shared_tool_loop_and_returns_final_text(self):
        use = SimpleNamespace(type="tool_use", id="tool-1", name="web_search", input={"q": "old"})
        first = SimpleNamespace(
            content=[use],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=3),
        )
        second = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Grounded answer")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )
        create = AsyncMock(side_effect=[first, second])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        snapshot = SimpleNamespace(generation=7, tool_specs=lambda: [_spec("web_search")])
        recorder = MagicMock()
        with patch.object(claude_utils, "ANTHROPIC_API_KEY", "test"), patch.object(
            claude_utils.anthropic, "AsyncAnthropic", return_value=client
        ), patch.object(claude_utils, "get_tool_snapshot", return_value=snapshot), patch.object(
            claude_utils, "execute_agent_tool", new_callable=AsyncMock, return_value=[{"url": "https://x.test"}]
        ) as execute, patch.object(
            claude_utils, "AgentRunRecorder", return_value=recorder
        ), patch.object(claude_utils, "_record_claude_usage"):
            result = await claude_utils.generate_claude_response(
                [{"role": "user", "content": "search it"}],
                enable_tools=True,
                forced_tool="web_search",
                forced_tool_args={"q": "forced current query"},
                tool_context={"request_text": "search it"},
            )

        self.assertEqual(result, "Grounded answer")
        self.assertEqual(create.await_count, 2)
        self.assertEqual(create.await_args_list[0].kwargs["tools"][0]["name"], "web_search")
        self.assertNotIn("tool_choice", create.await_args_list[0].kwargs)
        self.assertIn("call the web_search tool", create.await_args_list[0].kwargs["system"])
        self.assertNotIn("In your first response, call", create.await_args_list[1].kwargs["system"])
        self.assertEqual(execute.await_args.kwargs["context"]["request_text"], "search it")
        self.assertEqual(execute.await_args.args[1]["q"], "forced current query")
        recorder.finish.assert_called_with("completed")

    @patch("providers.openai_messages.USE_RESPONSES", True)
    async def test_responses_payload_receives_task_shaped_reasoning(self):
        create = AsyncMock(return_value=SimpleNamespace(output_text="answer", output=[]))
        snapshot = SimpleNamespace(generation=1, tool_specs=lambda: [])
        with patch.object(openai_messages, "_responses_create", create), patch.object(
            openai_messages, "get_tool_snapshot", return_value=snapshot
        ):
            result = await openai_messages.generate_openai_messages_response_with_tools(
                [{"role": "user", "content": "analyze"}],
                tools=[],
                model="gpt-5.6-sol",
                reasoning_effort="high",
            )
        self.assertEqual(result, "answer")
        self.assertEqual(create.await_args.kwargs["reasoning"], {"effort": "high"})
        self.assertNotIn("tools", create.await_args.kwargs)

    @patch("providers.openai_messages.USE_RESPONSES", True)
    async def test_empty_reverse_synthesis_falls_back_to_provider_evidence(self):
        specs = [_spec("reverse_image_search")]
        snapshot = SimpleNamespace(generation=1, tool_specs=lambda: specs)
        reverse_call = SimpleNamespace(
            type="function_call",
            call_id="reverse-1",
            name="reverse_image_search",
            arguments="{}",
        )
        create = AsyncMock(
            side_effect=[
                SimpleNamespace(output_text="", output=[reverse_call]),
                SimpleNamespace(output_text="", output=[]),
                SimpleNamespace(output_text="", output=[]),
            ]
        )
        lookup = AsyncMock(
            return_value={
                "ok": True,
                "provider": "google_cloud_vision_web_detection",
                "lookup_type": "reverse_image_search",
                "match_found": True,
                "pages_with_matches": [
                    {
                        "title": "Futoku no Guild Vol. 17 Ch. 99",
                        "url": "https://forums.mangadex.org/threads/example.1/",
                    }
                ],
            }
        )
        recorder = MagicMock()

        with patch.object(openai_messages, "_responses_create", create), patch.object(
            openai_messages, "get_tool_snapshot", return_value=snapshot
        ), patch.object(openai_messages, "execute_tool", lookup), patch.object(
            openai_messages, "AgentRunRecorder", return_value=recorder
        ):
            result = await openai_messages.generate_openai_messages_response_with_tools(
                [{"role": "user", "content": "what manga is this"}],
                forced_tool="reverse_image_search",
                forced_tool_args={"image_index": 0},
                tool_context={
                    "request_text": "what manga is this",
                    "image_urls": ["data:image/png;base64,abc"],
                },
                reasoning_effort="high",
            )

        self.assertIn("Futoku no Guild Vol. 17 Ch. 99", result)
        self.assertIn("https://forums.mangadex.org/threads/example.1/", result)
        self.assertEqual(create.await_args_list[-1].kwargs["max_output_tokens"], 1200)
        self.assertEqual(create.await_args_list[-1].kwargs["reasoning"], {"effort": "low"})
        recorder.finish.assert_called_with("completed_with_evidence_fallback")

    def test_cache_aware_current_pricing(self):
        usage = {
            "input_tokens": 1000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 100},
        }
        self.assertAlmostEqual(
            usage_costs.estimate_cost("gpt-5.6-sol", usage),
            0.004525,
        )


if __name__ == "__main__":
    unittest.main()
