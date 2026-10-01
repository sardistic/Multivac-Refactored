import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from providers import claude_utils


class ClaudeFableConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_route_uses_fable(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            stop_reason="end_turn",
            usage=None,
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))

        with patch.object(claude_utils, "ANTHROPIC_API_KEY", "test-key"), patch.object(
            claude_utils.anthropic, "AsyncAnthropic", return_value=client
        ):
            result = await claude_utils.generate_claude_response(
                [{"role": "user", "content": "hello"}]
            )

        self.assertEqual(result, "hello")
        self.assertEqual(create.await_args.kwargs["model"], "claude-fable-5-1")
        self.assertNotIn("temperature", create.await_args.kwargs)

    async def test_legacy_claude_model_keeps_temperature(self):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hello")],
            stop_reason="end_turn",
            usage=None,
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))

        with patch.object(claude_utils, "ANTHROPIC_API_KEY", "test-key"), patch.object(
            claude_utils.anthropic, "AsyncAnthropic", return_value=client
        ):
            await claude_utils.generate_claude_response(
                [{"role": "user", "content": "hello"}],
                model="claude-sonnet-4-20250514",
                temperature=0.2,
            )

        self.assertEqual(create.await_args.kwargs["temperature"], 0.2)

    async def test_fable_refusal_is_reported_without_index_error(self):
        response = SimpleNamespace(
            content=[],
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber"),
            usage=None,
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))

        with patch.object(claude_utils, "ANTHROPIC_API_KEY", "test-key"), patch.object(
            claude_utils.anthropic, "AsyncAnthropic", return_value=client
        ):
            result = await claude_utils.generate_claude_response(
                [{"role": "user", "content": "hello"}]
            )

        self.assertEqual(result, "❌ Claude Fable declined this request (cyber).")


class ClaudeVisionContentTests(unittest.IsolatedAsyncioTestCase):
    async def _call(self, messages):
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=None,
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch.object(claude_utils, "ANTHROPIC_API_KEY", "test-key"), patch.object(
            claude_utils.anthropic, "AsyncAnthropic", return_value=client
        ):
            await claude_utils.generate_claude_response(messages)
        return create.await_args.kwargs["messages"]

    async def test_image_blocks_survive_sanitization(self):
        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
        }
        sent = await self._call(
            [
                {"role": "user", "content": "earlier text turn"},
                {"role": "assistant", "content": "earlier reply"},
                {"role": "user", "content": [image_block, {"type": "text", "text": "fix this"}]},
            ]
        )
        self.assertEqual(sent[-1]["role"], "user")
        self.assertIn(image_block, sent[-1]["content"])
        self.assertEqual(sent[0]["content"], [{"type": "text", "text": "earlier text turn"}])

    async def test_consecutive_same_role_messages_merge_blocks(self):
        sent = await self._call(
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ]
        )
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            sent[0]["content"],
            [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        )

    def test_image_input_to_block_shapes(self):
        b64 = claude_utils.image_input_to_block("data:image/jpeg;base64,aGVsbG8=")
        self.assertEqual(b64["source"]["type"], "base64")
        self.assertEqual(b64["source"]["media_type"], "image/jpeg")
        self.assertEqual(b64["source"]["data"], "aGVsbG8=")

        url = claude_utils.image_input_to_block("https://cdn.discordapp.com/attachments/x/y.png")
        self.assertEqual(url["source"]["type"], "url")

        self.assertIsNone(claude_utils.image_input_to_block("not an image input"))
        self.assertIsNone(claude_utils.image_input_to_block(""))


if __name__ == "__main__":
    unittest.main()
