import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import provider_intents
from bot.draft_verifier import DraftVerdict
from bot.message_inputs import collect_image_inputs, resolve_reference_message
from providers import claude_utils


class ClaudeImageReplyTests(unittest.IsolatedAsyncioTestCase):
    IMAGE = "data:image/png;base64,aW1hZ2U="
    DESCRIPTION = "The image is a K-OFF ad with a woman resting on a bed."
    MISSING_IMAGE = "Send the image you mean, and I'll explain it."

    def setUp(self):
        self.message = SimpleNamespace(
            guild=None,
            channel=SimpleNamespace(id=456, fetch_message=AsyncMock()),
            author=SimpleNamespace(id=123, display_name="Test user"),
            attachments=[], embeds=[], content="claude wtf is this",
            reference=None, message_snapshots=[],
        )
        self.send = AsyncMock()
        self.create = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.DESCRIPTION)],
            stop_reason="end_turn", usage=None,
        ))
        self.review = AsyncMock(return_value=DraftVerdict(
            action="revise", revised_answer=self.MISSING_IMAGE,
        ))

    @staticmethod
    async def _live_status(message, *, coro, **kwargs):
        return SimpleNamespace(edit=AsyncMock()), await coro

    async def _run(self, images, ref_msg=None, prompt="claude wtf is this"):
        # Keep the real Anthropic message conversion; replace only external
        # calls and stored context. The hostile rewrite reproduces the bug.
        async def invoke(name, fallback, *args, **kwargs):
            self.assertEqual(name, "chat.claude")
            self.assertTrue(kwargs["enable_tools"])
            return await fallback(*args, **kwargs)

        with ExitStack() as stack:
            for name, value in (
                ("build_channel_message_window", []),
                ("build_message_user_style_system_messages", []),
            ):
                stack.enter_context(patch.object(provider_intents, name, return_value=value))
            stack.enter_context(patch.object(provider_intents, "invoke_provider", side_effect=invoke))
            stack.enter_context(patch.object(provider_intents, "verify_chat_draft", self.review))
            stack.enter_context(patch.object(
                provider_intents, "apply_personality_overrides",
                side_effect=lambda user_id, *, intent, text: text,
            ))
            stack.enter_context(patch.object(claude_utils, "ANTHROPIC_API_KEY", "test-key"))
            stack.enter_context(patch.object(
                claude_utils.anthropic, "AsyncAnthropic",
                return_value=SimpleNamespace(messages=SimpleNamespace(create=self.create)),
            ))
            stack.enter_context(patch.object(
                claude_utils, "get_tool_snapshot",
                return_value=SimpleNamespace(tool_specs=lambda: []),
            ))
            stack.enter_context(patch.object(claude_utils, "_record_claude_usage"))
            await provider_intents.handle_claude_chat_intent(
                message=self.message, prompt=prompt, stream_ok=False,
                live_status_with_progress=self._live_status,
                send_or_edit_with_truncation=self.send,
                image_urls=images, ref_msg=ref_msg,
            )

    async def test_reply_to_generated_image_keeps_claude_visual_answer(self):
        attachment = SimpleNamespace(
            url="https://cdn.discordapp.com/attachments/1/2/generated_image.png",
            filename="generated_image.png", content_type="image/png", size=100,
        )
        generated = SimpleNamespace(
            author=SimpleNamespace(id=999, display_name="MultiVAC"),
            content="Image generated (GPT Image 2.5 Sunburst)",
            attachments=[attachment], embeds=[], message_snapshots=[],
        )
        self.message.reference = SimpleNamespace(resolved=generated, message_id=789)
        ref_msg, is_bot_reply = await resolve_reference_message(
            self.message, SimpleNamespace(id=999),
        )
        self.assertTrue(is_bot_reply)
        convert = AsyncMock(return_value=self.IMAGE)
        images = await collect_image_inputs(self.message, ref_msg, convert)

        await self._run(images, ref_msg)

        blocks = self.create.await_args.kwargs["messages"][-1]["content"]
        self.assertEqual(blocks[0], {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="},
        })
        self.assertIn(generated.content, blocks[1]["text"])
        self.assertIn("wtf is this", blocks[1]["text"])
        self.assertEqual(self.send.await_args.args[0], self.DESCRIPTION)
        self.review.assert_not_awaited()
        self.create.assert_awaited_once()

    async def test_url_image_keeps_claude_answer_without_blind_review(self):
        image_url = "https://example.test/image.png"
        await self._run([image_url])

        blocks = self.create.await_args.kwargs["messages"][-1]["content"]
        self.assertEqual(blocks[0]["source"], {"type": "url", "url": image_url})
        self.assertEqual(self.send.await_args.args[0], self.DESCRIPTION)
        self.review.assert_not_awaited()

    async def test_text_only_requests_still_use_draft_review(self):
        await self._run([], prompt="claude explain this phrase")

        self.review.assert_awaited_once()
        self.assertEqual(self.send.await_args.args[0], self.MISSING_IMAGE)
        self.assertEqual(self.review.await_args.kwargs["prompt"], "explain this phrase")


if __name__ == "__main__":
    unittest.main()
