import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot import chat_handler


class _StatusMessage:
    def __init__(self):
        self.edits = []

    async def edit(self, *, content):
        self.edits.append(content)


class ChatFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_to_gemini_fallback_is_bounded_and_reuses_status(self):
        status = _StatusMessage()
        live_calls = []

        async def live_status(message, *, coro, existing_status_msg=None, **kwargs):
            live_calls.append(existing_status_msg)
            result = await coro
            return existing_status_msg or status, result

        message = SimpleNamespace(
            guild=None,
            channel=SimpleNamespace(id=456),
            author=SimpleNamespace(id=123),
        )
        openai = AsyncMock(return_value="⚠️ OpenAI tools error: Error code: 429 - quota")
        gemini = Mock(side_effect=[(None, []), ("must not retry", [])])

        with patch.object(chat_handler, "build_chat_context", return_value=[{"role": "user", "content": "hello"}]), patch.object(
            chat_handler, "generate_openai_messages_response_with_tools", openai
        ), patch.object(chat_handler, "generate_gemini_text", gemini), patch.object(
            chat_handler, "confirm_provider_fallback", new=AsyncMock(return_value=True)
        ) as confirm:
            await chat_handler.handle_chat_intent(
                message=message,
                prompt="hello",
                raw_prompt="hello",
                user_id=123,
                ref_msg=None,
                is_reply_to_bot=False,
                image_urls=[],
                gemini_parts=[],
                duration_estimate=1,
                stream_ok=False,
                live_status_with_progress=live_status,
                send_or_edit_with_truncation=AsyncMock(),
                moderation_view_factory=Mock(),
                default_model=chat_handler.OPENAI_CHAT_MODEL,
            )

        self.assertEqual(openai.await_count, 1)
        confirm.assert_awaited_once()
        self.assertEqual(gemini.call_count, 1)
        self.assertEqual(live_calls, [None, status])
        self.assertIn("fallback model returned no response", status.edits[-1])


if __name__ == "__main__":
    unittest.main()
