import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import image_handler


class ImageDescriptionFlowTests(unittest.IsolatedAsyncioTestCase):
    def test_build_image_explanation_messages_keeps_images_attached(self):
        msgs = image_handler._build_image_explanation_messages(
            prompt="explain this image",
            extracted_notes="Visible text: quote",
            image_urls=["data:image/png;base64,abc"],
            reply_context="",
        )

        user_content = msgs[-1]["content"]
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(
            user_content[1],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc", "detail": "high"},
            },
        )

    def test_needs_explanation_retry_for_incomplete_outline(self):
        text = (
            "1. What the image is\n"
            "It's a screenshot of a post.\n\n"
            "2. The key text or quote\n"
            "The post says something about Dune."
        )

        self.assertTrue(
            image_handler._needs_explanation_retry(
                "explain this image and the quote in it what should i understand about it",
                text,
            )
        )

    def test_needs_explanation_repair_for_trailing_fragment(self):
        text = (
            "1. What the image is\n"
            "It's a screenshot of a post.\n\n"
            "2. The key text or quote\n"
            "The visible quote from the book page says"
        )

        self.assertTrue(
            image_handler._needs_explanation_repair(
                "explain this image and the quote in it what should i understand about it",
                text,
            )
        )

    @patch("bot.image_handler.apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text)
    @patch("bot.image_handler.generate_openai_messages_response", new_callable=AsyncMock)
    async def test_handle_describe_image_intent_retries_until_takeaway_exists(self, mock_generate, _mock_personality):
        mock_generate.side_effect = [
            "1. Image type and setting\nA screenshot.\n\n2. Visible text\nA Dune quote.\n\n3. Uncertain or partially legible text\nNone.\n\n4. Visual cues that matter\nNested quote card.",
            "1. What the image is\nIt's a screenshot.\n\n2. The key text or quote\nA Dune quote.",
            "1. What the image is\nIt's a screenshot of a post quoting Dune.\n\n2. The key text or quote\nThe visible quote from the book page says",
            (
                "1. What the image is\nIt's a screenshot of a post quoting Dune.\n\n"
                "2. The key text or quote\nThe quote warns against handing human thinking over to machines.\n\n"
                "3. What it means / what you should understand\nThe point is that the post is connecting Dune's anti-machine warning to modern AI dependence."
            ),
        ]

        status_msg = SimpleNamespace()

        async def fake_live_status_with_progress(message, action_label, emoji, coro, duration_estimate, summarizer=None):
            return status_msg, await coro

        send_mock = AsyncMock()
        message = SimpleNamespace(author=SimpleNamespace(id=123))

        await image_handler.handle_describe_image_intent(
            message=message,
            prompt="explain this image and the quote in it what should i understand about it",
            image_urls=["data:image/png;base64,abc"],
            ref_msg=None,
            is_reply_to_bot=False,
            duration_estimate=8,
            stream_ok=False,
            live_status_with_progress=fake_live_status_with_progress,
            send_or_edit_with_truncation=send_mock,
        )

        self.assertEqual(mock_generate.await_count, 4)
        self.assertEqual(mock_generate.await_args_list[0].kwargs["temperature"], 0.0)
        self.assertEqual(mock_generate.await_args_list[1].kwargs["temperature"], 0.2)
        self.assertEqual(mock_generate.await_args_list[2].kwargs["temperature"], 0.2)
        self.assertEqual(mock_generate.await_args_list[3].kwargs["temperature"], 0.1)
        self.assertEqual(mock_generate.await_args_list[0].kwargs["max_tokens"], 1400)
        self.assertEqual(mock_generate.await_args_list[1].kwargs["max_tokens"], 1400)
        self.assertEqual(mock_generate.await_args_list[2].kwargs["max_tokens"], 1400)
        self.assertEqual(mock_generate.await_args_list[3].kwargs["max_tokens"], 1800)
        explanation_user_content = mock_generate.await_args_list[1].args[0][-1]["content"]
        self.assertEqual(explanation_user_content[1]["type"], "image_url")
        sent_text = send_mock.await_args.args[0]
        self.assertIn("3. What it means / what you should understand", sent_text)

    @patch("bot.image_handler.apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text)
    @patch("bot.image_handler.generate_openai_messages_response", new_callable=AsyncMock)
    async def test_handle_describe_image_intent_falls_back_after_incomplete_repair(self, mock_generate, _mock_personality):
        incomplete = (
            "1. What the image is\n"
            "It's a screenshot of a post quoting Dune.\n\n"
            "2. The key text or quote\n"
            "The photographed Dune passage reads"
        )
        mock_generate.side_effect = [
            (
                "1. Image type and setting\nA screenshot of an X post.\n\n"
                "2. Visible text\nA Dune quote about turning thinking over to machines.\n\n"
                "3. Uncertain or partially legible text\nA smaller quoted line is hard to read exactly.\n\n"
                "4. Visual cues that matter\nThe screenshot highlights a book passage about machines and control."
            ),
            incomplete,
            incomplete,
            incomplete,
        ]

        status_msg = SimpleNamespace()

        async def fake_live_status_with_progress(message, action_label, emoji, coro, duration_estimate, summarizer=None):
            return status_msg, await coro

        send_mock = AsyncMock()
        message = SimpleNamespace(author=SimpleNamespace(id=123))

        await image_handler.handle_describe_image_intent(
            message=message,
            prompt="explain this image and the quote in it what should i understand about it",
            image_urls=["data:image/png;base64,abc"],
            ref_msg=None,
            is_reply_to_bot=False,
            duration_estimate=8,
            stream_ok=False,
            live_status_with_progress=fake_live_status_with_progress,
            send_or_edit_with_truncation=send_mock,
        )

        sent_text = send_mock.await_args.args[0]
        self.assertIn("3. What it means / what you should understand", sent_text)
        self.assertIn("warning about power and control", sent_text)
        self.assertNotEqual(sent_text.rstrip().splitlines()[-1].strip(), "The photographed Dune passage reads")


    @patch("bot.image_handler.apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text)
    @patch("bot.image_handler.generate_gemini_text")
    @patch("bot.image_handler.generate_openai_messages_response", new_callable=AsyncMock)
    async def test_handle_describe_image_intent_falls_back_to_gemini(
        self, mock_generate, mock_gemini, _mock_personality
    ):
        """OpenAI vision being down must not cost the user the answer: the
        picture is already resolved, and Gemini can read it."""
        self.enterContext(patch.object(image_handler, "confirm_provider_fallback", AsyncMock(return_value=True)))
        mock_generate.return_value = (
            "\u26a0\ufe0f OpenAI error: the provider is rate-limited or out of quota (HTTP 429)"
        )
        mock_gemini.return_value = ("A screenshot of a post quoting Dune.", [])

        status_msg = SimpleNamespace()

        async def fake_live_status_with_progress(message, action_label, emoji, coro, duration_estimate, summarizer=None):
            return status_msg, await coro

        send_mock = AsyncMock()
        message = SimpleNamespace(author=SimpleNamespace(id=123))
        data_url = "data:image/png;base64," + base64.b64encode(b"pixel-bytes").decode("ascii")

        await image_handler.handle_describe_image_intent(
            message=message,
            prompt="what does this say",
            image_urls=[data_url],
            ref_msg=None,
            is_reply_to_bot=False,
            duration_estimate=8,
            stream_ok=False,
            live_status_with_progress=fake_live_status_with_progress,
            send_or_edit_with_truncation=send_mock,
        )

        self.assertEqual(len(mock_gemini.call_args.kwargs["extra_parts"]), 1)
        self.assertEqual(mock_gemini.call_args.kwargs["prompt"], "what does this say")
        sent_text = send_mock.await_args.args[0]
        self.assertIn("quoting Dune", sent_text)
        self.assertNotIn("OpenAI error", sent_text)

    @patch("bot.image_handler.apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text)
    @patch("bot.image_handler.generate_gemini_text")
    @patch("bot.image_handler.generate_openai_messages_response", new_callable=AsyncMock)
    async def test_handle_describe_image_intent_keeps_openai_error_when_gemini_has_nothing(
        self, mock_generate, mock_gemini, _mock_personality
    ):
        self.enterContext(patch.object(image_handler, "confirm_provider_fallback", AsyncMock(return_value=True)))
        mock_generate.return_value = "\u26a0\ufe0f OpenAI error: the provider is rate-limited or out of quota (HTTP 429)"
        mock_gemini.return_value = (None, [])

        status_msg = SimpleNamespace()

        async def fake_live_status_with_progress(message, action_label, emoji, coro, duration_estimate, summarizer=None):
            return status_msg, await coro

        send_mock = AsyncMock()
        message = SimpleNamespace(author=SimpleNamespace(id=123))
        data_url = "data:image/png;base64," + base64.b64encode(b"pixel-bytes").decode("ascii")

        await image_handler.handle_describe_image_intent(
            message=message,
            prompt="what does this say",
            image_urls=[data_url],
            ref_msg=None,
            is_reply_to_bot=False,
            duration_estimate=8,
            stream_ok=False,
            live_status_with_progress=fake_live_status_with_progress,
            send_or_edit_with_truncation=send_mock,
        )

        self.assertIn("OpenAI error", send_mock.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
