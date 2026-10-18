import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import chat_handler, video_handler
from bot.draft_verifier import DraftVerdict


class VideoGenerationFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_video_reference_upload_uses_collected_image_inputs(self):
        message = SimpleNamespace(attachments=[])

        upload = await video_handler._resolve_video_reference_upload(
            message,
            image_urls=["data:image/png;base64,Zm9v"],
        )

        self.assertEqual(upload, (b"foo", "input_1.png", "image/png"))

    @patch(
        "bot.chat_handler.verify_chat_draft",
        new_callable=AsyncMock,
        return_value=DraftVerdict(),
    )
    @patch("bot.chat_handler.apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text)
    @patch("bot.chat_handler.generate_openai_messages_response_with_tools", new_callable=AsyncMock)
    @patch("bot.chat_handler.build_chat_context")
    async def test_handle_chat_intent_passes_image_urls_into_tool_context(
        self,
        mock_build_chat_context,
        mock_generate,
        _mock_personality,
        _mock_verifier,
    ):
        mock_build_chat_context.return_value = [{"role": "user", "content": "make a video from this"}]
        mock_generate.return_value = "queued"

        status_msg = SimpleNamespace(edit=AsyncMock())

        async def fake_live_status_with_progress(message, action_label, emoji, coro, duration_estimate, summarizer=None):
            return status_msg, await coro

        send_mock = AsyncMock()
        message = SimpleNamespace(
            guild=SimpleNamespace(id=111),
            channel=SimpleNamespace(id=222),
            author=SimpleNamespace(id=333),
        )

        await chat_handler.handle_chat_intent(
            message=message,
            prompt="make a video from this",
            raw_prompt="make a video from this",
            user_id=333,
            ref_msg=None,
            is_reply_to_bot=False,
            image_urls=["data:image/png;base64,abc"],
            gemini_parts=[],
            duration_estimate=6,
            stream_ok=False,
            live_status_with_progress=fake_live_status_with_progress,
            send_or_edit_with_truncation=send_mock,
            moderation_view_factory=lambda **kwargs: None,
        )

        self.assertEqual(
            mock_generate.await_args.kwargs["tool_context"]["image_urls"],
            ["data:image/png;base64,abc"],
        )

    def test_build_video_config_options_includes_veo_when_available(self):
        options = video_handler.build_video_config_options(include_veo=True)
        options_by_value = {option["value"]: option for option in options}

        self.assertTrue(all(option["provider"] == "veo" for option in options))
        self.assertEqual(len(options), 6)
        self.assertIsNone(video_handler.get_video_config_by_value("sora|sora-2-pro|8", include_veo=True))
        self.assertEqual(video_handler.build_video_config_options(include_veo=False), [])
        self.assertEqual(options_by_value["veo|veo-3.1-generate-preview|6"]["cost"], 2.40)
        self.assertEqual(options_by_value["veo|veo-3.1-fast-generate-preview|8"]["cost"], 0.80)

    def test_video_cost_message_matches_dropdown_prices(self):
        message = video_handler._build_video_cost_message("a cinematic sunrise", include_veo=True)

        self.assertNotIn("Sora", message)
        self.assertIn("Veo 3.1**: 4s $1.60 | 6s $2.40 | 8s $3.20", message)
        self.assertIn("Veo 3.1 Fast**: 4s $0.40 | 6s $0.60 | 8s $0.80", message)
        self.assertIn("native audio is included", message)

    async def test_unavailable_video_returns_without_picker_or_provider_work(self):
        message = SimpleNamespace(attachments=[], reply=AsyncMock())
        with patch.object(video_handler, "veo_is_available", return_value=False), patch.object(
            video_handler, "VideoConfirmationView"
        ) as view, patch.object(video_handler, "_resolve_video_reference_upload", new_callable=AsyncMock) as upload, patch.object(
            video_handler, "invoke_provider", new_callable=AsyncMock
        ) as provider:
            await video_handler.handle_generate_video_intent(message, "a cat", 123, AsyncMock(), False)
        view.assert_not_called()
        upload.assert_not_awaited()
        provider.assert_not_awaited()
        self.assertIn("currently unavailable", message.reply.await_args.args[0])
        self.assertNotIn("view", message.reply.await_args.kwargs)

    async def test_server_with_no_allowed_video_returns_without_picker(self):
        message = SimpleNamespace(attachments=[], reply=AsyncMock())
        with patch.object(video_handler, "veo_is_available", return_value=True), patch(
            "services.server_policy.video_config_allowed", return_value=False
        ), patch.object(video_handler, "VideoConfirmationView") as view:
            await video_handler.handle_generate_video_intent(message, "a cat", 123, AsyncMock(), False)
        view.assert_not_called()
        self.assertIn("maximum video tier", message.reply.await_args.args[0])

    async def test_remix_returns_without_picker_or_provider_work(self):
        message = SimpleNamespace(attachments=[], reply=AsyncMock())
        with patch.object(video_handler, "VideoConfirmationView") as view, patch.object(
            video_handler, "invoke_provider", new_callable=AsyncMock
        ) as provider:
            await video_handler.handle_generate_video_intent(message, "remix my video", 123, AsyncMock(), False)
        view.assert_not_called()
        provider.assert_not_awaited()
        self.assertIn("no longer available", message.reply.await_args.args[0])

    async def test_retired_selection_and_changed_server_ceiling_cannot_submit(self):
        for selection in ("sora|sora-2|4", "veo|veo-3.1-generate-preview|8"):
            with self.subTest(selection=selection):
                class FakeView:
                    def __init__(self, **kwargs):
                        self.value = selection

                    async def wait(self):
                        return

                confirm = SimpleNamespace(edit=AsyncMock())
                message = SimpleNamespace(attachments=[], reply=AsyncMock(return_value=confirm))
                # Allow all six choices before the picker, reject after confirmation.
                with patch.object(video_handler, "veo_is_available", return_value=True), patch.object(
                    video_handler, "VideoConfirmationView", FakeView
                ), patch("services.server_policy.video_config_allowed", side_effect=[True] * 6 + [False]), patch.object(
                    video_handler, "invoke_provider", new_callable=AsyncMock
                ) as provider, patch.object(video_handler, "check_veo_limit") as limit:
                    await video_handler.handle_generate_video_intent(message, "a cat", 123, AsyncMock(), False)
                provider.assert_not_awaited()
                limit.assert_not_called()
                content = confirm.edit.await_args.kwargs["content"]
                self.assertIn("no longer available" if selection.startswith("sora") else "maximum video tier", content)

    @patch("bot.video_handler.veo_is_available", return_value=True)
    @patch("bot.video_handler.check_veo_limit", return_value=True)
    async def test_handle_generate_video_intent_uses_original_message_for_status_handoff(self, _mock_limit, _mock_available):
        class FakeView:
            def __init__(self, author_id, video_options):
                self.author_id = author_id
                self.video_options = video_options
                self.value = "veo|veo-3.1-fast-generate-preview|4"

            async def wait(self):
                return

        confirm_msg = SimpleNamespace(edit=AsyncMock())
        message = SimpleNamespace(
            attachments=[],
            reply=AsyncMock(return_value=confirm_msg),
        )
        captured = {}

        async def fake_live_status_with_progress(base_message, **kwargs):
            captured["base_message"] = base_message
            captured["existing_status_msg"] = kwargs.get("existing_status_msg")
            kwargs["coro"].close()
            status_msg = SimpleNamespace(edit=AsyncMock(), reply=AsyncMock())
            return status_msg, (None, "stopped")

        with patch("bot.video_handler.VideoConfirmationView", FakeView):
            await video_handler.handle_generate_video_intent(
                message=message,
                prompt="generate a video of this",
                user_id=123,
                live_status_with_progress=fake_live_status_with_progress,
                stream_ok=False,
                image_urls=None,
            )

        self.assertIs(captured["base_message"], message)
        self.assertIs(captured["existing_status_msg"], confirm_msg)

    @patch("bot.video_handler.generate_veo_video", new_callable=AsyncMock)
    @patch("bot.video_handler.log_veo_usage")
    @patch("bot.video_handler.check_veo_limit", return_value=True)
    async def test_handle_generate_video_intent_routes_selected_veo_option(
        self,
        _mock_limit,
        mock_log_usage,
        mock_generate_veo,
    ):
        class FakeView:
            def __init__(self, author_id, video_options):
                self.author_id = author_id
                self.video_options = video_options
                self.value = "veo|veo-3.1-fast-generate-preview|6"

            async def wait(self):
                return

        confirm_msg = SimpleNamespace(edit=AsyncMock())
        status_msg = SimpleNamespace(edit=AsyncMock(), reply=AsyncMock())
        message = SimpleNamespace(
            attachments=[],
            reply=AsyncMock(return_value=confirm_msg),
        )
        mock_generate_veo.return_value = (b"video-bytes", None)

        async def fake_live_status_with_progress(base_message, **kwargs):
            return status_msg, await kwargs["coro"]

        with patch("bot.video_handler.veo_is_available", return_value=True), patch("bot.video_handler._record_video_cost") as record:
            with patch("bot.video_handler.VideoConfirmationView", FakeView):
                await video_handler.handle_generate_video_intent(
                    message=message,
                    prompt="generate a cinematic canyon flythrough",
                    user_id=123,
                    live_status_with_progress=fake_live_status_with_progress,
                    stream_ok=False,
                    image_urls=["data:image/png;base64,Zm9v"],
                )

        self.assertEqual(mock_generate_veo.await_args.kwargs["model"], "veo-3.1-fast-generate-preview")
        self.assertEqual(mock_generate_veo.await_args.kwargs["seconds"], 6)
        self.assertFalse(mock_generate_veo.await_args.kwargs["generate_audio"])
        self.assertEqual(mock_generate_veo.await_args.kwargs["image_data"], b"foo")
        self.assertEqual(mock_generate_veo.await_args.kwargs["image_content_type"], "image/png")
        record.assert_called_once_with("veo-3.1-fast-generate-preview", 6)
        mock_log_usage.assert_called_once()
        status_msg.reply.assert_awaited()



if __name__ == "__main__":
    unittest.main()
