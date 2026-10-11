import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.chat_handler import GEMINI_FALLBACK_CHAT_MODEL
from bot.intent_dispatcher import DispatchContext, _dispatch_builtin_intent
from bot.message_inputs import (
    MAX_ATTACHMENT_BYTES,
    collect_gemini_parts,
    has_visual_inputs,
)
from providers.gemini_text import _has_video_part


def _attachment(content_type, filename, size=500_000, data=b"clip-bytes"):
    return SimpleNamespace(
        content_type=content_type,
        filename=filename,
        size=size,
        url=f"https://cdn.discordapp.com/attachments/1/2/{filename}?ex=signed",
        read=AsyncMock(return_value=data),
    )


def _message(attachments=(), content=""):
    return SimpleNamespace(
        attachments=list(attachments), embeds=[], content=content, message_snapshots=[]
    )


class VisualInputDetectionTests(unittest.TestCase):
    """The classifier is told an image exists only when one actually does.

    A video attachment answering "has images" routed the request to an image
    intent whose handler received nothing, so it fell through to chat and asked
    for a picture the user had plainly attached.
    """

    def test_video_attachment_is_not_a_picture(self):
        self.assertFalse(has_visual_inputs(_message([_attachment("video/mp4", "clip.mp4")])))

    def test_text_attachment_is_not_a_picture(self):
        self.assertFalse(has_visual_inputs(_message([_attachment("text/plain", "notes.txt")])))

    def test_image_attachment_still_counts(self):
        self.assertTrue(has_visual_inputs(_message([_attachment("image/png", "cat.png")])))

    def test_image_in_a_reply_still_counts(self):
        self.assertTrue(
            has_visual_inputs(_message(), _message([_attachment("image/png", "cat.png")]))
        )

    def test_content_type_missing_falls_back_to_the_filename(self):
        self.assertTrue(has_visual_inputs(_message([_attachment(None, "cat.png")])))
        self.assertFalse(has_visual_inputs(_message([_attachment(None, "clip.mp4")])))


class GeminiVideoPartTests(unittest.IsolatedAsyncioTestCase):
    """Gemini watches video inline, so a clip is forwarded rather than dropped."""

    async def test_video_attachment_becomes_a_part(self):
        attachment = _attachment("video/mp4", "clip.mp4")
        videos, unusable = [], []

        parts = await collect_gemini_parts(
            _message([attachment]), None, [], videos, unusable
        )

        self.assertEqual(len(parts), 1)
        self.assertTrue(_has_video_part(parts))
        self.assertEqual(videos, ["clip.mp4"])
        self.assertEqual(unusable, [])

    async def test_quicktime_is_relabelled_to_the_mime_gemini_accepts(self):
        videos = []
        parts = await collect_gemini_parts(
            _message([_attachment("video/quicktime", "clip.mov")]), None, [], videos, []
        )

        self.assertEqual(videos, ["clip.mov"])
        self.assertEqual(parts[0].inline_data.mime_type, "video/mov")

    async def test_video_in_a_reply_is_watched_too(self):
        videos = []
        parts = await collect_gemini_parts(
            _message(), _message([_attachment("video/webm", "clip.webm")]), [], videos, []
        )

        self.assertEqual(videos, ["clip.webm"])
        self.assertEqual(len(parts), 1)

    async def test_oversized_video_is_reported_not_silently_dropped(self):
        attachment = _attachment("video/mp4", "long.mp4", size=MAX_ATTACHMENT_BYTES + 1)
        videos, unusable = [], []

        parts = await collect_gemini_parts(
            _message([attachment]), None, [], videos, unusable
        )

        self.assertEqual(parts, [])
        self.assertEqual(videos, [])
        self.assertEqual(unusable, ["long.mp4"])
        attachment.read.assert_not_awaited()

    async def test_unreadable_container_is_reported_without_downloading(self):
        attachment = _attachment("video/x-matroska", "clip.mkv")
        videos, unusable = [], []

        parts = await collect_gemini_parts(
            _message([attachment]), None, [], videos, unusable
        )

        self.assertEqual(parts, [])
        self.assertEqual(videos, [])
        self.assertEqual(unusable, ["clip.mkv"])
        attachment.read.assert_not_awaited()

    async def test_image_attachments_are_unaffected(self):
        videos, unusable = [], []
        parts = await collect_gemini_parts(
            _message([_attachment("image/png", "cat.png")]), None, [], videos, unusable
        )

        self.assertEqual(len(parts), 1)
        self.assertFalse(_has_video_part(parts))
        self.assertEqual(videos, [])
        self.assertEqual(unusable, [])


class VideoChatRoutingTests(unittest.IsolatedAsyncioTestCase):
    """A clip is answered by the backend that can watch it.

    OpenAI chat takes images only, so an attached video routed there reaches the
    model as nothing at all.
    """

    def _context(self, **kwargs):
        return DispatchContext(
            intent="chat",
            message=SimpleNamespace(),
            prompt="what happens here",
            raw_prompt="what happens here",
            user_id=123,
            bot_user=SimpleNamespace(id=456),
            **kwargs,
        )

    async def _dispatched_model(self, context):
        self.enterContext(patch("bot.intent_dispatcher.confirm_provider_fallback", new=AsyncMock(return_value=True)))
        with patch(
            "bot.intent_dispatcher.handle_chat_intent", new_callable=AsyncMock
        ) as handle:
            self.assertTrue(await _dispatch_builtin_intent(context))
        return handle.await_args.kwargs["default_model"]

    async def test_attached_video_routes_chat_to_gemini(self):
        model = await self._dispatched_model(self._context(video_inputs=["clip.mp4"]))
        self.assertEqual(model, GEMINI_FALLBACK_CHAT_MODEL)

    async def test_text_only_chat_keeps_its_configured_model(self):
        model = await self._dispatched_model(self._context())
        self.assertNotEqual(model, GEMINI_FALLBACK_CHAT_MODEL)


class MediaResolutionTests(unittest.TestCase):
    """Video is billed per second of frames; stills keep full fidelity."""

    def test_only_video_parts_are_detected(self):
        video = SimpleNamespace(inline_data=SimpleNamespace(mime_type="video/mp4"))
        image = SimpleNamespace(inline_data=SimpleNamespace(mime_type="image/png"))
        text = SimpleNamespace(inline_data=None)

        self.assertTrue(_has_video_part([text, image, video]))
        self.assertFalse(_has_video_part([text, image]))
        self.assertFalse(_has_video_part(None))


if __name__ == "__main__":
    unittest.main()
