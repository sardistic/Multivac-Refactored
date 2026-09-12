import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

from openai.resources.images import AsyncImages

from bot import image_handler, intent_dispatcher
from providers import stability_generation


def image_client(**results):
    """A stand-in for the OpenAI images resource, specced against the real SDK.

    A hand-rolled namespace accepts any method name and any keyword, which is
    how a call to the non-existent `images.edits` passed its test while failing
    in production. Autospec makes the mock reject both.
    """
    images = create_autospec(AsyncImages, instance=True)
    # The SDK wraps these in @required_args, so autospec builds plain mocks for
    # them; the awaited call needs an AsyncMock, still specced by name.
    for method in ("generate", "edit"):
        setattr(images, method, AsyncMock(spec=getattr(AsyncImages, method)))
    for method, raw in results.items():
        getattr(images, method).return_value = SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(raw).decode("ascii"))]
        )
    return SimpleNamespace(images=images)


class ImageGenerationFlowTests(unittest.IsolatedAsyncioTestCase):
    @patch("bot.intent_dispatcher.handle_generate_image_intent", new_callable=AsyncMock)
    async def test_dispatch_intent_passes_reference_message_to_generate_image_handler(self, mock_handle_generate):
        ref_msg = SimpleNamespace(content="silver armor, long white hair")
        message = SimpleNamespace(content="@bot imagine a character based on this")

        await intent_dispatcher.dispatch_intent(
            intent_dispatcher.DispatchContext(
                intent="generate_image",
                message=message,
                prompt="imagine a character based on this",
                raw_prompt="imagine a character based on this",
                user_id=123,
                bot_user=SimpleNamespace(id=999),
                ref_msg=ref_msg,
                live_status_with_progress=AsyncMock(),
                send_or_edit_with_truncation=AsyncMock(),
                prompt_for_image_selection=AsyncMock(),
                moderation_view_factory=lambda **kwargs: None,
            )
        )

        self.assertIs(mock_handle_generate.await_args.kwargs["ref_msg"], ref_msg)

    @patch("providers.stability_generation.generate_gpt_image", new_callable=AsyncMock)
    async def test_handle_image_generation_includes_replied_message_content_in_prompt(self, mock_generate_gpt):
        mock_generate_gpt.return_value = "image-bytes"
        reply_msg = SimpleNamespace(
            content="A calm scholar with round glasses, short black hair, and a dark green coat.",
            author=SimpleNamespace(display_name="Ry7"),
        )

        result = await stability_generation.handle_image_generation(
            message=None,
            prompt="imagine a portrait based on these details",
            reply_msg=reply_msg,
        )

        self.assertEqual(result, "image-bytes")
        composed_prompt = mock_generate_gpt.await_args.args[0]
        self.assertIn("imagine a portrait based on these details", composed_prompt)
        self.assertIn("Replied message context from Ry7", composed_prompt)
        self.assertIn("A calm scholar with round glasses", composed_prompt)

    @patch("providers.stability_generation.generate_gpt_image", new_callable=AsyncMock)
    async def test_retry_context_replaces_generic_generated_status_text(self, mock_generate_gpt):
        mock_generate_gpt.return_value = "image-bytes"
        reply_msg = SimpleNamespace(
            content="✅ Image generated (GPT Image 2.5 Sunburst)",
            author=SimpleNamespace(display_name="Multivac"),
        )

        result = await stability_generation.handle_image_generation(
            message=None,
            prompt="not good, try again",
            reply_msg=reply_msg,
            retry_context="- draw a complex flow chart of your inner working logic",
        )

        self.assertEqual(result, "image-bytes")
        composed_prompt = mock_generate_gpt.await_args.args[0]
        self.assertIn("not good, try again", composed_prompt)
        self.assertIn("draw a complex flow chart of your inner working logic", composed_prompt)
        self.assertNotIn("✅ Image generated", composed_prompt)

    async def test_retry_context_walks_prior_generation_replies(self):
        original = SimpleNamespace(
            id=1,
            content="@Multivac draw a complex flow chart of your inner working logic",
            reference=None,
        )
        first_status = SimpleNamespace(
            id=2,
            content="✅ Image generated (GPT Image 2.5 Sunburst)",
            reference=SimpleNamespace(resolved=original),
        )
        first_retry = SimpleNamespace(
            id=3,
            content="not good, try again",
            reference=SimpleNamespace(resolved=first_status),
        )
        second_status = SimpleNamespace(
            id=4,
            content="✅ Image generated (GPT Image 2.5 Sunburst)",
            reference=SimpleNamespace(resolved=first_retry),
        )

        context = await image_handler._build_image_retry_context(second_status)

        self.assertLess(context.index("draw a complex flow chart"), context.index("not good, try again"))

    @patch("bot.image_handler.handle_image_generation", new_callable=AsyncMock)
    async def test_generated_image_is_attached_to_status_and_retry_context_is_forwarded(self, mock_generate):
        mock_generate.return_value = io.BytesIO(b"generated-image")
        original = SimpleNamespace(
            id=10,
            content="draw a complex flow chart of your inner working logic",
            reference=None,
        )
        replied_status = SimpleNamespace(
            id=11,
            content="✅ Image generated (GPT Image 2.5 Sunburst)",
            reference=SimpleNamespace(resolved=original),
        )
        output_status = SimpleNamespace(edit=AsyncMock())
        channel = SimpleNamespace(send=AsyncMock())
        message = SimpleNamespace(channel=channel)

        async def live_status(_message, **kwargs):
            return output_status, await kwargs["coro"]

        await image_handler.handle_generate_image_intent(
            message=message,
            prompt="not good, try again",
            ref_msg=replied_status,
            duration_estimate=10,
            stream_ok=False,
            live_status_with_progress=live_status,
        )

        forwarded = mock_generate.await_args.kwargs["retry_context"]
        self.assertIn("draw a complex flow chart", forwarded)
        edit_kwargs = output_status.edit.await_args.kwargs
        self.assertEqual(edit_kwargs["content"], "✅ Image generated (GPT Image 2.5 Sunburst)")
        self.assertEqual(len(edit_kwargs["attachments"]), 1)
        self.assertEqual(edit_kwargs["attachments"][0].filename, "generated_image.png")
        channel.send.assert_not_awaited()

    @patch("providers.stability_generation.generate_gemini_image")
    @patch("providers.stability_generation.generate_gpt_image", new_callable=AsyncMock)
    async def test_openai_image_failure_always_falls_back_to_gemini(self, mock_generate_gpt, mock_gemini):
        mock_generate_gpt.return_value = None
        mock_gemini.return_value = "gemini-image"
        provider_state = {}

        result = await stability_generation.handle_image_generation(
            message=None,
            prompt="imagine a moonlit library",
            provider_state=provider_state,
        )

        self.assertEqual(result, "gemini-image")
        mock_gemini.assert_called_once()
        self.assertEqual(provider_state["provider"], "Gemini")
        self.assertEqual(provider_state["model"], stability_generation.IMG_MODEL_GEMINI)

    @patch("providers.stability_generation.get_openai_image_client")
    async def test_generate_gpt_image_uses_auto_size(self, mock_get_client):
        client = image_client(generate=b"img")
        mock_get_client.return_value = client

        result = await stability_generation.generate_gpt_image("a neon city skyline")

        self.assertEqual(result.read(), b"img")
        self.assertEqual(
            client.images.generate.await_args.kwargs["model"],
            stability_generation.OPENAI_IMAGE_MODEL,
        )
        self.assertEqual(client.images.generate.await_args.kwargs["size"], "auto")

    @patch("providers.stability_generation.get_openai_image_client")
    async def test_edit_image_with_prompt_uses_auto_size(self, mock_get_client):
        client = image_client(edit=b"edited")
        mock_get_client.return_value = client

        result = await stability_generation.edit_image_with_prompt(
            "data:image/png;base64,Zm9v",
            "make it more cinematic",
        )

        self.assertEqual(result.read(), b"edited")
        self.assertEqual(
            client.images.edit.await_args.kwargs["model"],
            stability_generation.OPENAI_IMAGE_MODEL,
        )
        self.assertEqual(client.images.edit.await_args.kwargs["size"], "auto")


class AttachedImageAsGenerationReferenceTests(unittest.IsolatedAsyncioTestCase):
    """'imagine this in an art museum' with a picture attached.

    The picture used to reach neither the router (the "imagine" fast path skips
    the classifier) nor the renderer (only edit_image took image_urls, and only
    the Gemini branch collected references), so the model was handed "this"
    with nothing to resolve it against and invented a subject.
    """

    PNG_DATA_URI = "data:image/png;base64,Zm9v"  # b"foo"

    async def test_imagine_with_an_attachment_still_routes_to_generation(self):
        intent = intent_dispatcher.resolve_keyword_intent(
            "imagine this in an art museum",
            "imagine this in an art museum",
            has_attachments=True,
        )
        self.assertEqual(intent, "generate_image")

    @patch("bot.intent_dispatcher.handle_generate_image_intent", new_callable=AsyncMock)
    async def test_dispatch_hands_attached_images_to_the_generate_handler(self, mock_handle_generate):
        message = SimpleNamespace(content="@bot imagine this in an art museum")

        await intent_dispatcher.dispatch_intent(
            intent_dispatcher.DispatchContext(
                intent="generate_image",
                message=message,
                prompt="imagine this in an art museum",
                raw_prompt="imagine this in an art museum",
                user_id=123,
                bot_user=SimpleNamespace(id=999),
                image_urls=[self.PNG_DATA_URI],
                live_status_with_progress=AsyncMock(),
                send_or_edit_with_truncation=AsyncMock(),
                prompt_for_image_selection=AsyncMock(),
                moderation_view_factory=lambda **kwargs: None,
            )
        )

        self.assertEqual(
            mock_handle_generate.await_args.kwargs["image_urls"], [self.PNG_DATA_URI]
        )

    @patch("bot.image_handler.handle_image_generation", new_callable=AsyncMock)
    async def test_generate_handler_forwards_reference_urls_and_labels_the_status(self, mock_generate):
        mock_generate.return_value = io.BytesIO(b"generated-image")
        output_status = SimpleNamespace(edit=AsyncMock())
        labels = []

        async def live_status(_message, **kwargs):
            labels.append(kwargs["action_label"]())
            return output_status, await kwargs["coro"]

        await image_handler.handle_generate_image_intent(
            message=SimpleNamespace(channel=SimpleNamespace(send=AsyncMock())),
            prompt="imagine this in an art museum",
            ref_msg=None,
            duration_estimate=10,
            stream_ok=False,
            live_status_with_progress=live_status,
            image_urls=[self.PNG_DATA_URI],
        )

        self.assertEqual(
            mock_generate.await_args.kwargs["reference_urls"], [self.PNG_DATA_URI]
        )
        self.assertIn("from reference", labels[0])

    @patch("providers.stability_generation.generate_gpt_image", new_callable=AsyncMock)
    async def test_attached_image_reaches_gpt_image_as_a_reference(self, mock_generate_gpt):
        mock_generate_gpt.return_value = "image-bytes"

        result = await stability_generation.handle_image_generation(
            message=None,
            prompt="imagine this in an art museum",
            reference_urls=[self.PNG_DATA_URI],
        )

        self.assertEqual(result, "image-bytes")
        self.assertEqual(mock_generate_gpt.await_args.kwargs["references"], [b"foo"])
        composed_prompt = mock_generate_gpt.await_args.args[0]
        self.assertIn("visual reference", composed_prompt)
        self.assertIn("imagine this in an art museum", composed_prompt)

    @patch("providers.stability_generation.generate_gpt_image", new_callable=AsyncMock)
    async def test_generation_without_attachments_carries_no_reference_note(self, mock_generate_gpt):
        mock_generate_gpt.return_value = "image-bytes"

        await stability_generation.handle_image_generation(
            message=None,
            prompt="imagine a moonlit library",
        )

        self.assertEqual(mock_generate_gpt.await_args.kwargs["references"], [])
        self.assertNotIn("visual reference", mock_generate_gpt.await_args.args[0])

    @patch("providers.stability_generation.get_openai_image_client")
    async def test_references_go_to_the_edit_endpoint_not_generate(self, mock_get_client):
        client = image_client(edit=b"img")
        mock_get_client.return_value = client

        result = await stability_generation.generate_gpt_image(
            "a frog in an art museum", references=[b"foo"]
        )

        self.assertEqual(result.read(), b"img")
        client.images.generate.assert_not_awaited()
        call = client.images.edit.await_args.kwargs
        self.assertEqual(call["model"], stability_generation.OPENAI_IMAGE_MODEL)
        self.assertEqual(len(call["image"]), 1)
        self.assertEqual(call["image"][0].read(), b"foo")
        # Without high input fidelity the reference is only loose inspiration.
        self.assertEqual(call["input_fidelity"], "high")

    @patch("providers.stability_generation.generate_gemini_image")
    @patch("providers.stability_generation.generate_gemini_with_references")
    async def test_gemini_generation_uses_the_same_references(self, mock_with_refs, mock_plain):
        mock_with_refs.return_value = "gemini-image"

        result = await stability_generation.handle_image_generation(
            message=None,
            prompt="gemini imagine this in an art museum",
            use_gemini=True,
            reference_urls=[self.PNG_DATA_URI],
        )

        self.assertEqual(result, "gemini-image")
        mock_plain.assert_not_called()
        self.assertEqual(mock_with_refs.call_args.args[1][0].read(), b"foo")

    @patch("providers.stability_generation.get_openai_image_client")
    async def test_streaming_render_keeps_the_reference(self, mock_get_client):
        """The Discord path always passes a partial_callback, so the streaming
        branch is the one that actually runs in production."""

        class _Stream:
            def __init__(self, events):
                self._events = events

            def __aiter__(self):
                async def _iter():
                    for event in self._events:
                        yield event

                return _iter()

        client = image_client()
        client.images.edit.return_value = _Stream(
            [
                SimpleNamespace(
                    type="image_generation.partial_image",
                    b64_json=base64.b64encode(b"half").decode("ascii"),
                    partial_image_index=0,
                ),
                SimpleNamespace(
                    type="image_generation.completed",
                    b64_json=base64.b64encode(b"done").decode("ascii"),
                ),
            ]
        )
        mock_get_client.return_value = client
        partials = []

        result = await stability_generation.generate_gpt_image(
            "a frog in an art museum",
            partial_callback=lambda raw, index: partials.append((raw, index)),
            references=[b"foo"],
        )

        self.assertEqual(result.read(), b"done")
        self.assertEqual(partials, [(b"half", 0)])
        call = client.images.edit.await_args.kwargs
        self.assertEqual(call["model"], stability_generation.OPENAI_IMAGE_MODEL)
        self.assertTrue(call["stream"])
        self.assertEqual(call["image"][0].read(), b"foo")
        client.images.generate.assert_not_awaited()

    async def test_references_are_deduplicated_and_capped(self):
        distinct = [
            f"data:image/png;base64,{base64.b64encode(f'img{i}'.encode()).decode('ascii')}"
            for i in range(6)
        ]

        collected = await stability_generation.collect_reference_images(
            reference_urls=[distinct[0], distinct[0], *distinct[1:]]
        )

        self.assertEqual(len(collected), stability_generation.MAX_REFERENCE_IMAGES)
        self.assertEqual(len(set(collected)), len(collected))


class ImageEditFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Editing has to degrade to Gemini the way generation already does.
    With OpenAI out of quota the edit path reported "Edit failed" while a
    working image backend sat unused."""

    @staticmethod
    def _pixel_data_url() -> str:
        return "data:image/png;base64," + base64.b64encode(b"pixel-bytes").decode("ascii")

    @staticmethod
    async def _status(message, *, coro, **kwargs):
        return SimpleNamespace(edit=AsyncMock()), await coro

    async def _run_edit(self, message):
        await image_handler.handle_edit_image_intent(
            message=message,
            prompt="make them stand on him",
            image_urls=[self._pixel_data_url()],
            prompt_for_image_selection=AsyncMock(),
            live_status_with_progress=self._status,
        )

    async def test_openai_edit_failure_falls_back_to_gemini(self):
        message = SimpleNamespace(channel=SimpleNamespace(send=AsyncMock()))

        with patch.object(
            image_handler, "_openai_edit_image", new=AsyncMock(side_effect=RuntimeError("429 insufficient_quota"))
        ), patch.object(
            image_handler, "edit_gemini_image", return_value=io.BytesIO(b"gemini-edited")
        ) as gemini_edit:
            await self._run_edit(message)

        gemini_edit.assert_called_once()
        sent = [c.kwargs.get("file") for c in message.channel.send.await_args_list if c.kwargs.get("file")]
        self.assertEqual(len(sent), 1)

    async def test_openai_edit_without_an_image_falls_back_to_gemini(self):
        """A 200 that carries no image is the same outcome as an exception."""
        message = SimpleNamespace(channel=SimpleNamespace(send=AsyncMock()))

        with patch.object(
            image_handler,
            "_openai_edit_image",
            new=AsyncMock(return_value=SimpleNamespace(output=[])),
        ), patch.object(
            image_handler, "edit_gemini_image", return_value=io.BytesIO(b"gemini-edited")
        ) as gemini_edit:
            await self._run_edit(message)

        gemini_edit.assert_called_once()

    async def test_openai_edit_success_does_not_call_gemini(self):
        message = SimpleNamespace(channel=SimpleNamespace(send=AsyncMock()))
        result = base64.b64encode(b"openai-edited").decode("ascii")

        with patch.object(
            image_handler,
            "_openai_edit_image",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    output=[SimpleNamespace(type="image_generation_call", result=result)]
                )
            ),
        ), patch.object(image_handler, "edit_gemini_image") as gemini_edit:
            await self._run_edit(message)

        gemini_edit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
