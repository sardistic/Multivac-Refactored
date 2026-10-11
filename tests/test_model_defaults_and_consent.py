import asyncio
import base64
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot import chat_handler, image_handler, intent_dispatcher, video_handler
from bot.fallback_consent import FallbackConsentView, confirm_provider_fallback
from bot.model_defaults import ModelDefaultsCog
from providers import stability_generation
from services import model_preferences
from services.model_preferences import ImageModelChoice
from services.provider_fallback import ProviderFallback
from services.sqlite_store import SQLiteStore


def interaction(user_id=7):
    return SimpleNamespace(user=SimpleNamespace(id=user_id), response=SimpleNamespace(
        send_message=AsyncMock(), edit_message=AsyncMock(), defer=AsyncMock(),
    ), followup=SimpleNamespace(send=AsyncMock()))


def message():
    return SimpleNamespace(author=SimpleNamespace(id=7), guild=None,
                           channel=SimpleNamespace(id=9, send=AsyncMock()),
                           content="imagine a cat", reply=AsyncMock())


async def status(message, *, coro, **kwargs):
    return kwargs.get("existing_status_msg") or SimpleNamespace(edit=AsyncMock()), await coro


class FallbackConsentTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_requester_can_approve(self):
        view = FallbackConsentView(7, "Gemini")
        outsider = interaction(8)
        await view.confirm.callback(outsider)
        self.assertFalse(view.accepted)
        self.assertFalse(view.decided)
        outsider.response.send_message.assert_awaited_once()
        outsider.response.edit_message.assert_not_awaited()
        view.stop()

    async def test_first_click_wins_and_removes_buttons(self):
        view = FallbackConsentView(7, "Gemini")
        request = interaction()
        await asyncio.gather(view.confirm.callback(request), view.cancel.callback(request))
        self.assertTrue(view.accepted)
        self.assertTrue(view.is_finished())
        request.response.edit_message.assert_awaited_once()
        self.assertIsNone(request.response.edit_message.await_args.kwargs["view"])

    async def test_cancel_and_timeout_never_approve(self):
        view = FallbackConsentView(7, "Gemini")
        await view.cancel.callback(interaction())
        self.assertFalse(view.accepted)
        expired = FallbackConsentView(7, "Gemini")
        expired.message = SimpleNamespace(edit=AsyncMock())
        await expired.on_timeout()
        await expired.confirm.callback(interaction())
        self.assertFalse(expired.accepted)
        expired.message.edit.assert_awaited_once()
        expired.stop()

    async def test_click_past_deadline_cannot_approve(self):
        view = FallbackConsentView(7, "Gemini")
        view.deadline = 0
        await view.confirm.callback(interaction())
        self.assertFalse(view.accepted)
        self.assertTrue(view.is_finished())

    async def test_confirmation_waits_for_requesters_click(self):
        request = message()
        shown = asyncio.Event()
        views = []

        async def reply(content, *, view):
            views.append(view)
            shown.set()
            return SimpleNamespace(edit=AsyncMock())

        request.reply.side_effect = reply
        pending = asyncio.create_task(confirm_provider_fallback(request, model="Gemini Image", reason="Render failed."))
        await shown.wait()
        self.assertFalse(pending.done())
        await views[0].confirm.callback(interaction())
        self.assertTrue(await pending)
        self.assertIn("attachments", request.reply.await_args.args[0])

    async def test_failed_confirmation_delivery_fails_closed(self):
        request = message()
        request.reply.side_effect = OSError("offline")
        self.assertFalse(await confirm_provider_fallback(request, model="Gemini", reason="Failed."))


class PersonalModelDefaultsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.store = SQLiteStore(self.path)
        self.enterContext(patch.object(model_preferences, "store", self.store))

    async def test_preferences_survive_restart_and_do_not_change_other_accounts(self):
        model_preferences.set_preferences(7, "all", "economy")
        self.assertTrue(all(value == "economy" for value in self.store.get_model_preferences(7).values()))
        self.assertEqual(SQLiteStore(self.path).get_model_preferences(7), self.store.get_model_preferences(7))
        self.assertEqual(model_preferences.get_preferences(8), {})
        model_preferences.set_preferences(7, "image", "balanced")
        self.assertEqual(model_preferences.image_choice(7), ImageModelChoice(quality="medium"))
        model_preferences.set_preferences(7, "all", "auto")
        self.assertEqual(model_preferences.get_preferences(7), {})

    async def test_invalid_model_defaults_do_not_write(self):
        for workload, tier in (("other", "economy"), ("chat", "arbitrary-model")):
            with self.assertRaises(ValueError):
                model_preferences.set_preferences(7, workload, tier)
        self.assertEqual(model_preferences.get_preferences(7), {})

    async def test_slash_command_changes_only_callers_defaults(self):
        request = interaction()
        await ModelDefaultsCog.defaults.callback(ModelDefaultsCog(), request, "all", "economy")
        self.assertEqual(model_preferences.preference_tier(7, "image"), "economy")
        self.assertEqual(model_preferences.preference_tier(8, "image"), "auto")
        self.assertTrue(request.followup.send.await_args.kwargs["ephemeral"])

    async def test_dispatch_preserves_research_tools_with_economy_model(self):
        model_preferences.set_preferences(7, "research", "economy")
        context = intent_dispatcher.DispatchContext(
            intent="chat_research", message=message(), prompt="research cats", raw_prompt="research cats", user_id=7,
            bot_user=None,
        )
        with patch.object(intent_dispatcher, "handle_chat_intent", new_callable=AsyncMock) as handle:
            await intent_dispatcher._dispatch_builtin_intent(context)
        self.assertEqual(handle.await_args.kwargs["default_model"], "gpt-6-luna")
        self.assertTrue(handle.await_args.kwargs["force_web_search"])
        self.assertTrue(handle.await_args.kwargs["model_preference_locked"])

    async def test_saved_image_tier_and_explicit_gemini_provider(self):
        model_preferences.set_preferences(7, "image", "economy")
        for prompt in ("imagine a cat", "gemini imagine a cat"):
            context = intent_dispatcher.DispatchContext(
                intent="generate_image", message=message(), prompt=prompt, raw_prompt=prompt, user_id=7,
                bot_user=None,
            )
            with patch.object(intent_dispatcher, "handle_generate_image_intent", new_callable=AsyncMock) as handle:
                await intent_dispatcher._dispatch_builtin_intent(context)
            args = handle.await_args.kwargs
            if prompt.startswith("gemini"):
                self.assertTrue(args["use_gemini"])
                self.assertIsNone(args["image_choice"])
            else:
                self.assertEqual(args["image_choice"], ImageModelChoice("gpt-image-1-mini", "low"))


    async def test_video_default_preselects_without_starting_a_job(self):
        model_preferences.set_preferences(7, "video", "economy")
        request = message()
        async def reply(content, *, view):
            self.assertIsNone(view.value)
            selector = next(child for child in view.children if isinstance(child, video_handler.VideoConfigSelect))
            selected = [option.value for option in selector.options if option.default]
            self.assertEqual(selected, ["sora|sora-2|4"])
            await view.cancel_button.callback(interaction())
            return SimpleNamespace(edit=AsyncMock())
        request.reply.side_effect = reply
        with patch.object(video_handler, "_resolve_video_reference_upload", new=AsyncMock(return_value=None)), patch.object(
            video_handler, "veo_is_available", return_value=False
        ), patch.object(video_handler, "invoke_provider", new_callable=AsyncMock) as provider:
            await video_handler.handle_generate_video_intent(request, "a cat", 7, status, False)
        provider.assert_not_awaited()
        defaults = [option["value"] for option in video_handler.build_video_config_options(False) if option["default"]]
        self.assertEqual(defaults, ["sora|sora-2-pro|8"])


class ProviderRetryFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_weather_image_respects_economy_and_declined_fallback(self):
        proposal = ProviderFallback("Gemini", "Gemini Image", "OpenAI failed.", AsyncMock())
        with patch.object(image_handler, "get_location_details", new=AsyncMock(return_value={"name": "US", "lat": 1, "lon": 2})), patch.object(
            image_handler, "get_weather_data", new=AsyncMock(return_value={"current": {"main": {
                "temp": 70, "temp_min": 65, "temp_max": 75, "feels_like": 70,
            }}})
        ), patch.object(image_handler, "handle_image_generation", new=AsyncMock(return_value=proposal)) as render, patch.object(
            image_handler, "confirm_provider_fallback", new=AsyncMock(return_value=False)
        ), patch.object(image_handler, "_generate_gemini_image_threaded", new_callable=AsyncMock) as gemini:
            await image_handler.handle_generate_image_intent(
                message=message(), prompt="imagine weather New York", ref_msg=None,
                duration_estimate=1, stream_ok=False, live_status_with_progress=status,
                image_choice=ImageModelChoice("gpt-image-1-mini", "low"),
            )
        self.assertEqual(render.await_args.kwargs["image_model"], "gpt-image-1-mini")
        self.assertEqual(render.await_args.kwargs["image_quality"], "low")
        gemini.assert_not_awaited()
        proposal.retry.assert_not_awaited()

    async def test_economy_image_stream_retry_keeps_selected_model_and_quality(self):
        result = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"image").decode())])
        generate = AsyncMock(side_effect=[OSError("stream unavailable"), result])
        client = SimpleNamespace(images=SimpleNamespace(generate=generate))
        with patch.object(stability_generation, "get_openai_image_client", return_value=client), patch.object(
            stability_generation, "_record_image_usage"
        ) as record:
            image = await stability_generation.generate_gpt_image(
                "a cat", partial_callback=lambda *args: None, model="gpt-image-1-mini", quality="low",
            )
        self.assertEqual(image.read(), b"image")
        self.assertEqual(generate.await_count, 2)
        for call in generate.await_args_list:
            self.assertEqual(call.kwargs["model"], "gpt-image-1-mini")
            self.assertEqual(call.kwargs["quality"], "low")
        self.assertEqual(record.call_args.kwargs["model"], "gpt-image-1-mini")

    async def test_mini_reference_edit_keeps_input_without_unsupported_fidelity(self):
        result = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"image").decode())])
        edit = AsyncMock(return_value=result)
        client = SimpleNamespace(images=SimpleNamespace(edit=edit))
        with patch.object(stability_generation, "get_openai_image_client", return_value=client), patch.object(
            stability_generation, "_record_image_usage"
        ):
            await stability_generation.generate_gpt_image("a cat", references=[b"reference"], model="gpt-image-1-mini", quality="low")
        self.assertEqual(edit.await_args.kwargs["image"][0].read(), b"reference")
        self.assertNotIn("input_fidelity", edit.await_args.kwargs)
        self.assertEqual(edit.await_args.kwargs["model"], "gpt-image-1-mini")

    async def test_video_decline_never_calls_chat_provider(self):
        context = intent_dispatcher.DispatchContext(
            intent="chat", message=message(), prompt="describe", raw_prompt="describe", user_id=7,
            bot_user=None, video_inputs=["clip.mp4"],
        )
        with patch.object(intent_dispatcher, "confirm_provider_fallback", new=AsyncMock(return_value=False)), patch.object(
            intent_dispatcher, "handle_chat_intent", new_callable=AsyncMock
        ) as handle:
            self.assertTrue(await intent_dispatcher._dispatch_builtin_intent(context))
        handle.assert_not_awaited()

    async def test_image_render_is_not_sent_to_gemini_until_confirmed(self):
        for accepted in (False, True):
            retry = AsyncMock(return_value=io.BytesIO(b"image"))
            proposal = ProviderFallback("Gemini", "Gemini Image", "OpenAI failed.", retry)
            async def consent(*args, **kwargs):
                retry.assert_not_awaited()
                return accepted
            with patch.object(image_handler, "handle_image_generation", new=AsyncMock(return_value=proposal)), patch.object(
                image_handler, "confirm_provider_fallback", new=AsyncMock(side_effect=consent)
            ):
                await image_handler.handle_generate_image_intent(
                    message=message(), prompt="imagine a cat", ref_msg=None,
                    duration_estimate=1, stream_ok=False, live_status_with_progress=status,
                )
            self.assertEqual(retry.await_count, int(accepted))

    async def test_openai_failure_keeps_reference_bytes_for_approved_retry(self):
        reference = "data:image/png;base64," + base64.b64encode(b"reference").decode()
        with patch.object(stability_generation, "generate_gpt_image", new=AsyncMock(return_value=None)), patch.object(
            stability_generation, "generate_gemini_with_references", return_value=io.BytesIO(b"render")
        ) as gemini:
            proposal = await stability_generation.handle_image_generation(None, "imagine this", reference_urls=[reference])
            gemini.assert_not_called()
            await proposal.retry()
        self.assertEqual(gemini.call_args.args[1][0].read(), b"reference")

    async def test_declined_edit_never_invokes_gemini(self):
        with patch.object(image_handler, "_openai_edit_image", new=AsyncMock(side_effect=OSError())), patch.object(
            image_handler, "confirm_provider_fallback", new=AsyncMock(return_value=False)
        ), patch.object(image_handler, "edit_gemini_image") as gemini:
            await image_handler.handle_edit_image_intent(
                message=message(), prompt="make blue", image_urls=["data:image/png;base64,YQ=="],
                prompt_for_image_selection=AsyncMock(), live_status_with_progress=status,
            )
        gemini.assert_not_called()

    async def test_declined_vision_retry_never_invokes_gemini(self):
        with patch.object(image_handler, "generate_openai_messages_response", new=AsyncMock(return_value="⚠️ OpenAI error: offline")), patch.object(
            image_handler, "confirm_provider_fallback", new=AsyncMock(return_value=False)
        ), patch.object(image_handler, "generate_gemini_text") as gemini, patch.object(
            image_handler, "apply_personality_overrides", side_effect=lambda user_id, *, intent, text: text
        ):
            await image_handler.handle_describe_image_intent(
                message=message(), prompt="explain", image_urls=["data:image/png;base64,YQ=="],
                ref_msg=None, is_reply_to_bot=False, duration_estimate=1, stream_ok=False,
                live_status_with_progress=status, send_or_edit_with_truncation=AsyncMock(),
            )
        gemini.assert_not_called()

    async def test_chat_fallback_asks_and_stops_on_decline(self):
        calls = []
        async def invoke(name, fallback, *args, **kwargs):
            calls.append(name)
            return "⚠️ OpenAI error: offline" if name == "chat.openai" else ("Gemini answer", [])
        with patch.object(chat_handler, "build_chat_context", return_value=[{"role": "user", "content": "hello"}]), patch.object(
            chat_handler, "invoke_provider", new=AsyncMock(side_effect=invoke)
        ), patch.object(chat_handler, "confirm_provider_fallback", new=AsyncMock(return_value=False)) as confirm:
            await self._chat()
        confirm.assert_awaited_once()
        self.assertEqual(calls, ["chat.openai"])

    async def test_approved_chat_retry_runs_gemini_once(self):
        calls = []
        async def invoke(name, fallback, *args, **kwargs):
            calls.append(name)
            return "⚠️ OpenAI error: offline" if name == "chat.openai" else ("Gemini answer", [])
        async def consent(*args, **kwargs):
            self.assertEqual(calls, ["chat.openai"])
            return True
        with patch.object(chat_handler, "build_chat_context", return_value=[{"role": "user", "content": "hello"}]), patch.object(
            chat_handler, "invoke_provider", new=AsyncMock(side_effect=invoke)
        ), patch.object(chat_handler, "confirm_provider_fallback", new=AsyncMock(side_effect=consent)), patch.object(
            chat_handler, "verify_chat_draft", new=AsyncMock(return_value=SimpleNamespace(action="accept"))
        ):
            await self._chat()
        self.assertEqual(calls, ["chat.openai", "chat.gemini"])

    async def test_empty_economy_chat_never_upgrades_to_expensive_default(self):
        with patch.object(chat_handler, "build_chat_context", return_value=[{"role": "user", "content": "hello"}]), patch.object(
            chat_handler, "invoke_provider", new=AsyncMock(return_value="")
        ) as invoke:
            await self._chat(default_model="gpt-6-luna", model_preference_locked=True)
        invoke.assert_awaited_once()
        self.assertEqual(invoke.await_args.kwargs["model"], "gpt-6-luna")

    async def _chat(self, **kwargs):
        await chat_handler.handle_chat_intent(
            message=message(), prompt="hello", raw_prompt="hello", user_id=7,
            ref_msg=None, is_reply_to_bot=False, image_urls=[], gemini_parts=[],
            duration_estimate=1, stream_ok=False, live_status_with_progress=status,
            send_or_edit_with_truncation=AsyncMock(), moderation_view_factory=None, **kwargs,
        )


if __name__ == "__main__":
    unittest.main()
