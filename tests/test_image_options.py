import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot import app_actions, image_handler, image_options
from providers import gemini_images, stability_generation
from services import image_options as catalog, model_preferences, server_policy
from services.sqlite_store import SQLiteStore


def interaction(user_id=7):
    return SimpleNamespace(user=SimpleNamespace(id=user_id), response=SimpleNamespace(
        send_message=AsyncMock(), defer=AsyncMock()))


def config_for(model="gpt-image-1-mini", quality="low", size="1024x1024", aspect="1:1"):
    return next(c for c in catalog.IMAGE_CONFIGS if (c.model, c.quality, c.size, c.aspect) == (model, quality, size, aspect))


class ImageOptionsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SQLiteStore(Path(directory.name))
        self.enterContext(patch.object(server_policy, "store", self.store))
        self.enterContext(patch.object(model_preferences, "store", self.store))
        self.availability = self.enterContext(patch.object(image_options, "available_image_providers", return_value={"openai": True, "gemini": True}))

    async def pick(self, action=None, *, timeout=False):
        self.reply = SimpleNamespace(edit=AsyncMock())
        self.view = None
        async def send(content, **kwargs):
            self.view = kwargs["view"]
            self.assertIn("30 seconds", content)
            return self.reply
        async def wait(view):
            if action:
                await action(view)
            if timeout:
                view.deadline = 0
                raise asyncio.TimeoutError
            return False
        with patch.object(image_options.ImageOptionsView, "wait", wait):
            return await image_options.choose_image_config(7, send, guild_id=40)

    async def test_menu_serializes_all_prices_sizes_and_providers(self):
        view = image_options.ImageOptionsView(7, catalog.IMAGE_CONFIGS)
        self.addCleanup(view.stop)
        self.assertEqual(len(view.select.options), 24)
        for option in view.select.options:
            self.assertLessEqual(len(option.label), 100)
            self.assertLessEqual(len(option.value), 100)
            self.assertIn("$", option.label)
        self.assertEqual(config_for().cost, 0.005)
        self.assertEqual(config_for(quality="high", size="1536x1024").cost, 0.052)
        self.assertIsNone(config_for("gpt-image-2.5-sunburst").cost)
        self.assertEqual(config_for("gemini-3-pro-image", "standard", "4K").cost, 0.24)
        with patch.dict("os.environ", {"GEMINI_IMAGE_COST_4K_USD": "0.30"}):
            self.assertEqual(config_for("gemini-3-pro-image", "standard", "4K").cost, 0.30)
        view.to_components()

    async def test_timeout_uses_only_low_square_mini(self):
        self.assertEqual(image_options.IMAGE_CHOICE_SECONDS, 30)
        result = await self.pick(timeout=True)
        self.assertEqual(result, catalog.DEFAULT_IMAGE_CONFIG)
        self.assertIn("30 seconds elapsed", self.reply.edit.await_args.kwargs["content"])
        self.assertIsNone(self.reply.edit.await_args.kwargs["view"])
        self.assertTrue(self.view.is_finished())

    async def test_saved_quality_highlights_choice_but_never_upgrades_timeout(self):
        model_preferences.set_preferences(7, "image", "quality")
        self.assertEqual(await self.pick(timeout=True), catalog.DEFAULT_IMAGE_CONFIG)
        highlighted = [option.value for option in self.view.select.options if option.default]
        self.assertEqual(highlighted, [config_for("gpt-image-2.5-sunburst", "high").value])

    async def test_picked_gemini_configuration_is_returned_immediately(self):
        choice = config_for("gemini-3-pro-image", "standard", "4K", "16:9")
        async def choose(view):
            select = view.select
            select._values = [choice.value]
            await select.callback(interaction())
        self.assertEqual(await self.pick(choose), choice)
        self.assertIn("4K 16:9", self.reply.edit.await_args.kwargs["content"])

    async def test_cancel_never_uses_timeout_default(self):
        async def cancel(view):
            await view.cancel.callback(interaction())
        self.assertIsNone(await self.pick(cancel))
        self.assertIn("cancelled", self.reply.edit.await_args.kwargs["content"])

    async def test_only_requester_can_choose_or_cancel(self):
        view = image_options.ImageOptionsView(7, catalog.IMAGE_CONFIGS)
        self.addCleanup(view.stop)
        outsider = interaction(8)
        view.select._values = [catalog.IMAGE_CONFIGS[-1].value]
        await view.select.callback(outsider)
        await view.cancel.callback(outsider)
        self.assertIsNone(view.value)
        self.assertFalse(view.is_finished())
        self.assertEqual(outsider.response.send_message.await_count, 2)

    async def test_first_selection_wins_and_late_clicks_are_rejected(self):
        view = image_options.ImageOptionsView(7, catalog.IMAGE_CONFIGS)
        view.select._values = [catalog.IMAGE_CONFIGS[-1].value]
        await asyncio.gather(view.select.callback(interaction()), view.cancel.callback(interaction()))
        self.assertEqual(view.value, catalog.IMAGE_CONFIGS[-1].value)
        late = image_options.ImageOptionsView(7, catalog.IMAGE_CONFIGS)
        self.addCleanup(late.stop)
        late.deadline = 0
        late.select._values = [catalog.IMAGE_CONFIGS[-1].value]
        await late.select.callback(interaction())
        self.assertIsNone(late.value)

    async def test_unavailable_openai_never_autostarts_gemini(self):
        self.availability.return_value = {"openai": False, "gemini": True}
        self.assertIsNone(await self.pick(timeout=True))
        self.assertTrue(all(c.provider == "gemini" for c in self.view.allowed.values()))

    async def test_no_available_provider_or_picker_delivery_sends_no_configuration(self):
        self.availability.return_value = {}
        send = AsyncMock()
        self.assertIsNone(await image_options.choose_image_config(7, send))
        self.assertNotIn("view", send.await_args.kwargs)
        self.availability.return_value = {"openai": True}
        self.assertIsNone(await image_options.choose_image_config(7, AsyncMock(side_effect=OSError("offline"))))

    async def test_server_ceiling_filters_choices_and_is_rechecked_after_wait(self):
        server_policy.update_policy(40, workload="image", max_tier="economy")
        self.assertEqual(await self.pick(timeout=True), catalog.DEFAULT_IMAGE_CONFIG)
        self.assertEqual(len(self.view.allowed), 3)
        self.assertTrue(all(c.model == "gpt-image-1-mini" and c.quality == "low" for c in self.view.allowed.values()))
        server_policy.update_policy(40, workload="image", max_tier="auto")
        async def change_cap(view):
            view.value = catalog.IMAGE_CONFIGS[-1].value
            server_policy.update_policy(40, workload="image", max_tier="economy")
        self.assertIsNone(await self.pick(change_cap))
        self.assertIn("no longer available", self.reply.edit.await_args.kwargs["content"])

    async def test_balanced_cap_and_unknown_config(self):
        server_policy.update_policy(40, workload="image", max_tier="balanced")
        self.assertTrue(server_policy.image_config_allowed(config_for("gpt-image-2.5-sunburst", "medium"), 40))
        self.assertFalse(server_policy.image_config_allowed(config_for("gpt-image-2.5-sunburst", "high"), 40))
        self.assertFalse(server_policy.image_config_allowed(config_for("gemini-3-pro-image", "standard", "2K"), 40))
        self.assertFalse(server_policy.image_config_allowed(catalog.ImageConfig("openai", "unknown", "low", "1024x1024"), 40))

    async def test_timeout_does_not_extend_on_rejected_interaction(self):
        # Exercise the real outer timer with a short duration, without sleeping 30s.
        reply = SimpleNamespace(edit=AsyncMock())
        calls = 0
        async def send(content, **kwargs):
            async def rejected():
                nonlocal calls
                await kwargs["view"].interaction_check(interaction(8))
                calls += 1
            await rejected()
            return reply
        with patch.object(image_options, "IMAGE_CHOICE_SECONDS", 0.01):
            self.assertEqual(await asyncio.wait_for(image_options.choose_image_config(7, send), timeout=1), catalog.DEFAULT_IMAGE_CONFIG)
        self.assertEqual(calls, 1)

    async def test_provider_disappears_during_wait(self):
        async def gone(view):
            view.value = catalog.DEFAULT_IMAGE_CONFIG.value
            self.availability.return_value = {}
        self.assertIsNone(await self.pick(gone))


class SelectedImageExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.object(catalog, "available_image_providers", return_value={"openai": True, "gemini": True}))

    async def test_selected_openai_quality_size_and_references_reach_renderer(self):
        choice = config_for(quality="medium", size="1024x1536")
        with patch.object(stability_generation, "generate_gpt_image", AsyncMock(return_value=io.BytesIO(b"image"))) as render:
            await stability_generation.handle_image_generation(None, "gemini imagine this", reference_urls=["data:image/png;base64,YQ=="], **choice.provider_kwargs())
        self.assertEqual(render.await_args.kwargs["model"], choice.model)
        self.assertEqual(render.await_args.kwargs["quality"], "medium")
        self.assertEqual(render.await_args.kwargs["size"], "1024x1536")
        self.assertEqual(render.await_args.kwargs["references"], [b"a"])

    async def test_selected_gemini_size_aspect_apply_to_both_reference_and_new_images(self):
        choice = config_for("gemini-3-pro-image", "standard", "4K", "9:16")
        with patch.object(stability_generation, "generate_gemini_with_references", return_value=io.BytesIO(b"image")) as refs, patch.object(
            stability_generation, "generate_gemini_image", return_value=io.BytesIO(b"image")
        ) as plain:
            state = {}
            await stability_generation.handle_image_generation(None, "a cat", provider_state=state, **choice.provider_kwargs())
            await stability_generation.handle_image_generation(None, "a cat", reference_urls=["data:image/png;base64,YQ=="], **choice.provider_kwargs())
        for call in (refs.call_args, plain.call_args):
            self.assertEqual(call.kwargs, {"image_size": "4K", "aspect_ratio": "9:16"})
        self.assertEqual(refs.call_args.args[1][0].read(), b"a")
        self.assertEqual(state["model"], choice.label)

    async def test_selected_provider_failure_never_switches_or_proposes_an_unpriced_retry(self):
        with patch.object(stability_generation, "generate_gpt_image", AsyncMock(return_value=None)), patch.object(stability_generation, "generate_gemini_image") as gemini:
            self.assertIsNone(await stability_generation.handle_image_generation(None, "stable imagine a cat", **catalog.DEFAULT_IMAGE_CONFIG.provider_kwargs()))
        gemini.assert_not_called()

    async def test_provider_rechecks_policy_after_resolving_references(self):
        choice = config_for("gpt-image-2.5-sunburst", "high")
        async def references(**kwargs):
            # A manager lowers the ceiling while attachments are being fetched.
            policy.return_value = False
            return [b"reference"]
        with patch.object(server_policy, "image_config_allowed", return_value=True) as policy, patch.object(
            stability_generation, "collect_reference_images", AsyncMock(side_effect=references)
        ), patch.object(stability_generation, "generate_gpt_image", AsyncMock()) as provider:
            self.assertIsNone(await stability_generation.handle_image_generation(None, "a cat", **choice.provider_kwargs()))
        provider.assert_not_awaited()

    async def test_gemini_failure_does_not_submit_openai(self):
        choice = config_for("gemini-3-pro-image", "standard", "2K")
        with patch.object(stability_generation, "generate_gemini_image", return_value=None), patch.object(stability_generation, "generate_gpt_image", AsyncMock()) as openai:
            self.assertIsNone(await stability_generation.handle_image_generation(None, "a cat", **choice.provider_kwargs()))
        openai.assert_not_awaited()

    async def test_mini_default_reaches_chat_provider_without_billable_partial_previews(self):
        message = SimpleNamespace(author=SimpleNamespace(id=7), guild=None, reply=AsyncMock())
        output_status = SimpleNamespace(edit=AsyncMock())
        async def status(message, **kwargs):
            return output_status, await kwargs["coro"]
        with patch.object(image_handler, "choose_image_config", AsyncMock(return_value=catalog.DEFAULT_IMAGE_CONFIG)), patch.object(
            image_handler, "handle_image_generation", AsyncMock(return_value=io.BytesIO(b"image"))
        ) as render:
            await image_handler.handle_generate_image_intent(message=message, prompt="a cat", ref_msg=None, duration_estimate=1, stream_ok=True, live_status_with_progress=status)
        self.assertIsNone(render.await_args.kwargs["partial_callback"])
        self.assertEqual(render.await_args.kwargs["image_size"], "1024x1024")
        self.assertEqual(render.await_args.kwargs["image_quality"], "low")

    async def test_app_picker_is_private_and_executes_selected_gemini_configuration(self):
        item = SimpleNamespace(user=SimpleNamespace(id=7), guild=None, channel=None, response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()))
        choice = config_for("gemini-3-pro-image", "standard", "4K", "16:9")
        async def choose(user_id, send, **kwargs):
            await send("settings", view="picker")
            return choice
        with patch.object(app_actions, "choose_image_config", AsyncMock(side_effect=choose)), patch.object(
            app_actions, "check_rate_limit", return_value=SimpleNamespace(allowed=True)
        ), patch.object(stability_generation, "handle_image_generation", AsyncMock(return_value=io.BytesIO(b"image"))) as render:
            await app_actions.AppActionsCog(SimpleNamespace()).generate(item, "a cat")
        self.assertTrue(item.followup.send.await_args_list[0].kwargs["ephemeral"])
        self.assertTrue(item.followup.send.await_args_list[0].kwargs["wait"])
        self.assertTrue(render.await_args.kwargs["use_gemini"])
        self.assertEqual(render.await_args.kwargs["gemini_image_size"], "4K")
        self.assertEqual(render.await_args.kwargs["gemini_aspect_ratio"], "16:9")
        self.assertIn(choice.label, item.followup.send.await_args.args[0])

    async def test_app_rechecks_channel_after_selection(self):
        item = SimpleNamespace(user=SimpleNamespace(id=7), guild=None, channel=None, response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()))
        from services.premium import PremiumError
        with patch.object(app_actions, "choose_image_config", AsyncMock(return_value=catalog.DEFAULT_IMAGE_CONFIG)), patch.object(
            app_actions, "check_rate_limit", return_value=SimpleNamespace(allowed=True)
        ), patch.object(server_policy, "require_active_channel", AsyncMock(side_effect=[None, PremiumError("Channel disabled.")])), patch.object(
            stability_generation, "handle_image_generation", AsyncMock()
        ) as render:
            await app_actions.AppActionsCog(SimpleNamespace()).generate(item, "a cat")
        render.assert_not_awaited()
        self.assertIn("Channel disabled", item.followup.send.await_args.args[0])

    async def test_gemini_sdk_receives_chosen_config_and_matching_cost(self):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"image"))]))))
        for references in (False, True):
            with self.subTest(references=references), patch.object(gemini_images, "get_gemini_client", return_value=client), patch.object(
                gemini_images, "_record_gemini_image_cost"
            ) as record, patch.object(gemini_images.PILImage, "open", return_value=Mock()):
                if references:
                    output = gemini_images.generate_gemini_with_references("a cat", [io.BytesIO(b"a")], image_size="4K", aspect_ratio="16:9")
                else:
                    output = gemini_images.generate_gemini_image("a cat", image_size="4K", aspect_ratio="16:9")
                self.assertEqual(output.read(), b"image")
                config = client.models.generate_content.call_args.kwargs["config"].image_config
                self.assertEqual((config.image_size, config.aspect_ratio), ("4K", "16:9"))
                self.assertEqual(record.call_args.kwargs["image_size"], "4K")

    async def test_chat_cancel_and_app_cancel_make_no_provider_request(self):
        message = SimpleNamespace(author=SimpleNamespace(id=7), guild=None, reply=AsyncMock())
        with patch.object(image_handler, "choose_image_config", AsyncMock(return_value=None)), patch.object(image_handler, "invoke_provider", AsyncMock()) as provider:
            await image_handler.handle_generate_image_intent(message=message, prompt="a cat", ref_msg=None, duration_estimate=1, stream_ok=False, live_status_with_progress=AsyncMock())
        provider.assert_not_awaited()
        item = SimpleNamespace(user=SimpleNamespace(id=7), guild=None, channel=None, response=SimpleNamespace(defer=AsyncMock()), followup=SimpleNamespace(send=AsyncMock()))
        with patch.object(app_actions, "choose_image_config", AsyncMock(return_value=None)), patch.object(app_actions, "check_rate_limit", return_value=SimpleNamespace(allowed=True)), patch.object(stability_generation, "handle_image_generation", AsyncMock()) as provider:
            await app_actions.AppActionsCog(SimpleNamespace()).generate(item, "a cat")
        provider.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
