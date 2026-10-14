import unittest
import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
import discord_bot

from bot import app_actions, provider_intents
from bot.characters import CharacterCog
from bot.premium import PremiumCog
from bot.throttle import SHOP_CUSTOM_ID, ThrottleShopView, is_shop_message, shop_request, throttle_notice
from bot.unattractive import EffectActivationView, UnattractiveCog
from services.premium import SKUConfig
from services.security_limits import RateLimitDecision


class ThrottleShopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = SimpleNamespace(
            refresh=AsyncMock(), config=SKUConfig(unthrottled=400),
            catalog={400: SimpleNamespace(id=400, flags=SimpleNamespace(available=True))},
            user_boost=2, guild_boost=2,
        )
        self.bot = SimpleNamespace(user=SimpleNamespace(id=999), get_cog=lambda name: self.cog if name == "Premium" else None)
        self.cog = PremiumCog(self.bot, self.service)
        self.interaction = SimpleNamespace(
            user=SimpleNamespace(id=7), guild=None, guild_id=None, channel=None,
            response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
            followup=SimpleNamespace(send=AsyncMock()), edit_original_response=AsyncMock(),
        )

    def notice(self, scope="global"):
        result = throttle_notice(self.bot, 7, RateLimitDecision(False, 12, scope), "Retry in 12 seconds.")
        self.addCleanup(result["view"].stop)
        return result

    async def test_global_notice_offers_subscription_and_keeps_retry_time(self):
        result = self.notice()
        self.assertIn("Retry in 12 seconds", result["content"])
        self.assertIn("$4.99/month", result["content"])
        self.assertIn("bypasses bot-wide caps", result["content"])
        self.assertIn("Personal and server limits still apply", result["content"])
        self.assertEqual(result["view"].children[0].label, "Open shop")
        self.assertTrue(result["view"].is_persistent())
        self.assertIn("Use /shop or /premium any time", result["content"])
        self.service.refresh.assert_not_awaited()

    async def test_personal_and_server_notices_do_not_promise_relief(self):
        for scope in ("user", "guild", "guild_user", ""):
            with self.subTest(scope=scope):
                self.assertIn("does not remove personal or server caps", self.notice(scope)["content"])

    async def test_disabled_subscription_is_not_advertised(self):
        self.service.config = SKUConfig()
        result = self.notice()
        self.assertNotIn("fUnThrottled", result["content"])
        self.assertIsInstance(result["view"], ThrottleShopView)

    async def test_missing_shop_keeps_original_notice(self):
        self.assertEqual(throttle_notice(None, 7, RateLimitDecision(False), "Retry later."),
                         {"content": "Retry later.", "view": None})

    async def test_button_opens_existing_private_shop_with_verified_purchase_button(self):
        view = self.notice()["view"]
        self.assertTrue(await view.interaction_check(self.interaction))
        await view.open_shop.callback(self.interaction)
        self.service.refresh.assert_awaited_once_with(self.bot)
        self.interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        result = self.interaction.followup.send.call_args.kwargs
        self.assertTrue(result["ephemeral"])
        self.assertEqual(result["view"].children[0].sku_id, 400)
        self.assertEqual(result["view"].children[0].style, discord.ButtonStyle.premium)
        result["view"].stop()

    async def test_failed_verification_does_not_offer_checkout(self):
        self.service.refresh.side_effect = OSError()
        await self.notice()["view"].open_shop.callback(self.interaction)
        self.assertIsNone(self.interaction.followup.send.call_args.kwargs["view"])

    async def test_another_account_can_open_its_own_private_shop(self):
        self.interaction.user.id = 8
        view = self.notice()["view"]
        self.assertTrue(await view.interaction_check(self.interaction))
        await view.open_shop.callback(self.interaction)
        self.interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        result = self.interaction.followup.send.call_args.kwargs
        self.assertTrue(result["ephemeral"])
        result["view"].stop()

    async def test_persistent_view_registered_on_startup_and_routes_after_restart(self):
        bot = discord.ext.commands.Bot(command_prefix="/", intents=discord.Intents.none())
        self.addAsyncCleanup(bot.close)
        cog = PremiumCog(bot, self.service)
        bot.get_cog = lambda name: cog if name == "Premium" else None
        await cog.cog_load()
        self.addAsyncCleanup(cog.cog_unload)
        registered = next(view for view in bot.persistent_views if view.children[0].custom_id == SHOP_CUSTOM_ID)
        self.assertIsNone(registered.timeout)
        self.assertEqual(self.notice()["view"].children[0].custom_id, registered.children[0].custom_id)
        delivered = asyncio.Event()
        async def sent(**kwargs):
            delivered.set()
        self.interaction.followup.send.side_effect = sent
        self.interaction.message = SimpleNamespace(id=1234567, interaction_metadata=None)
        self.interaction.data = {"component_type": 2, "custom_id": SHOP_CUSTOM_ID}
        bot._connection._view_store.dispatch_view(2, SHOP_CUSTOM_ID, self.interaction)
        await asyncio.wait_for(delivered.wait(), timeout=2)
        result = self.interaction.followup.send.call_args.kwargs
        self.assertTrue(result["ephemeral"])
        result["view"].stop()

    async def test_native_shop_alias_opens_same_verified_private_store(self):
        await PremiumCog.shop_alias.callback(self.cog, self.interaction)
        client = discord.Client(intents=discord.Intents.none())
        self.addAsyncCleanup(client.close)
        tree = discord.app_commands.CommandTree(client)
        self.assertEqual(PremiumCog.shop_alias.to_dict(tree)["name"], "shop")
        result = self.interaction.followup.send.call_args.kwargs
        self.assertTrue(result["ephemeral"])
        self.assertEqual(result["view"].children[0].sku_id, 400)
        result["view"].stop()

    def legacy_message(self, custom_id="a" * 32, author_id=999):
        return SimpleNamespace(
            author=SimpleNamespace(id=author_id), embeds=[], guild=None, channel=None,
            components=[SimpleNamespace(children=[SimpleNamespace(custom_id=custom_id, label="Open shop")])],
        )

    async def test_old_expired_button_is_recovered_without_a_live_view(self):
        self.interaction.type = discord.InteractionType.component
        self.interaction.data = {"custom_id": "a" * 32}
        self.interaction.message = self.legacy_message()
        self.interaction.response.is_done = lambda: False
        await self.cog.on_interaction(self.interaction)
        self.interaction.response.defer.assert_awaited_once_with(ephemeral=True)
        result = self.interaction.followup.send.call_args.kwargs
        result["view"].stop()

    async def test_legacy_recovery_ignores_current_buttons_foreign_messages_and_other_actions(self):
        self.interaction.type = discord.InteractionType.component
        self.interaction.response.is_done = lambda: False
        for custom_id, author_id in ((SHOP_CUSTOM_ID, 999), ("a" * 32, 8), ("other", 999)):
            with self.subTest(custom_id=custom_id, author_id=author_id):
                self.interaction.data = {"custom_id": custom_id}
                self.interaction.message = self.legacy_message(custom_id, author_id)
                await self.cog.on_interaction(self.interaction)
        self.interaction.response.defer.assert_not_awaited()

    async def test_shop_text_and_reference_recovery_bypass_throttles(self):
        message = SimpleNamespace(author=SimpleNamespace(id=7), guild=None, channel=None, reply=AsyncMock())
        reference = self.legacy_message()
        for prompt in ("/shop", "/premium", "shop", "Where can I access the shop", "your shop", "Can you link this again please"):
            with self.subTest(prompt=prompt), patch.object(app_actions, "check_rate_limit") as quota:
                self.assertTrue(await app_actions.handle_app_text(self.bot, message, prompt, reference=reference))
                quota.assert_not_called()
                result = message.reply.call_args.kwargs
                self.assertEqual(result["view"].children[0].sku_id, 400)
                result["view"].stop()

    async def test_generic_link_requests_require_a_bot_owned_shop_reference(self):
        for prompt, reference in (("Can you link this again please", None),
                                  ("Can you link this again please", self.legacy_message(author_id=8)),
                                  ("How do I build an online shop?", self.legacy_message()),
                                  ("Where is the shop in this game?", None)):
            with self.subTest(prompt=prompt):
                self.assertFalse(shop_request(prompt, reference, 999))

    async def test_shop_ingress_avoids_classifier_and_quota_for_mentions_and_plain_prefix(self):
        reference = self.legacy_message()
        for content in ("/shop", "<@999> /shop", "<@999> your shop", "<@999> Can you link this again please"):
            message = SimpleNamespace(id=90702, content=content, author=SimpleNamespace(id=7, bot=False),
                                      guild=None, channel=SimpleNamespace(id=3), mention_everyone=False, reply=AsyncMock())
            reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()))
            with patch.object(discord_bot.bot._connection, "user", self.bot.user), patch.object(
                discord_bot.bot, "get_cog", side_effect=self.bot.get_cog
            ), patch.object(discord_bot, "_already_processed", return_value=False), patch.object(
                discord_bot, "_get_reflection_worker", return_value=reflection
            ), patch.object(discord_bot, "resolve_reference_message", new=AsyncMock(return_value=(reference, True))), patch.object(
                discord_bot, "check_rate_limit"
            ) as quota, patch.object(discord_bot, "should_suppress_response", new=AsyncMock()) as classify:
                await discord_bot._builtin_on_message(message)
            quota.assert_not_called()
            classify.assert_not_awaited()
            result = message.reply.call_args.kwargs
            self.assertEqual(result["view"].children[0].sku_id, 400)
            result["view"].stop()

    async def test_shop_removed_after_notice_returns_private_recovery(self):
        view = self.notice()["view"]
        self.bot.get_cog = lambda name: None
        await view.open_shop.callback(self.interaction)
        self.assertTrue(self.interaction.response.send_message.call_args.kwargs["ephemeral"])

    async def test_image_app_denial_stops_before_provider_and_preserves_private_notice(self):
        for action in ("request", "image"):
            with self.subTest(action=action), patch(
                "bot.app_actions.server_policy.require_active_channel", new=AsyncMock()
            ), patch("bot.app_actions.check_rate_limit", side_effect=lambda kind, **kw:
                     RateLimitDecision(False, 12, "global") if kind == action else RateLimitDecision(True)), patch(
                "providers.stability_generation.generate_gpt_image", new=AsyncMock()
            ) as generate:
                await app_actions.AppActionsCog(self.bot).generate(self.interaction, "a cat")
                generate.assert_not_awaited()
                result = self.interaction.followup.send.call_args.kwargs
                self.assertTrue(result["ephemeral"])
                self.assertIsInstance(result["view"], ThrottleShopView)
                result["view"].stop()

    async def test_paid_character_denial_preserves_purchase(self):
        service = SimpleNamespace(purchase=AsyncMock(), render=AsyncMock())
        with patch("bot.characters.server_policy.require_active_channel", new=AsyncMock()), patch(
            "bot.characters.check_rate_limit", return_value=RateLimitDecision(False, 12, "global")
        ):
            await CharacterCog(self.bot, service).handle_interaction(self.interaction)
        service.purchase.assert_not_awaited()
        service.render.assert_not_awaited()
        result = self.interaction.followup.send.call_args.kwargs
        self.assertIn("purchase is retained", result["content"])
        self.assertTrue(result["ephemeral"])
        result["view"].stop()

    async def test_effect_activation_denial_spends_no_purchase_and_replaces_confirmation(self):
        service = SimpleNamespace(activate=AsyncMock())
        self.interaction.guild = SimpleNamespace(id=9)
        self.interaction.guild_id = 9
        view = EffectActivationView(UnattractiveCog(self.bot, service), 7, 9)
        with patch("bot.unattractive.server_policy.require_active_channel", new=AsyncMock()), patch(
            "bot.unattractive.check_rate_limit", return_value=RateLimitDecision(False, 12, "guild")
        ):
            await view.confirm.callback(self.interaction)
        service.activate.assert_not_awaited()
        result = self.interaction.edit_original_response.call_args.kwargs
        self.assertIn("no purchase was spent", result["content"])
        self.assertIn("does not remove personal or server caps", result["content"])
        result["view"].stop()
        self.assertTrue(view.is_finished())

    async def test_provider_denials_show_shop_before_starting_work(self):
        message = SimpleNamespace(author=SimpleNamespace(id=7), guild=None, reply=AsyncMock())
        progress = AsyncMock()
        with patch("bot.provider_intents.check_rate_limit", return_value=RateLimitDecision(False, 12, "global")):
            await provider_intents.handle_gemini_chat_intent(
                message=message, bot=self.bot, prompt="gemini code print(1)", gemini_parts=[],
                live_status_with_progress=progress, send_or_edit_with_truncation=AsyncMock(),
                moderation_view_factory=None,
            )
            first = message.reply.call_args.kwargs
            self.assertIn("Code-execution quota", first["content"])
            first["view"].stop()
            await provider_intents.handle_summarize_url_intent(
                message=message, bot=self.bot, url="https://example.com", duration_estimate=10,
                stream_ok=False, live_status_with_progress=progress, send_or_edit_with_truncation=AsyncMock(),
            )
        progress.assert_not_awaited()
        result = message.reply.call_args.kwargs
        self.assertIn("URL-fetch quota", result["content"])
        result["view"].stop()

    async def test_media_cap_adds_shop_to_existing_progress_message(self):
        status = SimpleNamespace(edit=AsyncMock())
        message = SimpleNamespace(
            id=90701, content="<@999> imagine a cat", author=SimpleNamespace(id=7, bot=False),
            guild=None, channel=SimpleNamespace(id=3), reference=None, mention_everyone=False,
            created_at=datetime.now(timezone.utc), reply=AsyncMock(return_value=status),
        )
        previous = discord_bot.usage_costs.get_request_context()
        self.addCleanup(lambda: discord_bot.usage_costs.set_request_context(**previous))
        reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()), observe_invocation=AsyncMock())
        with ExitStack() as stack:
            patches = [
                patch.object(discord_bot.bot._connection, "user", SimpleNamespace(id=999)),
                patch.object(discord_bot.bot, "get_cog", return_value=self.cog),
                patch.object(discord_bot.bot, "is_owner", new=AsyncMock(return_value=False)),
                patch.object(discord_bot, "_already_processed", return_value=False),
                patch.object(discord_bot, "_get_reflection_worker", return_value=reflection),
                patch.object(discord_bot, "should_suppress_response", new=AsyncMock(return_value=False)),
                patch.object(discord_bot, "resolve_reference_message", new=AsyncMock(return_value=(None, False))),
                patch.object(discord_bot, "_index_message_async", new=AsyncMock()),
                patch.object(discord_bot, "collect_image_inputs", new=AsyncMock(return_value=[])),
                patch.object(discord_bot, "collect_gemini_parts", new=AsyncMock(return_value=[])),
                patch.object(discord_bot, "has_visual_inputs", return_value=False),
                patch.object(discord_bot, "fetch_recent_channel_context", new=AsyncMock(return_value=[])),
                patch.object(discord_bot, "get_user_seen", return_value=None),
                patch.object(discord_bot, "check_rate_limit", side_effect=[RateLimitDecision(True), RateLimitDecision(False, 12, "global")]),
            ]
            for item in patches:
                stack.enter_context(item)
            dispatch = stack.enter_context(patch.object(discord_bot, "dispatch_intent", new=AsyncMock()))
            await discord_bot._builtin_on_message(message)
        dispatch.assert_not_awaited()
        message.reply.assert_awaited_once()
        result = status.edit.call_args.kwargs
        self.assertIn("Media-generation quota", result["content"])
        self.assertIsInstance(result["view"], ThrottleShopView)
        self.assertNotIn(message.id, discord_bot._preflight_status_by_message_id)
        result["view"].stop()
