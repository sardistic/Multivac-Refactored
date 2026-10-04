import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import discord_bot
from bot import intent_dispatcher, response_policy


class SilencePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_commands_do_not_depend_on_the_classifier(self):
        with patch(
            "providers.openai_utils.classify_silence_request",
            new=AsyncMock(side_effect=RuntimeError("classifier unavailable")),
        ) as classify:
            for text in (
                "shutdown", "SHUT DOWN!", "please shutdown.",
                "stop responding", "stop replying please", "don't reply",
                "don’t respond", "do not reply",
            ):
                with self.subTest(text=text):
                    self.assertTrue(await response_policy.should_suppress_response(text))
            classify.assert_not_awaited()

    async def test_other_shutdown_language_still_uses_the_classifier(self):
        with patch(
            "providers.openai_utils.classify_silence_request",
            new=AsyncMock(return_value=False),
        ) as classify:
            for text in (
                "How does shutdown work?", "shutdown the server",
                'Explain "shutdown"', "don't shut down", "don't stop responding",
                "change your code to implement a shutdown gate", "hello",
            ):
                with self.subTest(text=text):
                    self.assertFalse(await response_policy.should_suppress_response(text))
                    classify.assert_awaited_with(text)


class DiscordSilenceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_mentioned_shutdown_is_silent_before_quota_and_progress(self):
        message = SimpleNamespace(
            id=101, content="<@999> shutdown", author=SimpleNamespace(id=7, bot=False),
            guild=None, channel=SimpleNamespace(id=3, send=AsyncMock()),
            reference=None, mention_everyone=False, reply=AsyncMock(),
        )
        reflection = SimpleNamespace(
            observe_message=AsyncMock(return_value=set()),
            observe_invocation=AsyncMock(),
        )
        with (
            patch.object(discord_bot.bot._connection, "user", SimpleNamespace(id=999)),
            patch.object(discord_bot, "_already_processed", return_value=False),
            patch.object(discord_bot, "_get_reflection_worker", return_value=reflection),
            patch.object(discord_bot.bot, "is_owner", new=AsyncMock()) as owner,
            patch.object(discord_bot, "check_rate_limit") as quota,
            patch.object(discord_bot, "classify_intent", new=AsyncMock()) as intent,
            patch.object(discord_bot, "dispatch_intent", new=AsyncMock()) as dispatch,
            patch("providers.openai_utils.classify_silence_request", new=AsyncMock()) as classify,
        ):
            await discord_bot._builtin_on_message(message)

        message.reply.assert_not_awaited()
        message.channel.send.assert_not_awaited()
        owner.assert_not_awaited()
        quota.assert_not_called()
        intent.assert_not_awaited()
        dispatch.assert_not_awaited()
        classify.assert_not_awaited()
        reflection.observe_invocation.assert_not_awaited()
        self.assertNotIn(message.id, discord_bot._preflight_status_by_message_id)

    async def test_model_silence_on_reply_to_bot_precedes_all_output(self):
        message = SimpleNamespace(
            id=102, content="I'd rather you kept quiet for this one",
            author=SimpleNamespace(id=7, bot=False), guild=None,
            channel=SimpleNamespace(id=3, send=AsyncMock()),
            reference=None, mention_everyone=False, reply=AsyncMock(),
        )
        reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()))
        with (
            patch.object(discord_bot.bot._connection, "user", SimpleNamespace(id=999)),
            patch.object(discord_bot, "_already_processed", return_value=False),
            patch.object(discord_bot, "_get_reflection_worker", return_value=reflection),
            patch.object(discord_bot, "resolve_reference_message", new=AsyncMock(return_value=(None, True))),
            patch("providers.openai_utils.classify_silence_request", new=AsyncMock(return_value=True)),
            patch.object(discord_bot, "check_rate_limit") as quota,
        ):
            await discord_bot._builtin_on_message(message)
        message.reply.assert_not_awaited()
        message.channel.send.assert_not_awaited()
        quota.assert_not_called()

    async def test_ordinary_message_continues_to_quota_handling(self):
        message = SimpleNamespace(
            id=103, content="<@999> how does server shutdown work?",
            author=SimpleNamespace(id=7, bot=False), guild=None,
            channel=SimpleNamespace(id=3, send=AsyncMock()),
            reference=None, mention_everyone=False, reply=AsyncMock(),
        )
        reflection = SimpleNamespace(observe_message=AsyncMock(return_value=set()))
        with (
            patch.object(discord_bot.bot._connection, "user", SimpleNamespace(id=999)),
            patch.object(discord_bot, "_already_processed", return_value=False),
            patch.object(discord_bot, "_get_reflection_worker", return_value=reflection),
            patch.object(discord_bot.bot, "is_owner", new=AsyncMock(return_value=False)),
            patch.object(discord_bot, "check_rate_limit", return_value=SimpleNamespace(allowed=False, retry_after=2)),
            patch("providers.openai_utils.classify_silence_request", new=AsyncMock(return_value=False)) as classify,
        ):
            await discord_bot._builtin_on_message(message)
        classify.assert_awaited_once_with("how does server shutdown work?")
        message.reply.assert_awaited_once()


class DispatchSilenceGateTests(unittest.IsolatedAsyncioTestCase):
    def context(self, **kwargs):
        return intent_dispatcher.DispatchContext(
            intent="chat", message=Mock(), prompt="shutdown", raw_prompt="shutdown",
            user_id=7, bot_user=SimpleNamespace(id=999), **kwargs,
        )

    async def test_direct_shutdown_dispatch_is_handled_without_any_handler(self):
        with (
            patch.object(intent_dispatcher, "dispatch_intent_override", new=AsyncMock()) as override,
            patch.object(intent_dispatcher, "_dispatch_builtin_intent", new=AsyncMock()) as builtin,
        ):
            self.assertTrue(await intent_dispatcher.dispatch_intent(self.context()))
        override.assert_not_awaited()
        builtin.assert_not_awaited()

    async def test_ingress_verdict_is_not_classified_again(self):
        context = self.context(response_gate_checked=True)
        context.prompt = context.raw_prompt = "hello"
        with (
            patch.object(intent_dispatcher, "should_suppress_response", new=AsyncMock()) as gate,
            patch.object(intent_dispatcher, "dispatch_intent_override", new=AsyncMock(return_value=(False, None))),
            patch.object(intent_dispatcher, "_dispatch_builtin_intent", new=AsyncMock(return_value=True)) as builtin,
        ):
            self.assertTrue(await intent_dispatcher.dispatch_intent(context))
        gate.assert_not_awaited()
        builtin.assert_awaited_once_with(context)
