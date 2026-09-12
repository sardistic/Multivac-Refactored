import asyncio
import os
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.intent_dispatcher import (
    DispatchContext,
    _dispatch_builtin_intent,
    chat_model_for_intent,
    get_duration_estimate,
    resolve_keyword_intent,
    validate_classified_intent,
    wants_gemini,
)
from bot.research_policy import build_fresh_search_query, requires_fresh_web


class KeywordRoutingTests(unittest.TestCase):
    """Keywords pick the provider; the LLM classifier picks the intent.
    resolve_keyword_intent only shortcuts explicit 'claude ...' messages."""

    def test_claude_prefix_shortcuts_to_claude_chat(self):
        self.assertEqual(
            resolve_keyword_intent("claude what is entropy", "claude what is entropy", False),
            "claude_chat",
        )
        self.assertEqual(
            resolve_keyword_intent("<@123> something", "Claude something", False),
            "claude_chat",
        )

    def test_imagine_is_a_deterministic_image_command(self):
        cases = [
            "imagine a castle at dusk",
            "imagine a cool hackerman yelling at a chatbot",
            "gemini imagine a liminal rose",
            "IMAGINE: neon rain",
        ]
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(resolve_keyword_intent(prompt, prompt, False), "generate_image")

    def test_imagine_translate_describe_with_image_defers_to_classifier(self):
        # "imagine a translation of this image" with an image present is a
        # describe task, not generation — must fall through to the classifier.
        describe_cases = [
            "imagine a translation of this image to english",
            "imagine what this image says",
            "imagine a description of this picture",
            "gemini imagine a translation of this to english",
        ]
        for prompt in describe_cases:
            with self.subTest(prompt=prompt):
                self.assertIsNone(resolve_keyword_intent(prompt, prompt, True))
        # …but the same words with NO image still generate (nothing to describe),
        # and a real scene request with an image still generates.
        self.assertEqual(
            resolve_keyword_intent("imagine a translation of this image", "imagine a translation of this image", False),
            "generate_image",
        )
        self.assertEqual(
            resolve_keyword_intent("imagine a dragon on a cliff", "imagine a dragon on a cliff", True),
            "generate_image",
        )

    def test_explicit_video_generation_is_deterministic(self):
        cases = [
            ("gemini make this into a video, he laughs smugly", True),
            ("generate a video of a cat", False),
            ("create me a short cinematic clip", False),
            ("animate this", True),
            ("turn the image into an animation", True),
            ("image-to-video: rain starts falling", True),
        ]
        for prompt, has_attachments in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    resolve_keyword_intent(prompt, prompt, has_attachments),
                    "generate_video",
                )

    def test_non_generation_requests_still_defer_to_classifier(self):
        cases = [
            ("gemini what is entropy", False),
            ("gemini generate an image of a liminal rose", False),
            ("gemini edit this image", True),
            ("generate imagine of liminal rose", False),
            ("make me a picture of a dog", False),
            ("summarize this video", True),
            ("what is happening in this video?", True),
            ("what's the weather", False),
            ("imagines are weird", False),  # not the command word
            ("", False),
        ]
        for prompt, has_attachments in cases:
            with self.subTest(prompt=prompt):
                self.assertIsNone(resolve_keyword_intent(prompt, prompt, has_attachments))

    def test_natural_code_change_requests_route_to_audited_pipeline(self):
        cases = [
            "/code_propose Add a sentence to your readme",
            "change your code so responses are more compact",
            "claude edit the codebase so responses are more compact",
            "fable refactor the repository to simplify routing",
            "modify the bot's routing to recognize this intent",
            "add a command to your implementation for diagnostics",
        ]
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertEqual(resolve_keyword_intent(prompt, prompt, False), "code_change")

    def test_natural_code_control_routes(self):
        cases = {
            "approve this code change": "code_approve",
            "reject proposal 4": "code_reject",
            "status of the code change": "code_status",
            "roll back that deployment": "code_rollback",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(resolve_keyword_intent(prompt, prompt, False), expected)

    def test_code_questions_remain_chat_classifier_work(self):
        for prompt in ["how does your code work?", "show me your routing code"]:
            with self.subTest(prompt=prompt):
                self.assertIsNone(resolve_keyword_intent(prompt, prompt, False))


class ProviderSelectionTests(unittest.TestCase):
    def test_wants_gemini(self):
        positive = [
            "gemini imagine a castle",
            "Gemini generate an image of a rose",
            "generate an image of a rose with gemini",
            "make a picture using gemini",
        ]
        negative = [
            "imagine a castle",
            "generate an image of the gemini zodiac sign",
            "draw the gemini constellation",
            "",
        ]
        for p in positive:
            with self.subTest(p=p):
                self.assertTrue(wants_gemini(p))
        for p in negative:
            with self.subTest(p=p):
                self.assertFalse(wants_gemini(p))


class ClassifiedIntentGuardTests(unittest.TestCase):
    def test_false_code_status_guesses_return_to_chat(self):
        for prompt in (
            "are you editing the code in 5.6",
            "why did you just do that",
            "what do you think about your code?",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(validate_classified_intent("code_status", prompt), "chat")

    def test_real_status_questions_remain_code_status(self):
        for prompt in (
            "did it deploy?",
            "what is the proposal status?",
            "is maple-saffron active?",
            "did that change take?",
            "what happened to the patch?",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    validate_classified_intent("code_status", prompt),
                    "code_status",
                )

    def test_other_classifier_intents_are_untouched(self):
        self.assertEqual(validate_classified_intent("chat_light", "hello"), "chat_light")
        self.assertEqual(validate_classified_intent("code_change", "change your code"), "code_change")

    def test_false_weather_guesses_return_to_chat(self):
        for prompt in (
            "how long till the new state of the union",
            "when is the next election",
            "how long until christmas",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(validate_classified_intent("get_weather", prompt), "chat")

    def test_real_weather_questions_remain_get_weather(self):
        for prompt in (
            "weather in raleigh",
            "what's the forecast for tomorrow",
            "will it rain tonight in 27601",
            "how hot is it outside",
            "is it chilly out",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    validate_classified_intent("get_weather", prompt),
                    "get_weather",
                )

    def test_live_fact_misses_are_promoted_to_research(self):
        cases = (
            "who won the world cup",
            "who won the super bowl",
            "who is the president of France",
            "what is the latest Python release",
            "what's the score of the game",
            "look up whether that product is available",
        )
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertTrue(requires_fresh_web(prompt))
                self.assertEqual(
                    validate_classified_intent("chat_light", prompt),
                    "chat_research",
                )

    def test_historical_and_timeless_facts_stay_on_normal_route(self):
        cases = (
            "who won the 2018 world cup",
            "who won world war two",
            "what is price elasticity",
            "who was the first president of France",
        )
        for prompt in cases:
            with self.subTest(prompt=prompt):
                self.assertFalse(requires_fresh_web(prompt))
                self.assertEqual(
                    validate_classified_intent("chat_light", prompt),
                    "chat_light",
                )

    def test_research_route_uses_full_chat_model_and_search_duration(self):
        from providers.openai_client import OPENAI_CHAT_MODEL

        self.assertEqual(chat_model_for_intent("chat_research"), OPENAI_CHAT_MODEL)
        self.assertEqual(get_duration_estimate("chat_research"), 8)

    def test_fresh_search_query_disambiguates_recurring_result(self):
        self.assertEqual(
            build_fresh_search_query(
                "who won the last world cup",
                today=date(2026, 7, 29),
            ),
            "who won the last world cup 2026 final result winner",
        )

    def test_fresh_search_query_preserves_explicit_historical_year(self):
        self.assertEqual(
            build_fresh_search_query(
                "who won the 2018 world cup",
                today=date(2026, 7, 29),
            ),
            "who won the 2018 world cup verified result",
        )


class ResearchDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_research_intent_forces_search_in_chat_handler(self):
        context = DispatchContext(
            intent="chat_research",
            message=SimpleNamespace(),
            prompt="who won the world cup",
            raw_prompt="who won the world cup",
            user_id=123,
            bot_user=SimpleNamespace(id=456),
        )
        with patch(
            "bot.intent_dispatcher.handle_chat_intent",
            new_callable=AsyncMock,
        ) as handle:
            result = await _dispatch_builtin_intent(context)

        self.assertTrue(result)
        self.assertTrue(handle.await_args.kwargs["force_web_search"])

    def test_forced_gemini_research_does_not_depend_on_keywords(self):
        from providers.gemini_text import _should_enable_google_search

        self.assertFalse(_should_enable_google_search("who won the world cup"))
        self.assertTrue(
            _should_enable_google_search(
                "who won the world cup",
                force_web_search=True,
            )
        )


class ClassifierOutageFallbackTests(unittest.TestCase):
    """When the OpenAI classifier can't run, routing must degrade gracefully:
    an explicit 'gemini ...' request still reaches Gemini instead of bouncing
    to 'chat' (which would hit the same dead OpenAI backend)."""

    def test_keyword_fallback_routes_gemini_prefix(self):
        from providers.openai_intents import _keyword_fallback_intent

        self.assertEqual(_keyword_fallback_intent("gemini tell me a joke"), "gemini_chat")
        self.assertEqual(_keyword_fallback_intent("  GEMINI what's up"), "gemini_chat")

    def test_keyword_fallback_prioritizes_video_operation(self):
        from providers.openai_intents import _keyword_fallback_intent

        self.assertEqual(
            _keyword_fallback_intent('gemini make this into a video, he says "just watch the youtube"'),
            "generate_video",
        )
        self.assertEqual(_keyword_fallback_intent("animate this"), "generate_video")
        self.assertEqual(_keyword_fallback_intent("summarize this video"), "chat")

    def test_keyword_fallback_acts_on_an_attached_image(self):
        """An outage must not turn 'edit this' into an apology. With a picture
        present the fallback routes to the image handlers, which have their own
        Gemini path when OpenAI is the thing that is down."""
        from providers.openai_intents import _keyword_fallback_intent

        for text in (
            "edit this image so they are standing on him",
            "edit this so they are each standing on him with one foot",
            "remove the text",
            "make it darker",
            "gemini add a cat to this",
        ):
            with self.subTest(text=text):
                self.assertEqual(_keyword_fallback_intent(text, has_images=True), "edit_image")

        self.assertEqual(
            _keyword_fallback_intent("what does this say", has_images=True), "describe_image"
        )
        self.assertEqual(
            _keyword_fallback_intent("translate this", has_images=True), "describe_image"
        )
        self.assertEqual(
            _keyword_fallback_intent("draw this as a woodcut", has_images=True), "generate_image"
        )
        # Commentary asks for nothing, and an operation word still beats the
        # picture when it names a different medium.
        self.assertEqual(_keyword_fallback_intent("kino", has_images=True), "chat")
        self.assertEqual(
            _keyword_fallback_intent("animate this", has_images=True), "generate_video"
        )

    def test_keyword_fallback_ignores_image_verbs_without_an_image(self):
        from providers.openai_intents import _keyword_fallback_intent

        self.assertEqual(_keyword_fallback_intent("make it darker"), "chat")
        self.assertEqual(_keyword_fallback_intent("what does this say"), "chat")

    def test_classify_intent_fallback_receives_the_image_flag(self):
        """The flag has to reach the keyword router; classify_intent used to
        drop it there, so every outage-time image request became 'chat'."""
        import providers.openai_intents as oi

        with patch.object(oi, "get_openai_client", side_effect=RuntimeError("insufficient_quota")), patch.object(
            oi, "_classify_intent_with_sonnet", new=AsyncMock(side_effect=RuntimeError("400"))
        ):
            self.assertEqual(
                asyncio.run(oi.classify_intent("edit this image", has_images=True)),
                "edit_image",
            )
            self.assertEqual(
                asyncio.run(oi.classify_intent("edit this image", has_images=False)),
                "chat",
            )

    def test_sonnet_classifier_sends_no_temperature(self):
        """Claude 5 rejects `temperature` outright (HTTP 400), which broke this
        fallback on every call."""
        import inspect

        import providers.openai_intents as oi

        source = inspect.getsource(oi._classify_intent_with_sonnet)
        self.assertNotIn("temperature=", source)

    def test_keyword_fallback_defaults_to_chat(self):
        from providers.openai_intents import _keyword_fallback_intent

        for text in ("what's the weather", "tell me about entropy", "", "claude hi"):
            with self.subTest(text=text):
                self.assertEqual(_keyword_fallback_intent(text), "chat")

    def test_classify_intent_falls_back_when_openai_down(self):
        import providers.openai_intents as oi

        def _boom():
            raise RuntimeError("insufficient_quota")

        original = oi.get_openai_client
        oi.get_openai_client = _boom
        try:
            self.assertEqual(asyncio.run(oi.classify_intent("gemini lose weight tips")), "gemini_chat")
            self.assertEqual(asyncio.run(oi.classify_intent("how do magnets work")), "chat")
        finally:
            oi.get_openai_client = original

    def test_classify_intent_uses_sonnet_before_keyword_fallback(self):
        import providers.openai_intents as oi

        with patch.object(oi, "get_openai_client", side_effect=RuntimeError("insufficient_quota")), patch.object(
            oi,
            "_classify_intent_with_sonnet",
            new=AsyncMock(return_value="generate_image"),
        ) as sonnet:
            result = asyncio.run(
                oi.classify_intent(
                    "fallback to gemini picture generation",
                    prev_intent="generate_image",
                )
            )

        self.assertEqual(result, "generate_image")
        sonnet.assert_awaited_once()


class EfficientClassifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_luna_router_disables_reasoning_and_caps_output(self):
        import providers.openai_intents as oi

        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="chat_light"))],
            usage=None,
            model="gpt-5.6-luna",
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.object(oi, "get_openai_client", return_value=client), patch(
            "services.usage_costs.record_response"
        ):
            result = await oi.classify_intent(
                "what is two plus two?",
                recent_turns=["user: old", "assistant: newer", "user: newest"],
            )

        self.assertEqual(result, "chat_light")
        kwargs = create.await_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.6-luna")
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["max_completion_tokens"], 64)
        self.assertEqual(kwargs["messages"][0]["role"], "developer")
        classifier_prompt = kwargs["messages"][0]["content"]
        self.assertIn("cheapest sufficient tier", classifier_prompt)
        self.assertIn("'chat_research'", classifier_prompt)
        user_prompt = kwargs["messages"][1]["content"]
        self.assertIn("CURRENT DATE (UTC):", user_prompt)
        self.assertNotIn("user: old", user_prompt)
        self.assertIn("assistant: newer", user_prompt)
        self.assertIn("user: newest", user_prompt)


class ChatOutageDetectionTests(unittest.TestCase):
    """The Gemini chat fallback fires on OpenAI's error sentinel strings and
    nothing else."""

    def test_detects_openai_error_sentinels(self):
        from bot.chat_handler import _is_openai_outage

        self.assertTrue(_is_openai_outage("⚠️ OpenAI tools error: Error code: 429 - quota"))
        self.assertTrue(_is_openai_outage("⚠️ OpenAI error: connection reset"))

    def test_ignores_normal_answers(self):
        from bot.chat_handler import _is_openai_outage

        for text in ("Here's how magnets work…", "", None, "⚠️ Response Blocked by Safety Filters"):
            with self.subTest(text=text):
                self.assertFalse(_is_openai_outage(text))


@unittest.skipUnless(
    os.getenv("RUN_LIVE_INTENT_TESTS") == "1",
    "live classifier test; set RUN_LIVE_INTENT_TESTS=1 to run",
)
class LiveReplyToImageClassifierTests(unittest.TestCase):
    """Replies to a just-generated image must split by what they ask for:
    reactions are commentary, change requests are edits, 'another one' is a
    fresh generation. Hits the real classifier model, so opt-in only."""

    RECENT_TURNS = [
        "user: imagine a retro movie theater marquee at night",
        "bot: ✅ Image generated (GPT Image 2.5 Sunburst) [image attached]",
    ]

    def _classify(self, text: str) -> str:
        from providers.openai_intents import classify_intent

        return asyncio.run(
            classify_intent(
                text,
                recent_turns=self.RECENT_TURNS,
                prev_intent="generate_image",
            )
        )

    def test_bare_reaction_is_chat_tiny(self):
        for reaction in ("kino", "nice", "based", "lol"):
            with self.subTest(reaction=reaction):
                self.assertEqual(self._classify(reaction), "chat_tiny")

    def test_change_request_is_edit_image(self):
        for prompt in ("make it darker", "remove the text on the marquee"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._classify(prompt), "edit_image")

    def test_another_one_is_generate_image(self):
        for prompt in ("another one", "same but at sunrise"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._classify(prompt), "generate_image")


@unittest.skipUnless(
    os.getenv("RUN_LIVE_INTENT_TESTS") == "1",
    "live classifier test; set RUN_LIVE_INTENT_TESTS=1 to run",
)
class LiveWriteVsDepictTests(unittest.TestCase):
    """Asking to WRITE text about a visual/horror subject must stay text, even
    when the text-intent word is garbled ('fan function' for 'fanfiction').
    Regression for 'one sentence doki doki fan function horror' -> image."""

    def _classify(self, text: str) -> str:
        from providers.openai_intents import classify_intent

        return asyncio.run(classify_intent(text))

    def test_write_requests_stay_text(self):
        for prompt in (
            "one sentence doki doki fan function horror",
            "write one sentence of doki doki horror",
            "a haiku about a burning city",
            "give me a caption for a spooky doki doki scene",
        ):
            with self.subTest(prompt=prompt):
                self.assertIn(
                    self._classify(prompt),
                    {"chat_light", "chat_standard", "chat"},
                )

    def test_real_image_requests_still_route_to_image(self):
        for prompt in (
            "a moody picture of rain on neon streets",
            "draw a doki doki character in horror style",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._classify(prompt), "generate_image")


@unittest.skipUnless(
    os.getenv("RUN_LIVE_INTENT_TESTS") == "1",
    "live classifier test; set RUN_LIVE_INTENT_TESTS=1 to run",
)
class LiveCodebaseQuestionTests(unittest.TestCase):
    """Questions about the bot's own code/commands need the code-search tool +
    reasoning, so they must route to 'chat' (full model), not 'chat_light'
    (gpt-5.4-mini, which whiffed on the search and gave up)."""

    def _classify(self, text: str) -> str:
        from providers.openai_intents import classify_intent

        return asyncio.run(classify_intent(text))

    def test_codebase_questions_route_to_full_chat(self):
        for prompt in (
            "can you list the /commands added from the codebase",
            "what commands do you have",
            "how are you built",
            "whats your latest commit",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._classify(prompt), "chat")

    def test_trivial_one_liners_use_tiny_and_brief_fact_uses_light(self):
        for prompt in ("lol", "whats up"):
            with self.subTest(prompt=prompt):
                self.assertEqual(self._classify(prompt), "chat_tiny")
        self.assertEqual(self._classify("what is a liminal space"), "chat_light")


if __name__ == "__main__":
    unittest.main()
