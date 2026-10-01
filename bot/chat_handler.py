import asyncio
import contextlib
import logging

from bot.chat_context import build_chat_context
from bot.draft_verifier import verify_chat_draft
from bot.research_policy import build_fresh_search_query
from bot.response_policy import apply_personality_overrides
from providers.openai_images import DEFAULT_VISION_DETAIL
from providers.gemini_utils import GeminiModerationError, generate_gemini_text
from providers.openai_client import OPENAI_CHAT_MODEL
from providers.openai_utils import OpenAIModerationError, generate_openai_messages_response_with_tools
from services.behavior_registry import invoke_provider

logger = logging.getLogger("discord_bot")

# Model used to answer chat when the OpenAI backend is unavailable (quota/429,
# rate limit, connection). Keeps the bot usable during an OpenAI outage.
GEMINI_FALLBACK_CHAT_MODEL = "gemini-3.8-flash"

_RESEARCH_TOOL_NAMES = {
    "web_search",
    "summarize_url",
    "get_agent_run_status",
    "list_available_tools",
}
_REVERSE_IMAGE_TOOL_NAMES = _RESEARCH_TOOL_NAMES | {"reverse_image_search"}


def _allowed_tools_for_intent(intent: str) -> set[str] | None:
    if intent in {"chat_tiny", "chat_light", "chat_standard", "clarify"}:
        return set()
    if intent == "chat_research":
        return _RESEARCH_TOOL_NAMES
    if intent == "chat_reverse_image":
        return _REVERSE_IMAGE_TOOL_NAMES
    return None


def _tool_call_limits_for_intent(intent: str) -> dict[str, int] | None:
    if intent == "chat_reverse_image":
        return {
            "reverse_image_search": 1,
            "web_search": 2,
            "summarize_url": 2,
        }
    return None


async def _generate_gemini_threaded(*args, **kwargs):
    return await asyncio.to_thread(generate_gemini_text, *args, **kwargs)


def _is_openai_outage(text) -> bool:
    """True when a chat generation returned one of the '⚠️ OpenAI ... error:'
    sentinel strings (see providers/openai_messages.py) instead of a real
    answer — meaning the OpenAI call itself failed rather than the model
    choosing to say something. Triggers the Gemini fallback below."""
    if not isinstance(text, str):
        return False
    stripped = text.lstrip()
    return stripped.startswith("⚠️ OpenAI") and "error:" in stripped


async def handle_chat_intent(
    *,
    message,
    prompt: str,
    raw_prompt: str,
    user_id,
    ref_msg,
    is_reply_to_bot: bool,
    image_urls,
    source_image_urls=None,
    gemini_parts,
    channel_context=None,
    duration_estimate: int,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
    moderation_view_factory,
    default_model=None,
    agent_intent: str = "chat",
    reasoning_effort: str | None = None,
    clarify_hint: bool = False,
    force_web_search: bool = False,
    force_reverse_image_search: bool = False,
    is_owner: bool = False,
):
    async def _do_chat_generation(
        model_name=None,
        *,
        existing_status_msg=None,
        allow_model_escalation: bool = True,
        force_research: bool = False,
    ):
        selected_model = model_name or default_model or OPENAI_CHAT_MODEL
        request_force_web = force_web_search or force_research
        request_force_reverse = force_reverse_image_search
        forced_tool = (
            "reverse_image_search"
            if request_force_reverse
            else ("web_search" if request_force_web else None)
        )
        phase = {
            "detail": (
                "Comparing the attached image against web matches…"
                if request_force_reverse
                else
                "Checking current sources…"
                if request_force_web
                else "Drafting answer…"
            )
        }
        verifier_requested_research = {"value": request_force_web}

        async def _chat_with_es_window():
            task_instructions = None
            if clarify_hint:
                task_instructions = [
                    (
                        "The latest user message is vague on its own ('do that', 'make it "
                        "better'). FIRST try to resolve what they mean from the conversation "
                        "history below — especially anything you just offered or discussed. "
                        "If it's resolvable, just do it. Only if genuinely unresolvable, ask "
                        "ONE short clarifying question."
                    )
                ]
            # build_chat_context makes several synchronous ES queries (window,
            # timeline, salient recall) — keep them off the event loop.
            msgs = await asyncio.to_thread(
                build_chat_context,
                message=message,
                user_id=user_id,
                raw_prompt=raw_prompt,
                ref_msg=ref_msg,
                is_reply_to_bot=is_reply_to_bot,
                task_instructions=task_instructions,
                channel_context_messages=channel_context,
            )
            ctx = {
                "guild_id": message.guild.id if message.guild else "DM",
                "channel_id": message.channel.id,
                "user_id": user_id,
                "image_urls": image_urls or [],
                "source_image_urls": source_image_urls or [],
                "intent": agent_intent,
                "request_text": raw_prompt,
                "is_owner": bool(is_owner),
                "latest_user_text": raw_prompt,
            }
            tool_trace = []

            if "gemini" in selected_model.lower():
                status_res = {"text": ""}
                # to_thread: generate_gemini_text is synchronous/blocking and
                # this runs on the event loop (as a live OpenAI-outage fallback).
                text_resp, artifacts = await invoke_provider(
                    "chat.gemini",
                    _generate_gemini_threaded,
                    prompt=prompt,
                    context=msgs,
                    extra_parts=gemini_parts or None,
                    status_tracker=status_res,
                    enable_code_execution=False,
                    search_ids=ctx,
                    model_name=selected_model,
                    force_web_search=request_force_web,
                )
                draft = text_resp
            else:
                if image_urls:
                    msgs = list(msgs)
                    msgs[-1] = {
                        "role": "user",
                        "content": [{"type": "text", "text": raw_prompt}] + [
                            {
                                "type": "image_url",
                                "image_url": {"url": u, "detail": DEFAULT_VISION_DETAIL},
                            }
                            for u in image_urls
                        ],
                    }

                draft = await invoke_provider(
                    "chat.openai",
                    generate_openai_messages_response_with_tools,
                    msgs,
                    allowed_tool_names=_allowed_tools_for_intent(agent_intent),
                    tool_context=ctx,
                    model=selected_model,
                    reasoning_effort=reasoning_effort,
                    forced_tool=forced_tool,
                    forced_tool_args=(
                        {"image_index": 0, "mode": "all", "max_results": 10}
                        if request_force_reverse
                        else {"q": build_fresh_search_query(prompt)}
                        if request_force_web
                        else None
                    ),
                    tool_trace=tool_trace,
                    tool_call_limits=_tool_call_limits_for_intent(agent_intent),
                )

            if not draft or not draft.strip() or _is_openai_outage(draft):
                return draft

            # The generic prose verifier does not receive raw tool results. For
            # reverse-image work, rewriting a grounded provider answer without
            # that evidence can invent a false "no attachment" or "no search"
            # explanation. Keep the tool-loop answer intact; the user's own
            # personality override is still applied at the presentation layer.
            if request_force_reverse and any(
                item.get("name") == "reverse_image_search" for item in tool_trace
            ):
                logger.info("Preserving reverse-image tool answer without post-draft rewrite")
                return draft

            phase["detail"] = "Reviewing evidence, length, and voice…"
            verdict = await verify_chat_draft(
                user_id=user_id,
                display_name=getattr(message.author, "display_name", None),
                prompt=raw_prompt,
                draft=draft,
                research_used=(
                    request_force_web
                    or request_force_reverse
                    or any(item.get("name") in {"web_search", "reverse_image_search"} for item in tool_trace)
                ),
            )
            logger.info(
                "Post-draft verdict=%s research_used=%s",
                verdict.action,
                request_force_web
                or request_force_reverse
                or any(item.get("name") in {"web_search", "reverse_image_search"} for item in tool_trace),
            )
            if verdict.action == "accept":
                return draft
            if verdict.action == "revise":
                return verdict.revised_answer or draft

            verifier_requested_research["value"] = True
            phase["detail"] = "Checking stronger current evidence…"
            search_query = build_fresh_search_query(
                verdict.research_query or prompt
            )
            repair_msgs = list(msgs)
            repair_msgs.extend(
                [
                    {"role": "assistant", "content": draft},
                    {
                        "role": "system",
                        "content": (
                            "A post-draft verifier found that the answer needs stronger "
                            "fresh evidence. Discard unsupported or stale claims, search "
                            "again, and return one corrected final answer. Preserve an "
                            "appropriate length and the user's established style. Start "
                            f"with this research query: {search_query}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Return the corrected, source-grounded answer only.",
                    },
                ]
            )
            if "gemini" in selected_model.lower():
                repaired, _artifacts = await invoke_provider(
                    "chat.gemini",
                    _generate_gemini_threaded,
                    prompt="Return the corrected, source-grounded answer only.",
                    context=repair_msgs,
                    extra_parts=gemini_parts or None,
                    status_tracker={"text": ""},
                    enable_code_execution=False,
                    search_ids=ctx,
                    model_name=selected_model,
                    force_web_search=True,
                )
                return repaired or draft

            repaired = await invoke_provider(
                "chat.openai",
                generate_openai_messages_response_with_tools,
                repair_msgs,
                allowed_tool_names=_RESEARCH_TOOL_NAMES,
                tool_context=ctx,
                model=selected_model,
                reasoning_effort=reasoning_effort,
                forced_tool="web_search",
                forced_tool_args={"q": search_query},
            )
            return repaired or draft

        def _summarizer():
            detail = f"• Using {selected_model}…\n• {phase['detail']}"
            if tool_trace:
                latest = tool_trace[-1]
                status = latest.get("status") or "running"
                detail += f"\n• Tool {latest.get('name', 'unknown')}: {status}"
            return detail

        try:
            status_kwargs = {
                "action_label": f"Responding ({selected_model})",
                "emoji": "💬",
                "coro": _chat_with_es_window(),
                "duration_estimate": duration_estimate,
                "summarizer": _summarizer if stream_ok else None,
            }
            if existing_status_msg is not None:
                status_kwargs["existing_status_msg"] = existing_status_msg
            status_msg, response = await live_status_with_progress(message, **status_kwargs)

            if _is_openai_outage(response) and "gemini" not in selected_model.lower():
                # OpenAI backend is down (quota/429/etc). Don't surface the raw
                # error or bounce between OpenAI tiers — answer with Gemini.
                logger.warning(
                    "OpenAI unavailable during chat (%.80r); falling back to Gemini",
                    response,
                )
                with contextlib.suppress(Exception):
                    await status_msg.edit(
                        content="⚠️ OpenAI is unavailable right now — answering with **Gemini** instead…"
                    )
                await _do_chat_generation(
                    model_name=GEMINI_FALLBACK_CHAT_MODEL,
                    existing_status_msg=status_msg,
                    allow_model_escalation=False,
                    force_research=verifier_requested_research["value"],
                )
            elif response and response.strip():
                response = apply_personality_overrides(user_id, intent="chat", text=response)
                await send_or_edit_with_truncation(
                    response,
                    target_msg=status_msg,
                    original_message=message,
                    model=selected_model,
                )
            elif selected_model != OPENAI_CHAT_MODEL and allow_model_escalation:
                # Light-tier model punted; escalate to the main model once.
                logger.info("Empty response from %s; escalating to %s", selected_model, OPENAI_CHAT_MODEL)
                await _do_chat_generation(
                    model_name=OPENAI_CHAT_MODEL,
                    existing_status_msg=status_msg,
                    allow_model_escalation=False,
                )
            else:
                await status_msg.edit(content="❌ The fallback model returned no response. Please try again.")

        except OpenAIModerationError as e:
            logger.warning(f"OpenAI moderation hit: {e}")
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_chat_generation)
            await message.reply(user_msg, view=view)
        except GeminiModerationError as e:
            logger.warning(f"Gemini moderation hit in Chat fallback: {e}")
            user_msg = f"⚠️ **Response Blocked by Safety Filters** (Reason: {str(e)})\nSelect a different model to retry:"
            view = moderation_view_factory(author_id=message.author.id, retry_callback=_do_chat_generation)
            await message.reply(user_msg, view=view)

    await _do_chat_generation()
