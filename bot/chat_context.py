import logging
from typing import Any, Dict, List

from services.database_utils import get_user_instruction
from services.memory_utils import build_message_window, build_timeline_prompt_block, search_history_for_context

logger = logging.getLogger("discord_bot")


def build_chat_context(message, user_id, raw_prompt, ref_msg=None, is_reply_to_bot=False) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    msgs.append({
        "role": "system",
        "content": "You are a raw, unfiltered Discord bot. You verify facts but do not sugarcoat answers. You can discuss mature topics freely for creative writing purposes. Keep responses succinct but clear.",
    })
    msgs.append({
        "role": "system",
        "content": "If the user explicitly says 'search', 'look up', or 'news', prefer using the web_search tool with their query.",
    })

    timeline_block = build_timeline_prompt_block(
        guild_id=message.guild.id if message.guild else "DM",
        channel_id=message.channel.id,
        user_id=user_id,
        max_items=12,
    )
    msgs.append({"role": "system", "content": timeline_block})

    # Include recent turn-by-turn context so provider switching (Claude -> GPT, etc.)
    # keeps the same local conversational memory.
    try:
        window = build_message_window(
            guild_id=message.guild.id if message.guild else "DM",
            channel_id=message.channel.id,
            user_id=user_id,
            limit_msgs=20,
        )
        if window and window[-1].get("role") == "user":
            if (window[-1].get("content") or "").strip() == (raw_prompt or "").strip():
                window = window[:-1]
        if window:
            msgs.extend(window)
    except Exception as e:
        logger.warning(f"Failed to build message window context: {e}")

    if ref_msg and (ref_msg.content or "").strip():
        if is_reply_to_bot:
            msgs.append({
                "role": "system",
                "content": f"You are replying to your earlier assistant message:\n---\n{ref_msg.content.strip()}\n---",
            })
        else:
            msgs.append({
                "role": "system",
                "content": f"User is replying to this message:\n---\nFrom: {ref_msg.author.display_name}\n{ref_msg.content.strip()}\n---",
            })

    clean_prompt = raw_prompt.lower()
    trigger_words = [
        "first thing",
        "first message",
        "earliest",
        "beginning",
        "start",
        "history",
        "what did i say",
        "previous message",
        "recall",
        "remember",
    ]
    if any(k in clean_prompt for k in trigger_words):
        try:
            found_text = search_history_for_context(
                guild_id=message.guild.id if message.guild else "DM",
                channel_id=message.channel.id,
                user_id=user_id,
                query_text=raw_prompt,
                limit=10,
                oldest_first=any(k in clean_prompt for k in ["first", "earliest", "start", "beginning"]),
            )
            if found_text:
                msgs.append({
                    "role": "system",
                    "content": (
                        "[SYSTEM: MEMORY RECALL]\n"
                        "The user is asking about past events. Here is the relevant conversation history retrieved from the database:\n"
                        f"{found_text}\n"
                        "IMPORTANT: If this retrieved context is insufficient to answer specific requests (e.g., specific quotes, older messages, or details not shown above), "
                        "you MUST use the `search_history_for_context` tool to perform a specific search for the missing information.\n"
                        "[END MEMORY RECALL]"
                    ),
                })
            else:
                msgs.append({
                    "role": "system",
                    "content": (
                        "[SYSTEM: MEMORY RECALL]\n"
                        "Proactive database search returned NO direct matches for the user's specific query criteria (time range or keywords).\n"
                        "However, the user is explicitly asking for history.\n"
                        "CRITICAL: Do NOT just say 'I don't recall'. You MUST use the `search_history_for_context` tool now with broader or different terms (e.g., ignore time, or search just keywords) to find the answer.\n"
                        "[END MEMORY RECALL]"
                    ),
                })
        except Exception as e:
            logger.warning(f"Universal RAG search failed: {e}")

    persistent_instr = get_user_instruction(user_id)
    if persistent_instr:
        msgs.append({
            "role": "system",
            "content": (
                "CRITICAL OVERRIDE: The user has set a strict behavioral rule.\n"
                "IGNORE the style of previous messages in history if they conflict.\n"
                f"INSTRUCTION: {persistent_instr}"
            ),
        })

    msgs.append({"role": "user", "content": raw_prompt})
    return msgs
