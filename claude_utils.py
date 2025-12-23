import os
import logging
import anthropic
from typing import List, Dict, Any, Optional

logger = logging.getLogger("discord_bot")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

async def generate_claude_response(
    messages: List[Dict[str, Any]], 
    model: str = "claude-3-5-sonnet-20240620",
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response from Claude.
    
    Args:
        messages: List of message dicts (role, content). 
                  Note: Claude API requires 'user' and 'assistant' roles in specific order. 
                  System prompts are passed separately in the API, so we handle extraction here.
        model: The model string to use.
        tools: List of tool definitions (not yet fully implemented in this minimal wrapper).
    
    Returns:
        The text response content.
    """
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is missing.")
        return "❌ Error: `ANTHROPIC_API_KEY` is not set in the environment."

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # 1. Extract System Prompt(s)
    system_prompt_parts = []
    raw_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            if msg.get("content"):
                system_prompt_parts.append(msg["content"])
        else:
            raw_messages.append(msg)
    
    system_prompt = "\n\n".join(system_prompt_parts).strip()

    # 2. Sanitize Messages (Claude Strict Rules)
    # - No empty content
    # - Alternating User/Assistant roles
    # - Must start with User
    
    sanitized_messages = []
    
    for msg in raw_messages:
        content = (msg.get("content") or "").strip()
        role = msg.get("role")
        
        if not content:
            continue
            
        if not sanitized_messages:
            # First message must be user
            if role == "user":
                sanitized_messages.append({"role": "user", "content": content})
            else:
                # If first is assistant, we skip it OR convert it? 
                # Better to skip to avoid confusion, or prepend a dummy user message?
                # Let's skip leading assistant messages for now.
                pass
        else:
            prev_role = sanitized_messages[-1]["role"]
            if role == prev_role:
                # Merge with previous
                sanitized_messages[-1]["content"] += f"\n\n{content}"
            else:
                sanitized_messages.append({"role": role, "content": content})

    # Fallback if everything was filtered out
    if not sanitized_messages:
        sanitized_messages.append({"role": "user", "content": "Hello."})

    try:
        # 3. Call API
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": sanitized_messages,
            "temperature": temperature,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt

        # TODO: Add tool support when needed. 
        
        response = await client.messages.create(**kwargs)
        
        # 4. Extract Text
        content_block = response.content[0]
        if content_block.type == "text":
            return content_block.text
        else:
            return f"[Non-text response type: {content_block.type}]"

    except anthropic.APIStatusError as e:
        logger.error(f"Claude API Error: {e}")
        return f"❌ Claude API Error: {e.message}"
    except Exception as e:
        logger.exception("Unexpected error calling Claude")
        return f"❌ internal Error: {e}"
