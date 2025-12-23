import os
import logging
import anthropic
from typing import List, Dict, Any, Optional

logger = logging.getLogger("discord_bot")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

async def generate_claude_response(
    messages: List[Dict[str, Any]], 
    model: str = "claude-3-5-sonnet-latest",
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
    # OpenAI/Gemini logic often puts system prompts in the messages list. 
    # Claude requires them as a top-level `system` parameter.
    system_prompt = ""
    filtered_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompt += msg["content"] + "\n\n"
        else:
            filtered_messages.append(msg)
    
    system_prompt = system_prompt.strip()

    try:
        # 2. Call API
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": filtered_messages,
            "temperature": temperature,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt

        # TODO: Add tool support when needed. 
        # For now, we are text-only to match the basic integration plan.
        
        response = await client.messages.create(**kwargs)
        
        # 3. Extract Text
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
