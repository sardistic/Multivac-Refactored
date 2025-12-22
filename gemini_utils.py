import logging
import base64
import json
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from config import GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logging.warning("google-genai SDK not found. Install it to use Gemini features.")

logger = logging.getLogger("gemini_utils")

def _get_client():
    if not SDK_AVAILABLE or not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

def generate_gemini_image(prompt: str, width: int = 1024, height: int = 1024) -> Optional[BytesIO]:
    """
    Generate an image using Google Gemini (Imagen 3) via the SDK.
    """
    client = _get_client()
    if not client:
        return None

    # Supported aspect ratios for Imagen: "1:1", "3:4", "4:3", "9:16", "16:9"
    aspect_ratio = "1:1"
    if width > height:
        aspect_ratio = "16:9"
    elif height > width:
        aspect_ratio = "9:16"

    try:
        # User requested: gemini-2.5-flash-image
        # And used client.models.generate_content in their example.
        model = "gemini-3-pro-image-preview"
        
        logger.info(f"Generating image with model: {model}")
        
        # Using the config structure from the user's snippet
        # to ensure proper image generation triggers.
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], # We primarily want image
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="1024x1024" # Equivalent to "1K"? SDK might accept enum or string
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=config
        )
        
        # Parse response for image
        if response.parts:
            for part in response.parts:
                # Check for inline data (common for generated images in Gemini models)
                if part.inline_data:
                    buf = BytesIO(part.inline_data.data)
                    return buf
                
                # Check for 'as_image()' method (SDK helper)
                if hasattr(part, "as_image"):
                    try:
                        img = part.as_image()
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        buf.seek(0)
                        return buf
                    except Exception:
                        pass
                        
        logger.warning(f"Response returned no image parts: {response}")

    except Exception as e:
        logger.exception(f"Gemini generation failed (model={model}): {e}")

    return None

def edit_gemini_image(image_bytes: BytesIO, prompt: str) -> Optional[BytesIO]:
    """
    Edit an image using Gemini.
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Load bytes into PIL Image
        input_image = PILImage.open(image_bytes)

        # Gemini 3 Pro Image Preview
        model = "gemini-3-pro-image-preview" 
        
        # Use config to enforce image output
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ]
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt, input_image],
            config=config
        )

        if not response.parts:
            logger.warning(f"Gemini edit returned no parts. Response: {response}")
            return None

        for part in response.parts:
            if part.inline_data: 
                return BytesIO(part.inline_data.data)
                
            try:
                if hasattr(part, "as_image"):
                    img = part.as_image()
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
            except:
                pass

    except Exception as e:
        logger.exception(f"Gemini edit failed: {e}")

    return None

def generate_gemini_with_references(prompt: str, reference_images: list[BytesIO]) -> Optional[BytesIO]:
    """
    Generate an image using a text prompt and multiple reference images.
    """
    client = _get_client()
    if not client:
        return None

    try:
        # Load all bytes into PIL Images
        pil_images = []
        for img_bytes in reference_images:
            pil_images.append(PILImage.open(img_bytes))

        # Use gemini-3-pro-image-preview for multimodal generation
        model = "gemini-3-pro-image-preview"
        
        contents = [prompt] + pil_images
        
        logger.info(f"Generating with references (count={len(pil_images)}) using model: {model}")
        
        # User snippet config pattern
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"], # Strict image mode again
            image_config=types.ImageConfig(
                aspect_ratio="1:1",
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        if not response.parts:
            logger.warning(f"Gemini ref-gen returned no parts. Response: {response}")
            return None

        for part in response.parts:
            if part.inline_data: 
                return BytesIO(part.inline_data.data)
                
            try:
                if hasattr(part, "as_image"):
                    img = part.as_image()
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    return buf
            except:
                pass

    except Exception as e:
        logger.exception(f"Gemini ref-gen failed: {e}")

    return None

def search_elasticsearch_resource(query_string: str, index: str = "discord_chat_memory", max_results: int = 10) -> str:
    """
    Query the internal Elasticsearch data store (Resources).
    Use this to pull historical logs, messages, or structured data directly into your context.
    'query_string' follows Lucene syntax. 'index' is the ES index name.
    """
    from memory_utils import _search_raw
    try:
        # Map query_string to a simple ES query_string query
        resp = _search_raw({"query_string": {"query": query_string}}, index=index, size=max_results)
        # We return it as a string so Gemini can read it
        return json.dumps(resp, default=str)
    except Exception as e:
        return f"Error fetching ES resource: {e}"

def generate_gemini_text(prompt: str, context: Optional[List[Dict[str, str]]] = None, extra_parts: Optional[List[Any]] = None, status_tracker: Optional[Dict[str, str]] = None, enable_code_execution: bool = False) -> Tuple[Optional[str], List[Tuple[bytes, str]]]:
    """
    Generate text using Gemini (Chat). Supports context history, multiple parts (images, text, documents), streaming, and optional code execution.
    Returns: (text_response, list_of_images_as_bytes_and_mime)
    """
    client = _get_client()
    if not client:
        return None, []

    try:
        # Config for tools
        # As per https://ai.google.dev/gemini-api/docs/code-execution
        model = "gemini-1.5-flash"
        logger.info(f"Generating text with model: {model} (extra_parts={len(extra_parts) if extra_parts else 0}, code={enable_code_execution})")

        # Build contents from context + current prompt
        contents = []
        if context:
            for msg in context:
                role = "user" if msg.get("role") == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg.get("content"))]
                ))
        
        # Current message parts
        current_parts = [types.Part(text=prompt)]
        
        # Add extra parts (images, text files, etc.)
        if extra_parts:
            current_parts.extend(extra_parts)

        if enable_code_execution:
            current_parts.append(types.Part(text="\n(Important: Use the code_execution tool if needed to solve this.)"))

        # Add current prompt content
        contents.append(types.Content(
            role="user",
            parts=current_parts
        ))

        # ---- Define Tools Manually to avoid SDK validation bugs ----
        es_tool_spec = types.FunctionDeclaration(
            name="search_elasticsearch_resource",
            description="Query the internal Elasticsearch data store (Resources). Used to retrieve documents, archives, or logs not in current context.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query_string": types.Schema(
                        type="STRING", 
                        description="Lucene query syntax, e.g. 'error AND service:auth'"
                    ),
                    "index": types.Schema(
                        type="STRING", 
                        description="ES index name. Default is 'discord_chat_memory'."
                    ),
                    "max_results": types.Schema(
                        type="INTEGER", 
                        description="Number of docs to return."
                    )
                },
                required=["query_string"]
            )
        )

        tools_list = []
        
        # 1. Add Custom Function via correct Tool wrapper
        tools_list.append(types.Tool(function_declarations=[es_tool_spec]))
        
        # 2. Add Code Execution (if enabled)
        if enable_code_execution:
            try:
                # Some SDK versions use different naming for the Tool field
                if hasattr(types, "ToolCodeExecution"):
                    tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
                else:
                    tools_list.append(types.Tool(code_execution=types.CodeExecution()))
            except Exception as e:
                logger.warning(f"Failed to init code_execution tool: {e}")
        
        # 3. Add Google Search
        try:
             tools_list.append(types.Tool(google_search=types.GoogleSearch()))
        except Exception as e:
             logger.warning(f"Failed to init google_search tool: {e}")

        config = types.GenerateContentConfig(
            response_modalities=["TEXT"],
            system_instruction=(
                "You are Multivac, a helpful AI assistant. "
                "You can search historical logs or memory using 'search_elasticsearch_resource'. "
                "You can perform live computations or file generation using 'code_execution'. "
                "You can search the live web using 'google_search'."
            ),
            tools=tools_list,
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            ]
        )

        # Full text accumulator
        final_text = []
        # Artifact accumulator
        generated_artifacts = [] # List[(bytes, mime_type)]
        
        # Stream State
        accumulated_code_block = ""
        current_lang = "python" # default

        # STREAMING REQUEST
        # We iterate over chunks to update status_tracker with code
        # And build the final text cleanly (merging code chunks)
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config
        )
        
        for chunk in response_stream:
            # Process each chunk
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    # 1. Text Parts
                    if part.text:
                        # If we had accumulated code, flush it first
                        if accumulated_code_block:
                            block = f"\n> 🐍 **Thinking (Code Execution)**\n> ```{current_lang}\n{accumulated_code_block}\n> ```\n"
                            final_text.append(block)
                            accumulated_code_block = ""
                        
                        final_text.append(part.text)
                   
                    # 2. Inline Data (Images/Plots)
                    if part.inline_data:
                        logger.info(f"Received inline data: mime={part.inline_data.mime_type}, size={len(part.inline_data.data)}")
                        generated_artifacts.append((part.inline_data.data, part.inline_data.mime_type))
                    
                    # 3. Executable Code (The "Thinking" part)
                    if part.executable_code:
                        code_chunk = part.executable_code.code
                        if part.executable_code.language:
                            current_lang = part.executable_code.language.lower()
                        
                        accumulated_code_block += code_chunk
                        
                        # Update shared status for Progress Bar (Show last few lines)
                        if status_tracker is not None:
                            # Show last 6 lines of code
                            snippet = "\n".join(accumulated_code_block.splitlines()[-6:])
                            # If snippet is empty (just newlines), show something
                            if not snippet.strip():
                                snippet = "..." 
                            status_tracker["text"] = f"Writing Code...\n```{current_lang}\n{snippet}\n```"

                    # 4. Execution Result
                    if part.code_execution_result:
                        # Flush any accumulated code first
                        if accumulated_code_block:
                            block = f"\n> 🐍 **Thinking (Code Execution)**\n> ```{current_lang}\n{accumulated_code_block}\n> ```\n"
                            final_text.append(block)
                            accumulated_code_block = ""

                        outcome = part.code_execution_result.outcome
                        output = part.code_execution_result.output.strip()
                        icon = "✅" if outcome == "OUTCOME_OK" else "❌"
                        
                        # Format block
                        block = f"> {icon} **Result**\n> ```text\n{output}\n> ```\n"
                        final_text.append(block)
                        
                        if status_tracker is not None:
                            status_tracker["text"] = f"Executed: {outcome}\nResult: {output[:50]}..."
        
        # Flush any remaining code at end of stream
        if accumulated_code_block:
             block = f"\n> 🐍 **Thinking (Code Execution)**\n> ```{current_lang}\n{accumulated_code_block}\n> ```\n"
             final_text.append(block)

        if final_text or generated_artifacts:
            # Join text, ensure string
            full_text = "".join(final_text) if final_text else None
            return full_text, generated_artifacts
            
        logger.warning(f"Gemini text generation returned no text in stream.")

    except Exception as e:
        logger.exception(f"Gemini text generation failed: {e}")

    return None, []
