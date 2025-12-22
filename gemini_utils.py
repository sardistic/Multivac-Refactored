import logging
import base64
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

def generate_gemini_text(prompt: str, context: Optional[List[Dict[str, str]]] = None, extra_parts: Optional[List[Any]] = None, status_tracker: Optional[Dict[str, str]] = None, enable_code_execution: bool = False) -> Tuple[Optional[str], List[Tuple[bytes, str]]]:
    """
    Generate text using Gemini (Chat). Supports context history, multiple parts (images, text, documents), streaming, and optional code execution.
    Returns: (text_response, list_of_images_as_bytes_and_mime)
    """
    client = _get_client()
    if not client:
        return None, []

    try:
        model = "gemini-3-flash-preview"
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
            # extra_parts should be a list of types.Part
            current_parts.extend(extra_parts)

        if enable_code_execution:
            # Force/Encourage usage
            current_parts.append(types.Part(text="\n(Important: Use the code_execution tool to solve this.)"))

        # Add current prompt content
        contents.append(types.Content(
            role="user",
            parts=current_parts
        ))

        # Config for code execution
        tools_list = []
        if enable_code_execution:
            code_tool = None
            try:
                if hasattr(types, "ToolCodeExecution"):
                    code_tool = types.Tool(code_execution=types.ToolCodeExecution())
                elif hasattr(types, "CodeExecution"):
                    code_tool = types.Tool(code_execution=types.CodeExecution())
                else:
                     code_tool = types.Tool(code_execution={})
            except Exception as e:
                logger.warning(f"Failed to init code_execution tool: {e}")
            
            if code_tool:
                tools_list = [code_tool]
        
        # Add Google Search Tool (always available for grounding)
        try:
             # Create the search tool configuration
             # Note: API now requests 'google_search' instead of 'google_search_retrieval'
             search_tool = types.Tool(
                 google_search=types.GoogleSearch()
             )
             tools_list.append(search_tool)
        except Exception as e:
             logger.warning(f"Failed to init google_search tool: {e}")

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"], # Explicitly allow IMAGE for code artifacts
            system_instruction="You are Multivac, a helpful AI assistant. You have access to the recent conversation history provided in the context. Use it to answer questions about what was previously said.",
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
