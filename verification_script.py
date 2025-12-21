# verification_script.py
import os
import sys
import logging
from io import BytesIO
from gemini_utils import generate_gemini_image, edit_gemini_image

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_generation():
    print("--- Testing 'gemini imagine' ---")
    prompt = "a cyberpunk cat sitting on a neon rooftop"
    print(f"Prompt: {prompt}")

    try:
        result = generate_gemini_image(prompt)
        if result:
            print("✅ Generation successful!")
            with open("test_gemini_gen.png", "wb") as f:
                f.write(result.getvalue())
            print("Saved to test_gemini_gen.png")
        else:
            print("❌ Generation failed (returned None).")
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_editing():
    print("\n--- Testing 'gemini edit' ---")
    if not os.path.exists("test_gemini_gen.png"):
        print("⚠️ Skipping edit test (no input image from generation step).")
        return

    prompt = "make the cat orange"
    print(f"Prompt: {prompt}")
    
    try:
        with open("test_gemini_gen.png", "rb") as f:
            img_bytes = BytesIO(f.read())
            
        result = edit_gemini_image(img_bytes, prompt)
        if result:
            print("✅ Edit successful!")
            with open("test_gemini_edit.png", "wb") as f:
                f.write(result.getvalue())
            print("Saved to test_gemini_edit.png")
        else:
            print("❌ Edit failed (returned None).")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found.")
    else:
        test_generation()
        test_editing()
