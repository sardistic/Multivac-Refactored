# verification_script.py
import os
import sys
import logging
from gemini_utils import generate_gemini_image

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_generation():
    print("Testing Gemini Image Generation...")
    
    # Check for API Key
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("Please set it before running this script: $env:GEMINI_API_KEY='your_key'")
        return

    prompt = "a cyberpunk cat sitting on a neon rooftop"
    print(f"Prompt: {prompt}")
    
    auth_redacted = key[:5] + "..." + key[-5:]
    print(f"Key: {auth_redacted}")

    try:
        result = generate_gemini_image(prompt)
        
        if result:
            print("✅ Image generated successfully!")
            size = result.getbuffer().nbytes
            print(f"Image size: {size} bytes")
            
            with open("test_gemini_output.png", "wb") as f:
                f.write(result.getvalue())
            print("Saved to test_gemini_output.png")
        else:
            print("❌ Image generation failed (returned None). Check logs.")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")

if __name__ == "__main__":
    test_generation()
