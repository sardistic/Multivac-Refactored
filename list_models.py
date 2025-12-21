import os
import requests
import json
from config import GEMINI_API_KEY

def list_models():
    """
    List available models from Gemini API to debug '404 Not Found' errors.
    """
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set.")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    
    try:
        print(f"Fetching models from: {url.split('?')[0]}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"✅ Found {len(models)} models:")
            for m in models:
                # Filter for likely image generation models if possible, or just print all
                name = m.get("name")
                supported_methods = m.get("supportedGenerationMethods", [])
                print(f" - {name} (Methods: {supported_methods})")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    list_models()
