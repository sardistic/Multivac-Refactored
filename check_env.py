import os
import sys
from config import GEMINI_API_KEY

try:
    from google import genai
    from google.genai import types
    print(f"SDK Available: Yes")
    print(f"SDK File: {genai.__file__}")
    
    # Inspect types for CodeExecution
    print("\n--- Types in google.genai.types ---")
    relevant_types = [t for t in dir(types) if "Code" in t or "Tool" in t]
    for t in relevant_types:
        print(t)
        
except ImportError as e:
    print(f"SDK Error: {e}")
    sys.exit(1)

def list_models():
    print("\n--- Listing Models ---")
    if not GEMINI_API_KEY:
        print("No GEMINI_API_KEY found in config.")
        return

    try:
        # Use the exact client setup we just put in gemini_utils
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        
        # Try to list models
        # paging logic might be needed but let's try basic list
        # The new SDK might vary on how it lists files, let's try the standard method if it exists
        if hasattr(client.models, 'list'):
             models = client.models.list()
             for m in models:
                 print(f"Name: {m.name} | Methods: {m.supported_generation_methods}")
        else:
            print("client.models has no 'list' method? Listing dir(client.models):")
            print(dir(client.models))

    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
