import openai
import sys

print(f"OpenAI Version: {openai.__version__}")

try:
    client = openai.OpenAI(api_key="sk-dummy")
    
    print("\n--- Client Attributes ---")
    print([x for x in dir(client) if not x.startswith("_")])
    
    if hasattr(client, 'beta'):
        print("\n--- Client.beta Attributes ---")
        print([x for x in dir(client.beta) if not x.startswith("_")])
        
    if hasattr(client, 'video'):
        print("\n--- Client.video Attributes ---")
        print([x for x in dir(client.video) if not x.startswith("_")])

except Exception as e:
    print(f"\nError: {e}")
