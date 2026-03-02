import openai
import json
import sys

print("Python version:", sys.version)
print("OpenAI version:", openai.__version__)

try:
    # Use dummy key to avoid validation error during init
    client = openai.OpenAI(api_key="sk-dummy-123")
    attrs = dir(client)
    relevant = [a for a in attrs if "video" in a or "beta" in a or "images" in a]
    print(f"Client attributes matching 'video/beta/images': {relevant}")
    
    if "beta" in attrs:
        print(f"client.beta attributes: {dir(client.beta)}")
        
    if hasattr(client, 'video'):
        print("client.video exists!")
    else:
        print("client.video MISSING")

except Exception as e:
    print(f"Error: {e}")
