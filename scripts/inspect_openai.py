import openai
import json

try:
    client = openai.OpenAI()
    attrs = dir(client)
    relevant = [a for a in attrs if "video" in a or "beta" in a or "images" in a]
    print(f"Client attributes matching 'video/beta/images': {relevant}")
    
    if "beta" in attrs:
        print(f"client.beta attributes: {dir(client.beta)}")
        
    # Check if 'video' is directly there?
    if hasattr(client, 'video'):
        print("client.video exists!")
    else:
        print("client.video MISSING")

except Exception as e:
    print(e)
