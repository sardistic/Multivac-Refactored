import os
import requests
from config import OPENAI_API_KEY

def check_models():
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Error {resp.status_code}: {resp.text}")
            return
        
        data = resp.json()
        models = [m["id"] for m in data.get("data", [])]
        
        # Filter
        image_models = [m for m in models if "image" in m or "dall-e" in m]
        print(f"Found {len(models)} models.")
        print("Image models:", image_models)
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    check_models()
