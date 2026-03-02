import requests
import json
from config import OPENAI_API_KEY

url = "https://api.openai.com/v1/videos"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}
payload = {
    "model": "sora-2-pro",
    "prompt": "test prompt for sora",
    "size": "1280x720",
    "seconds": "8"
}

print(f"POST {url} with {json.dumps(payload)}")
try:
    resp = requests.post(url, headers=headers, json=payload)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
