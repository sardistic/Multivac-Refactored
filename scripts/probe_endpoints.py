import requests
from config import OPENAI_API_KEY

endpoints = [
    "https://api.openai.com/v1/video/generations",
    "https://api.openai.com/v1/videos/generations",
    "https://api.openai.com/v1/sora/generations",
    "https://api.openai.com/v1/images/generations" # Control
]

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "sora-2-pro",
    "prompt": "test",
    "size": "1024x1024"
}

print("Probing endpoints...")
for url in endpoints:
    try:
        resp = requests.post(url, headers=headers, json=payload)
        print(f"{url} -> {resp.status_code}")
        if resp.status_code != 404:
            print(f"Response: {resp.text[:200]}")
    except Exception as e:
        print(f"{url} -> Error: {e}")
