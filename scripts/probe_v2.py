import requests
from config import OPENAI_API_KEY

endpoints = [
    "https://api.openai.com/v1/video/generations",
    "https://api.openai.com/v1/videos/generations",
    "https://api.openai.com/v1/sora/generations",
    "https://api.openai.com/v1/videos",
    "https://api.openai.com/v1/video"
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

print("--- PROBE START ---")
for url in endpoints:
    try:
        resp = requests.post(url, headers=headers, json=payload)
        print(f"URL: {url}")
        print(f"STATUS: {resp.status_code}")
        # Only print a snippet of error to keep logs clean
        if resp.status_code != 200:
             print(f"MSG: {resp.text[:100]!r}")
        print("-" * 20)
    except Exception as e:
        print(f"ERR: {url} -> {e}")
print("--- PROBE END ---")
