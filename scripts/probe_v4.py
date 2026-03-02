import requests
from config import OPENAI_API_KEY

endpoints = [
    "https://api.openai.com/v1/sora/generations",
    "https://api.openai.com/v1/chat/completions" 
]

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

payloads = {
    "https://api.openai.com/v1/sora/generations": {
        "model": "sora-2-pro",
        "prompt": "test",
        "size": "1024x1024"
    },
    "https://api.openai.com/v1/chat/completions": {
        "model": "sora-2-pro",
        "messages": [{"role": "user", "content": "generate a video of a cat"}]
    }
}

print("--- PROBE V4 ---")
for url in endpoints:
    try:
        data = payloads[url]
        resp = requests.post(url, headers=headers, json=data)
        print(f"URL: {url} -> {resp.status_code}")
        if resp.status_code != 200:
             print(f"ERR: {resp.text[:300]!r}")
    except Exception as e:
        print(f"EXC: {url} -> {e}")
print("--- END ---")
