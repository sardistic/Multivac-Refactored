import requests
import openai
from config import OPENAI_API_KEY

print("--- DEEP PROBE ---")

# 1. Check SDK Internals
try:
    import openai.resources.video
    print("SUCCESS: import openai.resources.video")
except ImportError:
    print("FAIL: import openai.resources.video")

try:
    from openai.resources.video.generations import AsyncGenerations
    print("SUCCESS: from openai.resources.video.generations import AsyncGenerations")
except ImportError:
    print("FAIL: import AsyncGenerations")

# 2. Probe URL Variations
endpoints = [
    "https://api.openai.com/v1/video/generations",
    "https://api.openai.com/v1/videos/generations",
    "https://api.openai.com/v1/sora/generations",
    "https://api.openai.com/v1/images/generations", # Try sora model here
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

for url in endpoints:
    try:
        resp = requests.post(url, headers=headers, json=payload)
        print(f"URL: {url} -> {resp.status_code}")
        if resp.status_code == 400:
             # Often contains useful "Supported models" list
             print(f"ERR MSG: {resp.text[:300]!r}")
    except Exception as e:
        print(f"ERR: {url} -> {e}")

print("--- END ---")
