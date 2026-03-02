from __future__ import annotations

import aiohttp

from config import OPENAI_API_KEY

API_BASE = "https://api.openai.com/v1"


def sora_headers(json_content: bool = False):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def build_session():
    return aiohttp.ClientSession()
