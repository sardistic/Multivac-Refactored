# url_utils.py
import logging
import re
import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_url_content(url: str, timeout: int = 15) -> str:
    """Return raw HTML (or raise for non-200)."""
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.text

def extract_main_text(html: str) -> tuple[str, str]:
    """
    Very lightweight content extraction.
    Returns (title, text).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Prefer article-ish containers
    candidates = []
    for sel in ["article", "[role=main]", "main", ".post", ".article", ".entry-content"]:
        candidates.extend(soup.select(sel))

    def node_text(node):
        # Drop script/style/nav/aside
        for bad in node(["script", "style", "noscript", "nav", "aside", "form", "footer", "header"]):
            bad.decompose()
        txt = " ".join(t.get_text(" ", strip=True) for t in node.find_all(["p", "li", "h2", "h3", "h4"]))
        return re.sub(r"\s+", " ", txt).strip()

    text = ""
    if candidates:
        biggest = max(candidates, key=lambda n: len(n.get_text(strip=True)))
        text = node_text(biggest)

    # Fallback: whole page paragraphs
    if len(text) < 400:
        for bad in soup(["script", "style", "noscript", "nav", "aside", "form", "footer", "header"]):
            bad.decompose()
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        text = re.sub(r"\s+", " ", text).strip()

    title = (soup.title.get_text(strip=True) if soup.title else "")[:300]
    return title, text

def reduce_text_length(text: str, max_chars: int = 3000) -> str:
    """Trim by sentences to fit within max_chars."""
    if len(text) <= max_chars:
        return text
    # Split on sentence-ish boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    out = []
    total = 0
    for part in parts:
        if total + len(part) + 1 > max_chars:
            break
        out.append(part)
        total += len(part) + 1
    s = " ".join(out).strip()
    if not s:
        s = text[:max_chars - 1]
    return s + "…"
