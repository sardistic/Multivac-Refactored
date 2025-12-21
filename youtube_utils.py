# youtube_utils.py
import re
from typing import Optional, List

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_YT_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:.*&)?v=|v/|embed/))([A-Za-z0-9_-]{11})"
)

def extract_youtube_id(url: str) -> Optional[str]:
    m = _YT_ID_RE.search(url)
    if m:
        return m.group(1)
    # fallback: try v= query param manually
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Returns a plain-text transcript (no timestamps), or None if unavailable.
    """
    try:
        transcript: List[dict] = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        # network or other error
        return None

    # join all text chunks with spaces
    return " ".join(chunk.get("text", "").strip() for chunk in transcript if chunk.get("text"))
