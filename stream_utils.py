# stream_utils.py
import asyncio
import time
import logging
from typing import Optional, Callable, List

DEFAULT_EDIT_INTERVAL_SEC = 1.5   # don’t spam Discord edits
DEFAULT_SUMMARY_TARGET_CHARS = 320  # short, coherent chunks

def _squash_ws(s: str) -> str:
    return " ".join(s.split())

class StreamEditSession:
    """
    Buffers tokens, emits periodic, *summarized* edits so users see coherent
    updates (not choppy token streams). Finalizes with the full answer.
    """
    def __init__(
        self,
        status_msg,                              # discord.Message to edit
        summarize_fn: Callable[[str], str],      # sync/async function that condenses text
        *,
        edit_interval: float = DEFAULT_EDIT_INTERVAL_SEC,
        summary_target_chars: int = DEFAULT_SUMMARY_TARGET_CHARS,
        prefix: str = "",
        suffix: str = "",
    ):
        self.status_msg = status_msg
        self.summarize_fn = summarize_fn
        self.edit_interval = edit_interval
        self.summary_target_chars = summary_target_chars
        self.prefix = prefix
        self.suffix = suffix

        self._buffer: List[str] = []
        self._full_text_parts: List[str] = []
        self._last_edit_ts = 0.0
        self._closed = False
        self._lock = asyncio.Lock()

    async def _maybe_edit(self, force: bool = False):
        if self._closed:
            return
        now = time.monotonic()
        if not force and (now - self._last_edit_ts) < self.edit_interval:
            return

        buf = _squash_ws(" ".join(self._buffer)).strip()
        if not buf:
            return

        # summarize buffer for a coherent partial
        partial = await _maybe_await(self.summarize_fn(_clip(buf, self.summary_target_chars)))
        content = f"{self.prefix}{partial}{self.suffix}".strip()

        try:
            await self.status_msg.edit(content=content)
            self._last_edit_ts = now
            self._buffer.clear()
        except Exception as e:
            logging.debug(f"stream edit skipped: {e}")

    async def feed_tokens(self, text: str):
        if not text or self._closed:
            return
        async with self._lock:
            self._buffer.append(text)
            self._full_text_parts.append(text)
            await self._maybe_edit()

    async def annotate(self, line: str):
        """
        Immediately add a high-level “what I’m doing” line on top.
        """
        async with self._lock:
            head = f"{line.strip()}\n\n"
            try:
                await self.status_msg.edit(content=head)
                self._last_edit_ts = time.monotonic()
            except Exception:
                pass

    async def finalize(self, final_text: str, *, header: Optional[str] = None):
        """
        Final, untruncated message. Clears any buffer and posts the full answer.
        """
        if self._closed:
            return
        async with self._lock:
            self._closed = True
            content = final_text
            if header:
                content = f"{header.strip()}\n\n{final_text}"
            try:
                await self.status_msg.edit(content=content)
            except Exception as e:
                logging.debug(f"finalize edit failed: {e}")

def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    # soft cut on sentence/punctuation
    cut = s[: n + 200]
    for splitter in [". ", "! ", "? ", "\n"]:
        idx = cut.rfind(splitter)
        if idx > 0 and idx >= n - 150:
            return cut[: idx + 1] + " …"
    return s[:n] + " …"

async def _maybe_await(x):
    if asyncio.iscoroutine(x):
        return await x
    return x
