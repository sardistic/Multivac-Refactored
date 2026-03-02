# stock_utils.py
from __future__ import annotations

from config import FINNHUB_API_TOKEN
import requests
from discord import Embed

FINNHUB_BASE = "https://finnhub.io/api/v1"


def _get(path: str, **params) -> dict | None:
    params = dict(params or {})
    params["token"] = FINNHUB_API_TOKEN
    url = f"{FINNHUB_BASE}/{path}"
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return None
    return None


def get_realtime_quote(ticker: str) -> dict:
    """
    Unified quote payload:
    {
      "ok": True/False,
      "ticker": "AAPL",
      "price": 194.23,
      "prev_close": 193.01,
      "change_pct": 0.63
    }
    """
    raw = _get("quote", symbol=ticker)
    if not isinstance(raw, dict):
        return {"ok": False, "error": "bad_response"}

    c = raw.get("c")  # current
    pc = raw.get("pc")  # prev close
    if c is None or pc is None:
        return {"ok": False, "error": "missing_fields"}

    try:
        change_pct = ((float(c) - float(pc)) / float(pc)) * 100.0 if float(pc) != 0 else 0.0
    except Exception:
        change_pct = None

    return {
        "ok": True,
        "ticker": ticker.upper(),
        "price": float(c),
        "prev_close": float(pc),
        "change_pct": float(change_pct) if change_pct is not None else None,
    }


def fetch_company_profile(ticker: str) -> dict | None:
    return _get("stock/profile2", symbol=ticker)


async def handle_stock_command(message, prompt: str):
    if message.author == message.guild.me:
        return

    parts = prompt.strip().split()
    ticker = parts[1] if len(parts) > 1 else None

    if not ticker:
        await message.channel.send("⚠️ Missing ticker symbol. Usage: `stock TSLA`")
        return

    q = get_realtime_quote(ticker)
    if not q.get("ok"):
        await message.channel.send(f"Could not fetch stock information for `{ticker.upper()}`.")
        return

    last_price = q.get("price")
    prev_close = q.get("prev_close")
    change_pct = q.get("change_pct")

    last_price_s = f"{last_price:.2f}" if last_price is not None else "N/A"
    prev_close_s = f"{prev_close:.2f}" if prev_close is not None else "N/A"
    change_pct_s = f"{change_pct:+.2f}%" if change_pct is not None else "N/A"

    color = 0xff0000 if (isinstance(change_pct, (int, float)) and change_pct < 0) else 0x00ff00
    google_search_link = f"https://www.google.com/search?q=stock%20quote%20{q['ticker']}"
    embed = Embed(
        title=f"💰 Latest stock quote for {q['ticker']}",
        url=google_search_link,
        color=color,
    )
    embed.add_field(name="Latest Today", value=f"${last_price_s} ({change_pct_s})", inline=False)
    embed.add_field(name="Previous Close", value=f"${prev_close_s}", inline=False)
    embed.set_footer(text="Source: Finnhub.io API")

    await message.channel.send(embed=embed)
