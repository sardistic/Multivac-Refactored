# weather_utils.py
# Hybrid geocoding (Google first, then OpenWeather) + OpenWeather data fetch.
# Exposes:
#   - get_location_details(query) -> {"name","lat","lon"}
#   - get_weather_data(lat,lon,units="imperial"|"metric") -> {"current","forecast"}
#   - handle_weather_request(*args, **kwargs) -> str (robust, backward-compatible)
#   - resolve_location(query) -> {"name","lat","lon"}  [alias for other bot features]
#
# Narrative:
# - Headline snapshot (name, emoji, sky, temp, humidity, wind).
# - One compact vignette for the *current period* (Morning / Afternoon / Evening / Overnight),
#   bounded by sunrise/sunset so it never crosses wildly into the next day.
# - A varied 3-day outlook (no POP, no copy-paste phrasing, hides tiny precip).

from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, Optional, Iterable, List, Tuple
from datetime import datetime, timezone, timedelta
import httpx

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or os.getenv("OWM_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY")

_DEFAULT_TIMEOUT = 6.0

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
_ZIP_RE = re.compile(r"\d{5}(?:-\d{4})?$")
_SNOWFLAKE_RE = re.compile(r"^\d{10,20}$")  # discord snowflakes

def _is_zip(s: str) -> bool:
    return bool(_ZIP_RE.fullmatch((s or "").strip()))

def _looks_like_snowflake(s: str) -> bool:
    return bool(_SNOWFLAKE_RE.fullmatch((s or "").strip()))

def _guess_units(query: str, country: Optional[str] = None) -> str:
    q = (query or "").lower()
    if " imperial" in q or " f " in f" {q} " or "fahrenheit" in q:
        return "imperial"
    if " metric" in q or " c " in f" {q} " or "celsius" in q:
        return "metric"
    if _is_zip(query) or (country and country.upper() in {"US"}):
        return "imperial"
    return "metric"

def _fmt_temp(v: Optional[float], units: str) -> str:
    if v is None:
        return "—"
    unit = "°F" if units == "imperial" else "°C"
    return f"{round(float(v))}{unit}"

def _fmt_speed(v: Optional[float], units: str) -> str:
    if v is None:
        return "—"
    return f"{round(float(v))} mph" if units == "imperial" else f"{round(float(v))} m/s"

def _mph_value(v: Optional[float], units: str) -> float:
    if v is None:
        return 0.0
    return float(v) if units == "imperial" else float(v) * 2.23694

def _wind_dir(deg: Optional[float]) -> str:
    if deg is None:
        return "—"
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    i = int((float(deg) / 22.5) + 0.5) % 16
    return dirs[i]

def _all_strings(items: Iterable[Any]) -> Iterable[str]:
    for x in items:
        if isinstance(x, str):
            yield x
        else:
            content = getattr(x, "content", None)
            if isinstance(content, str):
                yield content

def _pick_best_raw_text(*args, **kwargs) -> str:
    candidates = list(_all_strings(args)) + list(_all_strings(kwargs.values()))
    for s in candidates:
        if "weather" in s.lower():
            return s
    for s in candidates:
        ss = s.strip()
        if _looks_like_snowflake(ss):
            continue
        if any(ch.isalpha() for ch in ss) or ("," in ss) or (" " in ss):
            return ss
    for s in candidates:
        ss = s.strip()
        if _is_zip(ss):
            return ss
    q = kwargs.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return ""

# ----- time helpers (OneCall timezone_offset seconds) -----------------------------------
def _to_local(dt_utc_secs: int, tz_offset_secs: Optional[int]) -> datetime:
    if tz_offset_secs is None:
        return datetime.fromtimestamp(dt_utc_secs, tz=timezone.utc)
    return datetime.fromtimestamp(dt_utc_secs, tz=timezone(timedelta(seconds=int(tz_offset_secs))))

def _hour_label(dt: datetime) -> str:
    try:
        return dt.strftime("%-I %p")
    except Exception:
        return dt.strftime("%I %p").lstrip("0") or "12 AM"

def _weekday_label(dt: datetime) -> str:
    return dt.strftime("%a")

# ----- emoji + condition helpers --------------------------------------------------------
_EMOJI = {
    "Thunderstorm": "⛈️",
    "Drizzle": "🌦️",
    "Rain": "🌧️",
    "Snow": "❄️",
    "Clear": "☀️",
    "Clouds": "☁️",
    "Mist": "🌫️",
    "Smoke": "🌫️",
    "Haze": "🌫️",
    "Dust": "🌫️",
    "Fog": "🌫️",
    "Sand": "🌫️",
    "Ash": "🌫️",
    "Squall": "💨",
    "Tornado": "🌪️",
}

def _wx_emoji(wx: Dict[str, Any]) -> str:
    main = (wx or {}).get("main") or ""
    return _EMOJI.get(main, "🌡️")

def _wx_main(wx: Dict[str, Any]) -> str:
    return (wx or {}).get("main") or (wx or {}).get("description", "") or ""

def _precip_amount_1h(h: Dict[str, Any]) -> float:
    if isinstance(h.get("rain"), dict) and isinstance(h["rain"].get("1h"), (int, float)):
        return float(h["rain"]["1h"])  # mm
    if isinstance(h.get("snow"), dict) and isinstance(h["snow"].get("1h"), (int, float)):
        return float(h["snow"]["1h"])  # mm
    return 0.0

def _precip_amount_str_mm_to_units(mm: float, units: str) -> str:
    if units == "imperial":
        inches = mm * 0.0393701
        return f'{inches:.2f}"' if inches >= 0.95 else f'{inches:.1f}"'
    return f"{mm:.0f} mm"

# --------------------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------------------
async def get_location_details(query: str) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ValueError("Empty location query")

    # Prefer Google Geocoding if key available
    if GOOGLE_PLACES_API_KEY:
        try:
            params = {"key": GOOGLE_PLACES_API_KEY, "address": q}
            if _is_zip(q):
                params["components"] = "country:US"
            async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
                r = await client.get("https://maps.googleapis.com/maps/api/geocode/json", params=params)
                r.raise_for_status()
                data = r.json()
                if (data.get("status") == "OK") and data.get("results"):
                    top = data["results"][0]
                    loc = top["geometry"]["location"]
                    name = top.get("formatted_address") or q
                    return {"name": name, "lat": float(loc["lat"]), "lon": float(loc["lng"])}
                else:
                    logger.warning("Google geocode miss (%s): %s", q, data.get("status"))
        except Exception as e:
            logger.warning("Google geocode error, falling back to OpenWeather: %r", e)

    # Fallback: OpenWeather geocoder
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set (and Google Geocoding unavailable/failed).")

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        if _is_zip(q):
            url = "https://api.openweathermap.org/geo/1.0/zip"
            params = {"zip": f"{q},US", "appid": OPENWEATHER_API_KEY}
            r = await client.get(url, params=params)
            r.raise_for_status()
            j = r.json()
            name = f"{j.get('name')}, {j.get('country')}"
            return {"name": name, "lat": float(j["lat"]), "lon": float(j["lon"])}
        else:
            url = "https://api.openweathermap.org/geo/1.0/direct"
            params = {"q": q, "limit": 1, "appid": OPENWEATHER_API_KEY}
            r = await client.get(url, params=params)
            r.raise_for_status()
            arr = r.json() or []
            if not arr:
                raise LookupError(f"No results for location '{q}'")
            j = arr[0]
            parts = [p for p in [j.get("name"), j.get("state"), j.get("country")] if p]
            name = ", ".join(parts) or q
            return {"name": name, "lat": float(j["lat"]), "lon": float(j["lon"])}

# alias for other bot functions
resolve_location = get_location_details

# --------------------------------------------------------------------------------------
# Weather data
# --------------------------------------------------------------------------------------
async def get_weather_data(lat: float, lon: float, *, units: str = "imperial") -> Dict[str, Any]:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")
    units = "imperial" if units == "imperial" else "metric"

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        cur_params = {"lat": lat, "lon": lon, "units": units, "appid": OPENWEATHER_API_KEY}
        cur = await client.get("https://api.openweathermap.org/data/2.5/weather", params=cur_params)
        cur.raise_for_status()
        current = cur.json()

        fc_params = {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,alerts",
            "units": units,
            "appid": OPENWEATHER_API_KEY,
        }
        fc = await client.get("https://api.openweathermap.org/data/3.0/onecall", params=fc_params)
        fc.raise_for_status()
        forecast = fc.json()

    logger.info("OpenWeather Data: Current Weather: %s, Forecast: %s", current, forecast)
    return {"current": current, "forecast": forecast}

# --------------------------------------------------------------------------------------
# Narrative helpers
# --------------------------------------------------------------------------------------
def _period_bounds(now: datetime, sunrise: Optional[datetime], sunset: Optional[datetime]) -> Tuple[str, datetime]:
    """
    Map current local time to a named period and an end time that never jumps absurdly far.
    Periods:
      Morning: 06:00 -> 12:00 or sunrise->noon
      Afternoon: 12:00 -> 18:00 or noon->sunset
      Evening: 18:00 -> 22:00 or sunset->22:00
      Overnight: 22:00 -> 06:00 next day (cap)
    """
    hour = now.hour
    if sunrise and sunset:
        # use sun times when sensible
        noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now < noon:
            end = min(noon, sunset) if now >= sunrise else noon
            return ("Morning", end)
        if noon <= now < sunset:
            return ("Afternoon", sunset)
        if sunset <= now < now.replace(hour=22, minute=0, second=0, microsecond=0):
            end = now.replace(hour=22, minute=0, second=0, microsecond=0)
            return ("Evening", end)
    # fallback by clock
    if 6 <= hour < 12:
        return ("Morning", now.replace(hour=12, minute=0, second=0, microsecond=0))
    if 12 <= hour < 18:
        return ("Afternoon", now.replace(hour=18, minute=0, second=0, microsecond=0))
    if 18 <= hour < 22:
        return ("Evening", now.replace(hour=22, minute=0, second=0, microsecond=0))
    # Overnight spans to next 06:00
    next_morning = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
    return ("Overnight", next_morning)

def _dominant_condition_and_emoji(hours: List[Dict[str, Any]]) -> Tuple[str, str]:
    counts: Dict[str, int] = {}
    sample = None
    for h in hours:
        wx = (h.get("weather") or [{}])[0]
        main = _wx_main(wx)
        counts[main] = counts.get(main, 0) + 1
        if sample is None:
            sample = wx
    if not counts:
        return ("—", "🌡️")
    main = max(counts.items(), key=lambda kv: kv[1])[0]
    for h in hours:
        wx = (h.get("weather") or [{}])[0]
        if _wx_main(wx) == main:
            return (main, _wx_emoji(wx))
    return (main, _wx_emoji(sample or {}))

def _precip_window(hours: List[Dict[str, Any]]) -> Optional[Tuple[int, int, float, str]]:
    first = last = -1
    total = 0.0
    kind = ""
    for i, h in enumerate(hours):
        mm = _precip_amount_1h(h)
        if mm > 0:
            total += mm
            wx = (h.get("weather") or [{}])[0]
            k = "snow" if "Snow" in _wx_main(wx) else "rain"
            if not kind:
                kind = k
            if first == -1:
                first = i
            last = i
    if first == -1:
        return None
    return (first, last, total, kind)

def _wind_phrase(hours: List[Dict[str, Any]], units: str) -> str:
    speeds = [_mph_value(h.get("wind_speed"), units) for h in hours if isinstance(h.get("wind_speed"), (int, float))]
    gusts = [_mph_value(h.get("wind_gust"), units) for h in hours if isinstance(h.get("wind_gust"), (int, float))]
    dir_deg = next((h.get("wind_deg") for h in reversed(hours) if isinstance(h.get("wind_deg"), (int, float))), None)

    avg = sum(speeds) / len(speeds) if speeds else 0.0
    gmax = max(gusts) if gusts else 0.0

    if avg < 1 and gmax < 6:
        return "air barely moving."
    def label(v: float) -> str:
        if v < 6: return "light"
        if v < 12: return "lazy"
        if v < 18: return "breezy"
        if v < 28: return "restless"
        if v < 38: return "gusty"
        return "howling"
    bits = [f"{label(avg)} {_wind_dir(dir_deg) if dir_deg is not None else ''} winds ~{round(avg)} mph".strip()]
    if gmax and gmax - avg >= 5:
        bits.append(f"gusts near {round(gmax)}")
    if units != "imperial":
        bits.append(f"({avg/2.23694:.0f} m/s)")
    return (", ".join(bits) + ".").replace("  ", " ")

def _temp_trend_phrase(temps: List[float], units: str) -> str:
    if not temps:
        return ""
    t0 = temps[0]; tmin = min(temps); tmax = max(temps)
    if tmax - t0 >= 3:
        return f"edges warmer toward { _fmt_temp(tmax, units) }"
    if t0 - tmin >= 3:
        return f"slips toward { _fmt_temp(tmin, units) }"
    return f"holds near { _fmt_temp(t0, units) }"

def _sky_color(main: str, is_day: bool) -> str:
    m = (main or "").lower()
    if "thunder" in m: return "air cracked with thunder"
    if "drizzle" in m: return "fine drizzle threading the streets"
    if "rain" in m:    return "rain under a low ceiling" if is_day else "rain tapping at the dark"
    if "snow" in m:    return "snow smoothing the edges"
    if "clear" in m:   return "clear and still" if not is_day else "clear and bright"
    if "cloud" in m:   return "clouds moving like slow ships"
    if m in {"mist","haze","fog"}: return "haze close to the ground"
    return "weather in between"

# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------
def _format_current(current: Dict[str, Any], units: str) -> str:
    name = current.get("name") or "Current location"
    wx = (current.get("weather") or [{}])[0]
    desc = (wx.get("description") or "").capitalize()
    emoji = _wx_emoji(wx)
    main = current.get("main", {})
    wind = current.get("wind", {})

    temp = _fmt_temp(main.get("temp"), units)
    feels = _fmt_temp(main.get("feels_like"), units)
    rh = main.get("humidity")
    wind_spd = _fmt_speed(wind.get("speed"), units)
    wind_deg = wind.get("deg")
    gust = wind.get("gust")
    gust_s = f" (gusts { _fmt_speed(gust, units) })" if isinstance(gust, (int, float)) else ""

    line1 = f"**{name}** — {emoji} {desc}"
    line2 = f"Temp {temp} (feels {feels}) • Humidity {rh}%"
    line3 = f"Wind {wind_spd} {_wind_dir(wind_deg)}{gust_s}"
    return f"{line1}\n{line2} • {line3}"

def _format_period_vignette(forecast: Dict[str, Any], units: str) -> str:
    hourly = forecast.get("hourly") or []
    if not hourly:
        return "_No short-term data_"

    tz_off = forecast.get("timezone_offset")
    now_local = _to_local(forecast.get("current", {}).get("dt", hourly[0].get("dt", 0)), tz_off)

    # find sunrise/sunset from daily[0] if present
    sunrise = sunset = None
    if forecast.get("daily"):
        d0 = forecast["daily"][0]
        sr = d0.get("sunrise"); ss = d0.get("sunset")
        sunrise = _to_local(sr, tz_off) if isinstance(sr, int) else None
        sunset  = _to_local(ss, tz_off) if isinstance(ss, int) else None

    label, end_local = _period_bounds(now_local, sunrise, sunset)

    # slice hours strictly within [now, end_local]
    window: List[Dict[str, Any]] = []
    for h in hourly:
        t = _to_local(h.get("dt", 0), tz_off)
        if now_local <= t <= end_local:
            window.append(h)
    if not window:
        window = hourly[:6]  # graceful fallback

    temps = [float(h.get("temp")) for h in window if isinstance(h.get("temp"), (int, float))]
    trend = _temp_trend_phrase(temps, units)

    is_day = sunrise is not None and sunset is not None and sunrise <= now_local < sunset
    dom, emoji = _dominant_condition_and_emoji(window)
    sky = _sky_color(dom, is_day)

    # precipitation
    pwin = _precip_window(window)
    precip_sent = ""
    if pwin:
        i0, i1, total_mm, kind = pwin
        total_in = total_mm * 0.0393701
        if total_in >= 0.1 or total_mm >= 3:  # hide trivial spit
            t0lab = _hour_label(_to_local(window[i0].get("dt", 0), tz_off))
            t1lab = _hour_label(_to_local(window[i1].get("dt", 0), tz_off))
            span = f"{t0lab}" if i0 == i1 else f"{t0lab}–{t1lab}"
            amt = _precip_amount_str_mm_to_units(total_mm, units)
            noun = "snow" if kind == "snow" else "rain"
            precip_sent = f"{noun.capitalize()} around {span} (~{amt})."

    wind_sent = _wind_phrase(window, units)

    # compact, no direct “today/tonight/local” references
    lead = f"**{label}:** {emoji} {sky}."
    tempo = trend + "." if trend else ""
    return " ".join(x for x in [lead, tempo, precip_sent, wind_sent] if x).replace("  ", " ")

def _daily_line(d: Dict[str, Any], tz_offset: Optional[int], units: str) -> str:
    dt = _to_local(d.get("dt", 0), tz_offset)
    day = _weekday_label(dt)
    wx = (d.get("weather") or [{}])[0]
    emoji = _wx_emoji(wx)
    desc = (wx.get("description") or "").capitalize()

    temps = d.get("temp") or {}
    hi = _fmt_temp(temps.get("max"), units)
    lo = _fmt_temp(temps.get("min"), units)

    bits = [f"- {day}: {emoji} {desc}. High {hi}, low {lo}."]

    # precip totals if notable
    if isinstance(d.get("rain"), (int, float)) and d["rain"] > 0:
        amt = d["rain"]
        if units == "imperial":
            inches = amt * 0.0393701
            if inches >= 0.1:
                bits.append(f" Rain at times (~{inches:.1f}\").")
    elif isinstance(d.get("snow"), (int, float)) and d["snow"] > 0:
        amt = d["snow"]
        if units == "imperial":
            inches = amt * 0.0393701
            if inches >= 0.1:
                bits.append(f" Snow possible (~{inches:.1f}\").")

    # wind color without repeating the same phrase every day
    wspd = d.get("wind_speed"); wgust = d.get("wind_gust"); wdeg = d.get("wind_deg")
    if isinstance(wspd, (int, float)):
        mph = _mph_value(float(wspd), units)
        if   mph < 6:  feel = "light"
        elif mph < 12: feel = "easy"
        elif mph < 18: feel = "steady"
        elif mph < 28: feel = "noticeable"
        elif mph < 38: feel = "gusty"
        else:          feel = "strong"
        tail = []
        tail.append(_wind_dir(wdeg) if isinstance(wdeg, (int, float)) else "")
        if isinstance(wgust, (int, float)) and _mph_value(float(wgust), units) - mph >= 6:
            tail.append(f"gusts { _fmt_speed(wgust, units) }")
        tail = ", ".join([t for t in tail if t])
        bits.append(f" Winds {feel}{(', ' + tail) if tail else ''}.")
    return " ".join(bits).replace("  ", " ")

def _format_daily(forecast: Dict[str, Any], units: str, *, days: int = 3) -> str:
    daily = forecast.get("daily") or []
    if not daily:
        return "_No daily data_"
    tz_offset = forecast.get("timezone_offset")
    lines = ["**Daily:**"]
    for d in daily[:max(1, days)]:
        lines.append(_daily_line(d, tz_offset, units))
    return "\n".join(lines)

def format_weather_response(loc_name: str, data: Dict[str, Any], units: str) -> str:
    current = data.get("current") or {}
    forecast = data.get("forecast") or {}
    if current.get("name"):
        loc_name = current["name"]

    parts = [
        _format_current(current, units),
        _format_period_vignette(forecast, units),
        _format_daily(forecast, units, days=3),
    ]
    return "\n\n".join(parts)

# --------------------------------------------------------------------------------------
# Robust dispatcher used by discord_bot
# --------------------------------------------------------------------------------------
async def handle_weather_request(*args, **kwargs) -> str:
    """
    Flexible entry-point so older/newer discord_bot call sites keep working.
    Accepts strings or message objects; ignores Discord snowflake IDs.
    """
    raw = _pick_best_raw_text(*args, **kwargs).strip()
    if not raw:
        return "I couldn’t find a location. Try: `weather 10001` or `weather Raleigh, NC`."

    m = re.search(r"\bweather\b(.*)$", raw, flags=re.IGNORECASE)
    loc_str = (m.group(1) if m else raw).strip(" ,;-")
    if not loc_str:
        return "Tell me a location, e.g. `weather 90210` or `weather Paris`."

    try:
        loc = await get_location_details(loc_str)
    except Exception as e:
        logger.error("get_location_details failed for %r: %r", loc_str, e)
        return f"I couldn’t resolve that location (`{loc_str}`)."

    units_pref = (kwargs.get("units") or _guess_units(loc_str, None)).lower()
    if units_pref not in ("imperial", "metric"):
        units_pref = _guess_units(loc_str, None)

    try:
        data = await get_weather_data(loc["lat"], loc["lon"], units=units_pref)
    except Exception as e:
        logger.error("get_weather_data failed: %r", e)
        return "Weather service is having trouble right now. Please try again shortly."

    try:
        return format_weather_response(loc["name"], data, units_pref)
    except Exception as e:
        logger.error("format_weather_response failed: %r", e)
        cur = data.get("current", {})
        temp = _fmt_temp((cur.get("main") or {}).get("temp"), units_pref)
        wx = (cur.get("weather") or [{}])[0]
        desc = (wx.get("description") or "").capitalize()
        name = loc.get("name", "Selected location")
        return f"**{name}** — {desc}\nTemp: {temp}"

# --------------------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------------------
__all__ = [
    "get_location_details",
    "get_weather_data",
    "handle_weather_request",
    "format_weather_response",
    "resolve_location",
]
