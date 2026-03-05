import os
import requests

MD_BASE = "http://metadata.google.internal/computeMetadata/v1"
MD_HEADERS = {"Metadata-Flavor": "Google"}

def _md_fetch(path: str, timeout: float = 1.0) -> str | None:
    try:
        r = requests.get(f"{MD_BASE}/{path}", headers=MD_HEADERS, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text.strip()
    except Exception:
        pass
    return None

def get_metadata(key: str, default: str | None = None, mirror_env: bool = True) -> str | None:
    """
    Priority: ENV -> instance metadata -> project metadata -> default
    Optionally mirrors discovered values back into os.environ so libraries that
    only read env vars keep working.
    """
    # 1) env
    val = os.environ.get(key)
    if val:
        return val

    # 2) instance attribute
    val = _md_fetch(f"instance/attributes/{key}")
    if val:
        if mirror_env and not os.environ.get(key):
            os.environ[key] = val
        return val

    # 3) project attribute
    val = _md_fetch(f"project/attributes/{key}")
    if val:
        if mirror_env and not os.environ.get(key):
            os.environ[key] = val
        return val

    return default

def _truthy(s: str | None, default: bool = False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}

# --- App/API keys ---
DISCORD_TOKEN           = get_metadata("DISCORD_TOKEN")
OPENAI_API_KEY          = get_metadata("OPENAI_API_KEY")
OPENAI_DEFAULT_MODEL    = get_metadata("OPENAI_DEFAULT_MODEL", "gpt-5.2")
STABILITY_HOST          = get_metadata("STABILITY_HOST")
STABILITY_KEY           = get_metadata("STABILITY_KEY")
GOOGLE_PLACES_API_KEY   = get_metadata("GOOGLE_PLACES_API_KEY")
OPENWEATHER_API_KEY     = get_metadata("OPENWEATHER_API_KEY")
FINNHUB_API_TOKEN       = get_metadata("FINNHUB_API_TOKEN")

# NEW: Google Programmable Search (Custom Search JSON API)
GOOGLE_API_KEY          = get_metadata("GOOGLE_API_KEY")     # <-- required by web_search
GOOGLE_CSE_ID           = get_metadata("GOOGLE_CSE_ID")      # <-- required by web_search
GEMINI_API_KEY          = get_metadata("GEMINI_API_KEY")     # <-- for Gemini image gen
ANTHROPIC_API_KEY       = get_metadata("ANTHROPIC_API_KEY")  # <-- for Claude

# Optional toggle for Responses API
OPENAI_USE_RESPONSES    = _truthy(get_metadata("OPENAI_USE_RESPONSES", "false"))

# --- OpenSearch ---
OPENSEARCH_HOST         = get_metadata("OPENSEARCH_HOST", "https://localhost:9200")
OPENSEARCH_USER         = get_metadata("OPENSEARCH_USER", "elastic")
OPENSEARCH_PASS         = get_metadata("OPENSEARCH_PASS")  # set via env/metadata
# default false because localhost TLS is self-signed
OPENSEARCH_VERIFY_CERTS = _truthy(get_metadata("OPENSEARCH_VERIFY_CERTS", "false"), default=False)

# --- Context scope flags (recommended defaults) ---
ALLOW_CROSS_CHANNEL_USER_CONTEXT = _truthy(os.environ.get("ALLOW_CROSS_CHANNEL_USER_CONTEXT", "true"), True)
ALLOW_CROSS_GUILD_USER_CONTEXT   = _truthy(os.environ.get("ALLOW_CROSS_GUILD_USER_CONTEXT", "false"), False)
ALLOW_CONTEXT_SEARCH_OTHERS      = _truthy(os.environ.get("ALLOW_CONTEXT_SEARCH_OTHERS", "false"), False)
