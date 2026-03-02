# Multivac

Multivac is a Discord bot that routes chat and tool requests across multiple AI providers and local service modules.

Current runtime capabilities include:
- Discord mention/reply chat
- OpenAI, Gemini, and Claude chat paths
- Image generation, image editing, and image description
- Sora video generation flow
- Weather, stock, URL summarization, web search, and repo self-inspection tools
- Optional Elasticsearch/OpenSearch-backed memory

## Layout

The repo is organized into a few top-level areas:

- `main.py`
  Starts the Discord bot.
- `discord_bot.py`
  Main Discord event/controller layer.
- `config.py`
  Reads environment variables and metadata-backed configuration.
- `bot/`
  Discord-facing runtime helpers:
  chat routing, image routing, video routing, UI helpers, context assembly.
- `providers/`
  External provider integrations:
  OpenAI, Gemini, Claude, Stability, and Sora.
- `services/`
  Internal service layer:
  memory, database, search, weather, stock, URL, git, progress, and tool registry.
- `scripts/`
  Probe, inspection, and migration scripts.
- `dev/`
  Local verification and maintenance helpers.
- `docs/`
  Notes, walkthroughs, and reference docs.
- `assets/`
  Repo assets.

## Runtime Notes

The bot can run with partial configuration, but some features will disable themselves if keys are missing.

Common examples:
- no `STABILITY_KEY`: Stability image features are disabled
- no `GOOGLE_API_KEY` or `GOOGLE_CSE_ID`: Google CSE search is disabled
- no `OPENAI_API_KEY`: OpenAI-backed paths cannot run
- no Elasticsearch/OpenSearch server: memory falls back to disabled mode

If you do not want memory at all, set:

```powershell
$env:OPENSEARCH_ENABLED="false"
```

## Required Environment

Minimum for a basic Discord bot boot:

```bash
DISCORD_TOKEN=...
```

Common provider/tool variables:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
STABILITY_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
OPENWEATHER_API_KEY=...
OPENSEARCH_HOST=...
OPENSEARCH_USER=...
OPENSEARCH_PASS=...
OPENSEARCH_VERIFY_CERTS=false
OPENSEARCH_ENABLED=true
```

## Install

This repo currently runs fine with a normal pip-based install even if Poetry is not available:

```bash
python -m pip install -r requirements.txt
```

There is no generated `requirements.txt` committed right now, so in practice you will either:
- install from `pyproject.toml` with Poetry, or
- install the needed packages manually with pip

If you use Poetry:

```bash
poetry install
poetry run python main.py
```

If you use plain pip:

```bash
python main.py
```

## Run

PowerShell example:

```powershell
$env:DISCORD_TOKEN="your-token"
python main.py
```

## Current Warnings You May See

These are expected unless you configure the related feature:

- `STABILITY_KEY not set`
- `Google CSE credentials not fully resolved`
- `PyNaCl is not installed, voice will NOT be supported`

If you want voice support:

```bash
python -m pip install pynacl
```

## Smoke-Test Status

The refactored runtime currently passes:
- AST parsing across root modules, `bot/`, `providers/`, `services/`, and `scripts/`
- import smoke tests for the main runtime modules

The bot has also been started successfully with a Discord token after the refactor. The main remaining runtime dependency is whether optional external services like OpenSearch, Stability, or Google CSE are configured.
