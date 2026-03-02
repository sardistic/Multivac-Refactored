# Multivac

Multivac is a modular Discord bot with multiple chat providers, image and video generation flows, tool-calling, and optional long-term memory.

## Features

- Discord mention and reply based chat
- OpenAI, Gemini, and Claude chat routing
- Image generation, image editing, and image description
- Sora video generation and remix flow
- Weather, stock, URL summarization, search, and repo inspection tools
- Optional Elasticsearch/OpenSearch backed memory and timeline recall

## Current Layout

- `main.py`
  Boot entrypoint.
- `discord_bot.py`
  Thin Discord event adapter and top-level bot wiring.
- `config.py`
  Environment and metadata-backed configuration.
- `bot/`
  Discord runtime behavior:
  context building, intent dispatch, provider intent handlers, UI, and message parsing.
- `providers/`
  External AI and media integrations:
  OpenAI, Gemini, Claude, Stability, and Sora.
  Most `*_utils.py` files here are now compatibility facades over smaller provider modules.
- `services/`
  Internal app services:
  SQLite storage, memory client/query layers, tool specs/handlers/dispatch, search, weather, stock, URL, git, streaming, and progress helpers.
- `scripts/`
  Probe, inspection, and migration scripts.
- `dev/`
  Local verification and maintenance helpers.
- `docs/`
  Notes and reference docs.
- `assets/`
  Static project assets.

## Runtime Behavior

The bot is designed to boot with partial configuration. Missing keys disable only the related feature.

Examples:
- no `OPENAI_API_KEY`: OpenAI chat and OpenAI image paths are unavailable
- no `GEMINI_API_KEY`: Gemini chat and Gemini image paths are unavailable
- no `ANTHROPIC_API_KEY`: Claude path is unavailable
- no `STABILITY_KEY`: Stability image backend is unavailable
- no `GOOGLE_API_KEY` or `GOOGLE_CSE_ID`: Google CSE search is unavailable
- no OpenSearch server: memory auto-disables and the bot continues running

To disable memory explicitly:

```powershell
$env:OPENSEARCH_ENABLED="false"
```

## Environment

Minimum boot requirement:

```bash
DISCORD_TOKEN=...
```

Common optional variables:

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

This repo is defined by `pyproject.toml`. Use Poetry if you want dependency management from the repo metadata:

```bash
poetry install
poetry run python main.py
```

If you are using plain pip, install the runtime packages you need and run:

```bash
python main.py
```

## Run

PowerShell example:

```powershell
$env:DISCORD_TOKEN="your-token"
python main.py
```

## Expected Warnings

These warnings are normal unless you intend to use the related feature:

- `STABILITY_KEY not set`
- `Google CSE credentials not fully resolved`
- `PyNaCl is not installed, voice will NOT be supported`

For Discord voice support:

```bash
python -m pip install pynacl
```

## Validation Status

The current refactored codebase passes:

- `compileall` across root runtime files, `bot/`, `providers/`, and `services/`
- import smoke tests for the runtime modules

The main remaining runtime dependency is external configuration, not code structure.
