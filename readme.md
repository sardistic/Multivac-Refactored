# Multivac

Multivac is a modular Discord bot with multiple chat providers, image and video generation flows, tool-calling, and optional long-term memory.

## Features

- Discord mention and reply based chat
- OpenAI, Gemini, and Claude chat routing
- Gemini text, image, multimodal chat, code execution, Google Search, and reference-image flows
- Image generation, image editing, image description, and multimodal file analysis
- Sora and Veo video generation, including image-to-video references and Sora remix flow
- Reply-aware video prompts that can turn a replied-to message into the generation brief
- Weather, stock, URL summarization, web search, genuine reverse-image lookup, YouTube transcript, and repo inspection tools
- Responses-first OpenAI orchestration plus the same bounded tool loop for Claude Fable
- Durable, user-scoped agent-run status with step/retry/approval/evidence audit data
- User facts, saved behavioral instructions, user-awareness profiles, and time-passage context
- Default `Mistake Not…` conversational identity with durable conversation-scoped opt-out and resume controls
- Owner diagnostic helpers for redacted bot log reads and tool listing
- Rate-limited live progress with breathing scan frames, honest estimated
  completion, phase details, and provider/job summaries
- Optional Elasticsearch/OpenSearch backed memory and timeline recall
- Explicit, versioned hotloading for trusted model-callable tool modules
- Signed, rollback-capable hotloading for reviewed Discord command Cogs
- Lifecycle-managed hotloading for reviewed events, intents, providers, and runtime settings
- Opt-in, invocation-scoped reflection that detects recurring pain points and runtime errors

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

## Runtime Behavior and Tool Functions

The bot is designed to boot with partial configuration. Missing keys disable only the related feature.
When tool-calling is available, current callable functions include:

- `web_search`
  Search the web through the configured search backend. Its `image` flag is a
  keyword image search, not reverse-image matching.
- `reverse_image_search`
  Submit an attached image to Google Cloud Vision Web Detection. When Vision
  completes without an exact/partial match or source page, auto mode escalates
  public image URLs—including the original URL retained for a Discord
  attachment—to SerpApi Google Lens. Results identify the complete provider
  chain and separate exact matches from visual source candidates.
- `get_agent_run_status`
  Report completed model/tool runs for only the requesting guild/channel/user,
  including actual tools, retries, approvals, timing, and public evidence URLs.
- `get_weather`
  Resolve current or ranged weather requests.
- `get_stock_quote`
  Fetch a real-time stock quote.
- `summarize_url`
  Fetch a URL, extract the main text, and return a condensed summary payload.
- `get_youtube_transcript`
  Fetch a transcript for supported YouTube URLs.
- `search_memory`
  Search scoped OpenSearch-backed conversation memory, with temporal recall support.
- `remember_fact` / `forget_fact`
  Store or remove user-scoped remembered facts.
- `update_behavioral_instruction`
  Update the user's active behavior instruction projection.
- `generate_sora_video`
  Queue a Sora video job from a prompt, optionally using available image inputs.
- `read_own_logs`
  Owner diagnostic helper that reads recent bot logs with credential-like values redacted.
- `list_available_tools`
  Return the active tool names and descriptions.
- `git_recent_commits`
  List recent repository commits.
- `git_commit_diff`
  Read a commit diff by SHA.
- `git_read_file`
  Read a repository file.
- `git_search_code`
  Search repository code.
- `git_search_history`
  Search repository history.
- `git_file_list`
  List repository files.
- `git_find_api_calls`
  Find provider API call sites.
- `git_repo_info`
  Return repository metadata.

OpenAI uses the Responses API by default so current reasoning models can reason
and call tools on the supported surface. Set `OPENAI_USE_RESPONSES=false` only
as an operational rollback. Reasoning and tool exposure are task-shaped:
tiny/light/standard chat carries no tool schemas; current-information research
gets the web reader/search set; reverse-image work gets the reverse/search/read
set and the deep model, with one reverse lookup and at most two targeted web
searches; difficult general analysis gets the deep model and the
full registry. Explicit Claude Fable chat uses the same immutable registry
snapshot and bounded executor rather than a separate hard-coded search path.

Every tool loop has independent round, step, and elapsed-time caps. Read-only
transient failures retry once. State-changing or billable model tools require
an explicit matching request in the current user message. The existing
post-draft verifier remains the final answer check. Durable traces contain
orchestration metadata and public evidence links, not prompts, transcripts,
attached image bytes, user profiles, memories, credentials, or private model
reasoning. The trace/status scope follows the same guild/channel/user boundary
as conversation personalization.

Chat context is channel-aware rather than pretending every Discord exchange is
one-to-one. When a user mentions or replies to Multivac, the bot reads a bounded
window of the immediately preceding channel conversation (up to 24 messages and
12,000 characters), labels every human speaker by display name and stable user
ID, excludes unrelated bots, and supplies the same shared context to intent
routing, OpenAI, Claude, and Gemini. This live window is processed ephemerally;
the bot's existing indexed trigger/reply history is only a fallback if Discord
history cannot be read. The window never crosses the current guild/channel.

Shared conversation does not mean shared identity. Only the latest speaker's
profile, saved memories, behavioral instruction, and persona setting are loaded.
Earlier participant messages help resolve topics, references, and what Multivac
said to somebody else, but they are not active instructions or authorization and
are never written into the current requester's personal memory.

Normal user-facing prose uses the compact `Mistake Not…` identity by default.
The identity changes voice only; it does not change safety rules, tools, memory,
model routing, or structured-output behavior. Each user's saved awareness
profile and active behavioral instruction remain user-scoped and take
precedence over the default persona whenever they conflict. A user can mention
or reply to the bot with `drop the persona`, `answer normally`, or
`disable Mistake Not` to use a neutral voice in that conversation. `resume the
persona` or `enable Mistake Not` restores it. The setting is stored per
guild/channel/user (or DM/channel/user) and survives restarts without affecting
another conversation.

## Versioned Behavior Changes

Users can draft and audit personal behavior changes without immediately changing
the running bot:

```text
/behavior_propose Always answer in terse technical prose
/behavior_show 1
/behavior_activate 1
/behavior_history
/behavior_rollback
```

Use `/behavior_propose --clear` to draft a return to baseline behavior. Drafts
are user-scoped and inert until activated. Activation updates the existing
behavior-instruction projection, so all current chat/provider paths continue to
apply the active version. Rollback restores the activated version's parent, or
the local baseline when there is no parent.

This first phase changes prompt-level behavior only. It does not permit users to
edit Python, execute shell commands, restart the bot, or modify the audit system.
See `docs/self_modification.md` for the local architecture and the proposed
reviewed-code-change phase.

Trusted tool modules can also be activated, unloaded, and rolled back without
disconnecting the bot. This is an explicit owner operation over a configured
read-only artifact directory, not an automatic writable-directory watcher. See
`docs/tool_hotloading.md` for the module contract and security boundary.
Reviewed Discord Cogs use the parallel contract in
`docs/command_hotloading.md`.
Reviewed behavior components use request-scoped generations and lifecycle hooks
documented in `docs/behavior_hotloading.md`.

The optional reflection worker opens a bounded channel window only after a user
explicitly mentions or replies to Multivac; that invocation is consent for the
bounded session. Each new message in the active channel extends the session;
a tiny budget-gated nano reflection runs in the background for each message,
and the session receives one final synthesis after five quiet minutes. It can
inspect a short lookback, detect recurring sanitized runtime errors, and turn
corroborated observations into owner-visible improvement ideas. Surrounding
Discord text is processed ephemerally; the reflection store retains derived
observations, hashed actor references, message IDs, frequency counters, ideas,
and cost records rather than a second raw-chat archive.

```text
/reflection_off
/reflection_status
/reflection_activity
/reflection_ideas
/reflection_propose 3
```

The last four commands are owner-only. `/reflection_activity` exposes sanitized
derived observations, model-run outcomes, thresholds, and next eligible run
times without transcripts, identities, or private reasoning. Proposal generation remains inert until
the ordinary static validation and explicit owner approval pipeline succeeds;
eligible standalone tool, command, or behavior changes can then use the signed
hotload channels. See `docs/reflection.md` for privacy, model tiers, budgeting,
and operator settings.

The owner can also record and statically review executable-code proposals with
the `/code_*` commands. Patches are checked against their committed baseline in
a disposable local snapshot. Approval records a review decision only; it never
applies or executes the patch and never restarts the bot.

`/code_generate <proposal_id>` can generate the unified diff from bounded,
policy-allowed repository context. Generation never grants approval: the owner
must inspect the attached diff and explicitly approve it. A separate Debian
supervisor tests, signs, activates, monitors, reports, and rolls back releases.

Examples:
- no `OPENAI_API_KEY`: OpenAI chat and OpenAI image paths are unavailable
- no `GEMINI_API_KEY`: Gemini chat and Gemini image paths are unavailable
- no `ANTHROPIC_API_KEY`: Claude path is unavailable
- no `STABILITY_KEY`: Stability image backend is unavailable
- no `GOOGLE_API_KEY` or `GOOGLE_CSE_ID`: Google CSE search is unavailable
- no Cloud Vision Web Detection access on `GOOGLE_VISION_API_KEY` (or legacy `GOOGLE_API_KEY` fallback): the primary reverse-image provider is unavailable
- no `SERPAPI_API_KEY`: Vision still runs, but no-match public images cannot escalate to Google Lens
- no `OPENWEATHER_API_KEY`: weather lookup and weather-widget flows are unavailable
- no `GEMINI_API_KEY`: Veo video paths are unavailable
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
GOOGLE_VISION_API_KEY=...
SERPAPI_API_KEY=...
OPENWEATHER_API_KEY=...
OPENSEARCH_HOST=...
OPENSEARCH_USER=...
OPENSEARCH_PASS=...
OPENSEARCH_VERIFY_CERTS=false
OPENSEARCH_ENABLED=true
REFLECTION_ENABLED=false
REFLECTION_DAILY_BUDGET_USD=1.50
OPENAI_USE_RESPONSES=true
AGENT_TRACE_ENABLED=true
AGENT_MAX_TOOL_ROUNDS=4
AGENT_MAX_TOOL_STEPS=8
AGENT_MAX_TOOL_SECONDS=45
```

Reverse-image provider call estimates use configurable list-price values before
free tiers: `GOOGLE_VISION_WEB_COST_PER_CALL_USD` defaults to `0.0035`, and
`SERPAPI_LENS_COST_PER_CALL_USD` defaults to `0.025`. Token accounting separates
ordinary input, cache reads, cache writes, and output for the current GPT-6
and Claude Fable rates. Override price tables with `OPENAI_PRICE_JSON` when a
provider changes rates or an account has custom pricing.

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

:)
