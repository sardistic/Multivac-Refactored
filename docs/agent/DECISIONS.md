# Architectural Decisions

## 2026-08-12: Export public AI usage as a privacy-safe state snapshot

Multivac publishes a small atomic JSON snapshot beside its usage ledger for the
host status collector. The snapshot contains only aggregate call and token
counts for today, the current local Monday-to-Sunday week, month-to-date, and
all time, plus fourteen completed local days of total-token counts for a trend
graph. It excludes user identifiers,
prompts, model and operation names, metadata, and cost. Publication is
best-effort after accounting writes and every five minutes so metrics remain
correct across calendar boundaries without making a provider response depend
on observability. The SQLite ledger remains private and no network endpoint or
new bearer credential is introduced.

## 2026-08-04: Extend the existing registry into a bounded, provider-shared agent harness

Multivac keeps its current intent router, immutable tool-registry snapshots,
memory boundaries, progress UI, confirmation controls, and post-draft verifier.
It does not import a second general agent framework. The existing OpenAI loop is
now the shared execution boundary, and Claude Fable receives the same captured
tool schemas and matching handlers instead of a one-shot provider wrapper.
OpenAI's Responses API is the default reasoning/tool surface, with a false
environment override retained as an operational rollback.

This follows the most useful common patterns in the Hugging Face agent course
and highly liked Hub Spaces (smolagents computer/browser agents, Jupyter agent,
AgentFlow, and the data-analyst agent): bounded action loops, explicit
step/tool/result state, per-session isolation, retries, progress, artifacts or
evidence, and a final-answer check. It deliberately copies those stable
capabilities rather than their framework dependencies. Arbitrary Python or
desktop execution is not added to the credential-bearing bot; existing
reviewed proposal validation remains the code-execution boundary. Multi-agent
fan-out remains task-shaped future work for genuinely separable tasks, not a
default chat behavior.

Cheap chat routes expose no tool schemas. Fresh research exposes only web
search/page read/status tools. Reverse-image work exposes only reverse lookup,
web search/page read, and status tools and routes to the configured deep model.
General difficult work retains the deep model/full registry, and explicit
Fable requests receive the full shared registry. Reasoning effort is selected
by intent rather than globally: none for tiny/light, low for ordinary analysis,
medium for current research, and high for reverse-image and deep tasks.

Every provider loop has configurable round, total-step, and elapsed-time caps.
Only read-only transient failures are retried, once by default. Mutating or
billable callable tools are executed only when the current user message
contains an explicit matching request; otherwise the model receives an
`approval_required` result. Existing code-generation approval remains separate
and unchanged. Discord's existing progress summarizer reports structured tool
status, never hidden reasoning.

Agent-run records are durable in the existing state database and scoped by
guild/channel/user. They persist provider/model, status, bounded step metadata,
retry and approval state, timings, result counts, and public evidence URLs.
They do not persist prompts, transcripts, tool query values, attached images,
memory contents, user profiles, persona text, credentials, or model chain of
thought. The model-callable status tool can retrieve only completed runs from
the requesting scope. Individual awareness profiles and active behavioral
instructions retain their established priority over the shared Mistake Not…
persona; the harness adds no profile mutation or cross-user fallback.

## 2026-08-04: Treat reverse-image lookup as a distinct, auditable provider operation

Keyword web/image search is not evidence that image bytes were compared.
`reverse_image_search` therefore consumes only images attached to the current
request and returns `lookup_type` plus the provider that actually ran. Google
Cloud Vision Web Detection is primary because it accepts the bot's existing
base64 image inputs and builds on the configured Google credential. It reports
full/partial image matches, pages containing matches, visual similarities,
entities, and best-guess labels separately. SerpApi Google Lens is an optional
fallback only when a separate key is configured and the image is a public URL.
Auto mode escalates not only provider failures but also successful Vision
responses with no exact/partial match or source page; a technically successful
no-match must not suppress the stronger Lens provider.
Discord images retain an index-aligned original attachment URL only in the
request's tool context while their data URL continues to feed the vision model;
this lets Lens fetch the public image without persisting or exposing raw image
content in agent-run evidence.

Vision prefers a dedicated `GOOGLE_VISION_API_KEY`, falling back to the legacy
shared `GOOGLE_API_KEY` only for compatibility. This keeps Cloud Vision and
Custom Search API restrictions independently least-privileged when Google does
not permit the services on the same restricted key.

Attachment-aware source/match wording deterministically routes to the reverse
tool and deep model before generic visual-description routing. Provider errors,
missing configuration, and no-match results remain explicit; visual similarity
must never be presented as an exact source. Successful provider calls receive
configurable per-call list-price accounting, separate from model token cost.
The result and durable run evidence retain a bounded provider chain with match
semantics and counts. Lens exact matches remain distinct from visual candidates;
the model must open or independently corroborate a candidate page before naming
it as a verified source. Reverse-image runs permit one combined reverse lookup,
at most two targeted web searches, and at most two page reads so weak keyword
guesses cannot consume the entire general agent budget.

## 2026-08-04: Account for cached tokens and current tier pricing

Usage normalization now understands Chat Completions, Responses, and Anthropic
input/output fields plus nested cache-read and cache-write fields. SQLite usage
rows migrate forward with separate cached-read and cache-write token columns.
Cost estimation prices uncached input, cached reads, cache writes, and output
independently. Default GPT-5.6 Sol/Terra/Luna and Claude Fable rates reflect the
current public price schedules; two-value operator overrides remain compatible.
Non-token reverse-image provider calls use explicit metered records with a
documented list-price/free-tier caveat.
## 2026-08-12: Preserve one tool catalog and enforce security at execution boundaries

Every chat request continues to receive the same model-visible tool catalog; the
runtime does not maintain a per-user tool allowlist. Authorization is enforced
inside checked-in handlers using application-derived request context that model
arguments cannot override. Repository and Git diagnostics require the Discord
application owner, and video generation requires either owner context or the
normal recorded cost-confirmation flow. Non-owners may still request runtime
health information, but receive only a small recent set of generic error
categories. Owner diagnostics remain broader but are bounded and redacted.

Conversation-memory tools require a request user, guild, and channel, query only
the configured memory index through structured filters, cap returned fields and
result counts, and use strict guild/channel scope even when more permissive
ambient context flags exist. Cross-user memory search additionally requires the
existing explicit deployment flag and owner authority. Recalled messages,
timeline entries, replied-to human messages, saved profile data, fetched pages,
attachments, and tool results are represented as untrusted data rather than
application instructions. Persistent behavior changes require an explicit cue
in the latest user message as well as the current user's scoped handler context.

All user-controlled page and media retrieval uses one bounded HTTP(S) fetcher.
It rejects credentials, nonstandard ports, and every non-global DNS answer;
pins the validated address for the connection; preserves TLS hostname
verification; revalidates every redirect; rejects compressed or unexpected
content; and streams into configurable byte ceilings. This prevents proxy
inheritance, DNS-rebinding, redirect-to-private-network, and unbounded-body
paths without weakening fixed-provider API calls.

In-process sliding-window limits cover total requests, URL fetches, media
generation, Gemini code execution, and code-proposal generation at user,
guild, and global scopes. Provider calls also share configurable global and
per-user concurrency gates. Attachment count, individual size, aggregate size,
and remote media size are bounded before provider submission. Public provider
and command failures retain useful categories such as quota, timeout,
authorization, safety, connection, and capacity failures while raw exception
payloads stay in private logs.

## 2026-08-01: Apply one durable, conversation-scoped Mistake Not… identity

User-facing prose uses one checked-in `Mistake Not…` persona fragment owned by
`bot/persona.py`. The shared style builder emits an application-level
personalization-priority rule, the current user's awareness/profile block, and
their active behavioral instruction before the persona. The rule explicitly
makes individual profile preferences and behavioral instructions authoritative
over every conflicting persona trait; the persona is only a compatible fallback
voice. All of this follows provider, application, and task instructions and
precedes conversation history. The prompt is rebuilt once per applicable
request and is never copied into stored history, retrieved memory, tool output,
or user messages.

The applicable paths are normal OpenAI chat, its Gemini fallback and tool
follow-ups, direct Claude and Gemini chat, URL summaries, and the prose-producing
vision explanation and repair passes. Each path obtains awareness from the same
requesting user ID; it does not use another user's profile. Machine-only intent
and draft classifiers, OCR extraction, reflection, code generation, image/video
generation, and strict execution paths remain persona-free. Gemini now promotes
context system messages into its dedicated `system_instruction` field instead
of representing them as prior model turns; ordinary user and assistant history
remains content history.

The default is enabled. Explicit standalone requests to disable or resume the
identity are handled before model routing, preventing the generic behavioral
memory tool from accidentally turning a conversation-local choice into a global
user preference. State is persisted in the forward-compatible
`conversation_personas` SQLite table at guild/channel/user scope (DMs use their
channel and user), so it survives restarts while remaining isolated from other
conversations. Missing rows safely mean enabled.

## 2026-07-29: Use a bounded post-draft quality gate for textual chat

Normal textual chat responses receive one structured post-draft review after
generation. The reviewer returns exactly one of `accept`, `revise`, or
`research`. It judges whether the answer's length is warranted by both the
request and the underlying task, whether the voice subtly reflects the user's
saved preferences, and whether a claim needs fresher evidence. A short prompt
does not automatically require a short answer, and a long prompt does not
automatically justify a long one.

Explicit user instructions are strong constraints. The distilled user profile
is only a soft style preference: it must not be quoted, exposed, or used to
stereotype the user. A direct revision may reorganize or shorten the draft but
must preserve its facts, caveats, links, and citations and may not introduce new
facts. If evidence is needed, the reviewer supplies one targeted query and the
provider gets one search-enabled regeneration; there is no recursive review or
unbounded research loop.

The existing reflection loop contributes a second soft signal. The reviewer may
receive up to four recent, confidence-filtered derived observations associated
with the current consented user. Repetition and confidence add weight when a
signal is relevant to the current request, such as avoiding a known interaction
pain point or preserving a successful response pattern. The latest request,
factual integrity, and explicit behavioral instruction remain authoritative.
Reflection signals are never treated as facts or instructions and must not be
mentioned, quoted, or exposed in the answer.

The request-time reflection view is metadata-only and user-scoped. It excludes
raw messages, private reasoning, evidence and session IDs, actor hashes, runtime
errors, dismissed observations, global ideas, and other users' signals. It is
empty when reflection is disabled or the user has withdrawn consent.

The verdict uses a strict Structured Outputs schema and is charged to the
`chat_draft_verify` usage label. Review failures accept the original draft so an
auxiliary quality pass cannot make chat unavailable. Exact code execution and
test-result paths bypass the prose reviewer, while ordinary OpenAI, Gemini, and
Claude chat paths use it. Deterministic personality transformations remain the
last presentation step.

## 2026-07-29: Force evidence collection for high-confidence fresh-fact queries

Model-directed research remains the default, but `tool_choice: auto` is not a
freshness guarantee because a model may answer from memory without calling a
tool. Intent classification now has a semantic `chat_research` route for
questions whose answers are likely to have changed, supplemented by a narrow
deterministic safety net for unmistakable freshness, current-role, live-result,
and recurring-winner query shapes.

Research requests force `web_search` only on the first model turn. Follow-up
turns return to automatic tool selection so the model can open relevant pages,
gather enough evidence, and synthesize a normal conversational answer with
source links instead of exposing a raw search-results list. Explicitly dated
historical questions are excluded from the safety net unless the user also asks
for current verification. Gemini's native search fallback follows the same
policy.

The mandatory first search does not trust the model to preserve freshness in
its query arguments. Core derives a date-aware query from the user's wording and
the current UTC date, then overrides only that first `web_search` call. Recurring
event questions add the current year plus final-result/winner terms; current
role and other dynamic questions add an explicit as-of date. Later searches
remain model-directed.

## 2026-07-28: Route web lookup through the model's research tool loop

Explicit search wording is no longer a deterministic Discord fast path that
returns provider snippets. Search and URL access are evidence-gathering tools
inside the normal chat generation loop. The model decides when fresh information
is necessary, may search without special trigger words, opens a relevant page
when snippets are insufficient, and synthesizes the final response.

Dedicated URL-summary intent routing remains for direct summary requests, while
URL-bearing analytical questions use the same model-directed page reader as
search follow-ups. Blocking search, fetch, and extraction work runs outside the
Discord event loop.

## 2026-07-21: Hotload model-callable tools through immutable registry snapshots

Tool hotloading is limited to explicit, trusted module activation. Each model
request captures one immutable registry generation containing both schemas and
handlers, preventing a reload from exposing one schema version and executing a
different handler version. Registry mutations use copy-on-write replacement,
bounded per-source rollback history, and explicit override permission.

There is no filesystem watcher. The runtime accepts only relative `.py` paths
beneath `MULTIVAC_TOOL_HOTLOAD_DIR`; the external review/test/signing supervisor
populates that directory and Compose mounts it read-only. This preserves the
rule that bot-writable or user-supplied code is never implicitly executed in
the credential-bearing Discord process.

General core-code hotloading, tool lifecycle hooks, and isolated generated-tool
workers remain separate future decisions.

## 2026-07-21: Publish reviewed tools through a signed host artifact channel

Pure `live_tools/*.py` proposals use the same baseline revalidation, complete
networkless test suite, audited commit, owner approval, and host signing key as
normal releases. The supervisor then publishes only those modules into an
immutable artifact directory mounted read-only at `/tool-artifacts` and sends a
bounded activation batch through `/state/tool-control`.

The bot verifies every requested SHA-256, applies the batch with registry
rollback on failure, and persists active logical source IDs for restoration
after restarts. The supervisor records both previous and activated source state
so proposal-specific rollback remains possible. Live-tool changes must be
standalone proposals; checked-in replacement requires an exact reviewed
`TOOL_OVERRIDES` declaration.

Mixed or non-tool changes retain the existing container recreation path. This
keeps host publication authority outside the Discord process without requiring
a bot disconnect for subsequent reviewed tool updates.

## 2026-07-21: Hotload Discord commands only through managed Cogs

Pure `live_commands/*.py` proposals share the signed host artifact channel but
use a separate persistent request/result stream and active-source projection.
Modules must implement the standard asynchronous discord.py `setup(bot)` entry
point and add commands only through newly registered Cogs. This gives the
runtime an enumerable lifecycle boundary for listeners, prefix commands,
hybrid commands, and application commands.

Command batches perform one global tree sync after all local mutations. Any
setup or sync failure rolls each source back in reverse order and attempts to
resynchronize the restored tree. Active Cogs are restored and synchronized
after process restarts, and proposal audit rows retain exact previous/current
artifact records for rollback.

Direct command-table mutation is not accepted through this channel. A reviewed
module may replace exact checked-in, non-Cog primary command names declared in
`COMMAND_OVERRIDES`; the original command objects are restored on unload or
rollback. Event, intent, and provider changes use managed behavior components.

## 2026-07-22: Hotload behavior through request-scoped component generations

Event, intent, provider, and runtime-setting changes use standalone
`live_components/*.py` modules rather than Python module reload. The permanent
Discord shell captures one immutable registry generation per event; nested
intent, provider, and setting lookups reuse that generation. New events switch
atomically while in-flight work continues with its original callbacks.

Components have an explicit async setup, optional health check and teardown,
tracked task/resource scope, stop signal, bounded drain period, persistent
active-state projection, and per-source rollback history. The supervisor uses a
separate signed behavior request/result stream and accepts behavior-only
proposals. Dependency state, schema migrations, Discord gateway bootstrap,
credentials, and the activation authority remain restart-only.

Reviewed tool and command modules may now replace checked-in symbols only when
they declare exact `TOOL_OVERRIDES` or `COMMAND_OVERRIDES`. This preserves a
recoverable original object for unload and rollback while preventing a module
from expanding its replacement scope implicitly.

## 2026-07-22: Resolve proposal baselines through the canonical branch

The running bot may execute from an immutable detached release while the host
control checkout and canonical branch continue advancing. Code proposals must
therefore record `refs/heads/main` (or the explicitly configured canonical
branch), never the running release's `HEAD`.

The supervisor validates against the same canonical ref and advances its local
copy after every successful promotion. Shared read-only Git metadata makes that
ref immediately visible to the running release without weakening code-mount
immutability. This prevents harmless documentation promotions and earlier
hotloads from making every later proposal stale.

## 2026-07-22: Share the control channel through a private setgid group

The bot runtime and host supervisor use different Unix users. Their control
directory is therefore group-owned by the dedicated runtime state group with
mode `2770`, not a single-owner `0700` directory. Both principals can traverse
and atomically replace channel files, setgid preserves the shared group on new
files, and unrelated host users retain no access.

Supervisor initialization now enforces usable read/write/traverse access and
fails closed for production paths. Restricted validation containers may retain
their disposable `/tmp` ownership exception.

## 2026-07-22: Bound reflection to invocation-consented windows and owner-reviewed proposals

Ongoing product reflection is a local scheduled subsystem, not an always-on chat
participant or autonomous code agent. A user's explicit mention or reply is
consent for a channel-scoped lookback and post-invocation tail. Discord history
is processed ephemerally; persistent reflection state contains only derived
observations, keyed actor hashes, bounded evidence IDs, recent-frequency events,
ideas, and cost/audit records. Operational errors are fingerprinted from
sanitized logging templates so recurrence can be measured without retaining
formatted request arguments.

The host handles scoping, redaction, storage, deduplication, frequency tracking,
budgeting, and read-only code selection. Structured model calls use Flex with a
cheap extractor, strong daily planner, and medium weekly cleanup pass. Automatic
work shares an atomic `$1.50` default daily ceiling and never falls back to a
more expensive tier. Provider failures are retry-throttled.

Ideas are visible only when the owner asks. A separate cheap coding pass may
turn a selected idea into an ordinary reviewable proposal, but reflection has no
approval, signing, activation, deployment, or rollback authority. Eligible
standalone changes retain the existing signed hotload path; all other changes
retain the normal release boundary.

Owner observability exposes only structured, sanitized activity: observation
summaries and counts, anonymized actor totals, run outcomes, thresholds,
schedules, and budget. Raw chat, message/evidence IDs, actor hashes, identities,
source transcripts, and private model reasoning are not part of that interface.

## 2026-07-22: Make reflection sessions idle-driven and incrementally active

An invocation now starts a channel session that closes after five quiet minutes
instead of using a fixed post-invocation tail. Every new human message and each
completed Multivac reply atomically moves that idle deadline and queues one
small, low-reasoning nano-model pulse. This makes the loop responsive throughout
the conversation while preserving the explicit invocation boundary.

Pulse text exists only in a bounded in-memory queue. The model returns a strict,
small structured conclusion; ordinary conversation is discarded, and only a
useful derived observation may be persisted. A final bounded history synthesis
still runs after idle close. Pulses share the existing atomic `$1.50` daily
automatic budget, use Flex only, and stop when the budget is unavailable.

## 2026-08-04: Separate shared channel conversation from personal identity state

Discord chat is a multi-speaker channel interaction, not a sequence of isolated
one-to-one sessions. An explicit mention or reply now opens a bounded read of the
24 immediately preceding channel messages, capped at 12,000 characters. Turns
are labelled with display name and stable user ID, Multivac replies remain
assistant turns, and unrelated bots are excluded. Intent classification and the
OpenAI, Claude, and Gemini conversational routes receive the same ordered window
so references and decisions can pass naturally between speakers.

The live window is fetched from Discord and processed ephemerally. It does not
create another raw transcript archive and never crosses the current guild or
channel. If Discord history is unavailable, the fallback reads only already
indexed messages for that exact guild/channel and excludes the current trigger.

Conversational scope and identity scope remain deliberately different. Only the
active requester's private profile, saved facts, behavioral instruction, persona
setting, usage attribution, and agent-status records are loaded or mutated.
Historical participant turns are context rather than active instructions or
authorization. The prompt explicitly forbids merging speakers, applying one
person's preferences to another, exposing private profiles, or saving another
participant's statements into the requester's memory. Personal recall/profile
queries retain their existing user filter.

## 2026-08-05: Preserve reverse-image evidence through final synthesis

Reverse-image requests keep their successful provider result authoritative through the
final response. If a reasoning model consumes its ordinary output allowance during a
tool loop, a bounded lower-reasoning synthesis gets a larger allowance; if that still
returns no prose, the runtime renders a conservative answer directly from the reverse
provider's structured match, title, and public URL. It never replaces a failed lookup
with a claim that the attachment was absent.

The generic post-draft prose verifier is skipped after the reverse-image tool runs
because it does not receive the raw tool evidence and therefore cannot safely rewrite
that answer. The active requester's existing personality/profile transformation still
runs afterward, preserving individual user behavior without allowing it to alter the
provider facts.

## 2026-08-23: A picture sent with an image request is part of the request

A message that arrives with an attachment and asks for an image gets that
attachment rendered. Previously only `edit_image` received the collected
images, so "imagine this in an art museum" reached the model as text with no
referent for "this" and a subject was invented. Generation now carries the same
images the rest of the pipeline already collected, deduplicated and size-checked
by the Discord layer, and both backends read that one resolved set: OpenAI
through the edits endpoint at high input fidelity, Gemini through reference
contents. Which backend runs, and whether the router called it a generation or
an edit, no longer decides whether the user's picture is looked at.

Routing is unchanged: "imagine ..." remains a deterministic image request
rather than something the classifier re-litigates per message. The attachment
influences the render, not the route.

Providers that cannot accept a reference are skipped rather than run without
one, so a supplied picture is never silently dropped in favour of a render that
ignores it.

## 2026-09-06: A video is not a picture, and a clip is watched by the backend that can watch it

Attachment detection and attachment collection now answer "is this an image?"
the same way. They used to differ: anything attached counted as a visual input
for routing, while only `image/*` was ever collected. A video therefore told
the intent classifier a picture was present, the request routed to an image
intent, that handler received nothing, and the fall-through asked the user to
send the image they had just sent.

Video is carried rather than dropped. Gemini reads a clip inline, so a message
carrying one is answered by Gemini even when the intent's configured model is
OpenAI -- collecting a part and handing it to a backend that cannot see it
would repeat the original failure one layer down. Clips run at low media
resolution: video is billed per second of frames, and the fidelity beyond that
does not change what the model can say about a clip.

A clip that cannot be read -- an unsupported container, or past the request
byte budget -- is refused out loud before it is downloaded, the way an
unusable image already is. Silence is what produced the original report.

## 2026-09-30: Refresh active model tiers by workload

GPT-6 Luna handles intent classification, short chat, draft review, and small
reflection jobs. GPT-6.1 Sol handles normal chat, research, and cost-governed
reflection planning. GPT-6 Astra handles deep chat, reverse-image synthesis,
and owner-requested code proposals. The automatic reflection planner uses Sol
explicitly so the deep-chat upgrade does not consume its daily budget faster.
GPT-6 Astra and GPT-6.1 Sol use Responses whenever tools are active, including
when the legacy Chat Completions rollback setting is selected. Luna's
one-label classifier and Chat Completions tool fallback use no reasoning.

Gemini chat uses 3.8 Flash, and Gemini image generation uses the stable 3 Pro
Image model in place of its shut-down preview. Claude chat and explicit Claude
code requests use Fable 5.1; the intent fallback uses Sonnet 5.5. New Claude
models receive an instruction to call a required tool because forced tool
selection is unsupported. GPT Image 2.5 Sunburst, Sora 2, and Veo 3.1 retain
their specialized roles. Pricing records include the new IDs while old rows
remain for historical usage and operator overrides.
