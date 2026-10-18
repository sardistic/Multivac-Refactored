# Architectural Decisions

## 2026-10-17: Self-hosted Discord owner portal and approved server admission

Keep the existing public page design and deploy an independent aiohttp website
alongside the bot, using the existing production image. Discord authorization
code sign-in requests only identity and guild list. Resolve owner authority from
Discord's application metadata; team-owned applications use the real team owner,
not the synthetic application-owner account. Request submission freshly checks
Manage Server or Administrator permissions. Owner-only reviews are durable,
audited and protected by Origin plus session CSRF checks. Cookies are Secure,
HttpOnly and SameSite=Lax; access tokens and one-hour sessions stay in memory.

An explicit one-time import preserves existing installations. New servers need
approved access; the bot blocks messages/application commands before processing
and leaves unapproved joins or revoked servers within one minute. Read errors
deny work but defer removal until state is readable. Never auto-approve guilds
on startup. Requesters retrieve decisions and invites on the site; no automatic
Discord message or email is introduced. Invite permissions exclude Administrator.

The live usage ledger uses WAL, so file-only read-only database mounts cannot
provide reliable current reporting. The bot instead atomically exports private
aggregate reports every minute for 1/7/30/90-day periods. The website mounts only
these aggregates and the existing public snapshot read-only, plus its own
private request database. It receives no provider keys, raw usage ledger,
conversation, memory, purchase or reflection database. Reports separate DMs and
unattributed/background usage, expose estimated costs as estimates, and label
calendar windows and UTC daily buckets. Authentication, admission, reporting and
website files are protected from ordinary generated-code proposals. Host routing
and deployment facts remain in the external infrastructure runbook.

## 2026-10-14: Persistent shop access and explicit storefront routing

The five-minute requester-bound navigation stranded users at expired messages
and left ordinary shop requests to a model without storefront knowledge. Shop
navigation now uses a stable persistent component ID, registered at startup;
any clicker opens their own verified ephemeral shop. A narrow interaction listener
recovers legacy random-ID Open shop buttons only on this bot's own messages and
never handles the new registered ID. Browsing carries no purchase or redemption
authority, so binding public navigation to the original capped user is unnecessary.

Add the native /shop alias and route exact plain-text shop/premium commands,
explicit storefront questions and link-again replies to bot-owned shop notices
before quota/classifier work. Reuse the existing shop presentation and media
eligibility gates; text listings are channel replies, interaction listings remain
private. Unrelated link requests and general questions about shops retain chat.
No entitlement, quota, database or provider behavior changes are required.

## 2026-10-13: Offer the existing shop from throttle notices

Discord throttle replies keep their retry delay and add a requester-bound
Open shop button. Clicking opens the existing ephemeral premium command without
another request-limit check, so capped users can reach verified Discord checkout.
It never buys, consumes, grants access, or retries work automatically. Global cap
copy suggests fUnThrottled; other scopes explicitly retain personal/server caps.
A disabled subscription is not advertised, and a missing shop preserves the
original notice. Provider failures and capacity limits remain separate errors.
Paid-product quota failures carry their limiter decision to the Discord layer,
preserving the purchase and its unspent render attempts.

## 2026-10-11: Exact GFL2 image prompt and direct image replies

The operator's reference result came from the ordinary prompt `[character] as a
gfl2 3d character model`. Paid rendering now uses exactly that template through
the text generation endpoint. Remove catalog portrait inputs and the added
full-body, neutral-background, no-text and no-UI restrictions that steered the
output toward a generic portrait. Preserve Sunburst low quality, square size,
selection, entitlement ownership, consumption, quotas and saved-result guards.

Replies show the PNG and a short subject caption; source links and download
instructions are removed. The optional purchase selector remains supported.
The operator requested correction of their last completed image; a one-off
audited repair may replace that saved output without consuming another purchase.
This does not grant general rerolls or change automatic purchase processing.

## 2026-10-11: Repeatable consumable character portraits with legacy recovery

The operator replaced new character checkout with published consumable SKU
1558875787321679992 at $0.99. Each entitlement gets one separately frozen
selection and saved Sunburst PNG. The previous durable SKU 1558749059609665566
is unpublished but remains accepted for recovery and is never consumed. No
existing entitlement, image, selection or retry state is migrated to a new SKU.

Commit ready image bytes before calling Discord consume. A consumption error
keeps the ready state; restart/retry performs only consumption confirmation.
Consumed purchases without local ready bytes are denied. An additive confirmation
column avoids repeated successful consumes, while actual ownership, validity and
refund checks remain authoritative. Consumption never grants another render for
the same entitlement. Existing limits and fixed Sunburst low 1024-square settings
remain. Refund races revoke bytes before delivery.

Commands prioritize unfinished, unused, then newest saved purchases. Captions
include a purchase selector for older images; runtime ownership checks protect
it. A saved response offers new checkout where allowed, and a post-completion
subject choice targets only the next purchase. Gateway and REST processing watch
both SKU IDs. Product descriptions now say one image per purchase; local legal
draft changes do not publish the public legal site.

## 2026-10-11: Image edit parameters depend on model support

The first live paid reference render failed with OpenAI's
invalid_input_fidelity_model error: Sunburst does not accept input_fidelity.
Send high input_fidelity only for the known supporting legacy GPT Image 1 and
1.5 models. Streaming and single-shot edit calls use the same capability rule;
Sunburst, Flare, GPT Image 2, Mini and unknown models omit the optional parameter.
Reference files remain attached, and requested model, quality and size remain
unchanged. No provider switch or extra generation attempt is added. The paid
render regression exercises the real image provider path with a rejecting SDK
mock, rather than mocking away the request construction.

## 2026-10-11: Expose app actions in profile and message menus

Discord user/profile and message context menus use separate command types.
Register the existing three action labels under both types so the bot's profile
Apps menu exposes character checkout/recovery, ordinary image generation, and
personal model defaults. Each action uses the requester, never the selected
account. Profile image actions open a blank editable prompt; message actions
continue prefilling the selected message. Both routes share the existing active-
channel checks and requester-bound form, with no model call until submission.
Paid-product ownership, generation limits, server policy and manager permissions
remain enforced by the existing flows.

## 2026-10-11: Plain-text Unattractive slash syntax and checkout recovery

A plain /unattractive message previously entered the legacy prefix-command
handler, although Unattractive is registered as an application command only.
Route its exact known text forms, including action:VALUE, through the existing
app text handler before legacy dispatch. Other prefix commands retain their
handler. Unknown text commands return command-picker guidance instead of an
internal CommandNotFound error. Purchase confirmation with no unused entitlement
opens the verified native checkout offer directly, without consumption or grant.
These checkout/control messages do not establish historical channel activity.

## 2026-10-11: Consumable Unattractive server-wide generation context

The owner selected $1.99, everyone in the selected server, and both chat and
visual results for SKU 1558762748324876308. Discord verified a consumable SKU,
initially unavailable and available during live post-deployment verification.
Explicit requester confirmation selects the
server before consuming a purchase; no timer or guild is inferred from a user
entitlement gateway event. Each purchase adds a 600-second persisted window.
Pending recovery freezes the target guild; refunds leave durable tombstones.
Managers can stop active/queued time, and normal model/quota policies remain.

Separate tables in premium.db preserve existing SKU mappings and redemption
behavior. Public response provider dispatch receives an additional system style;
image/edit/video submission paths receive visual context. The effect is resolved
from request-local guild identity at submission, with no cross-server/DM scope,
no model change, and no classifier/tool/control modifier. Facts and task content
remain accurate. Cached paid images and automatic DM renders remain unchanged.
Effects expire without a background timer; gateway plus periodic REST checks
reconcile revocation. Only transaction metadata is added to retained storage.

## 2026-10-11: App actions, guild policy, and active-channel purchase access

Discord message context menus expose character checkout/recovery, ordinary
image generation, and personal model selectors. The $1 text alias targets the
existing $0.99 durable SKU; it never charges an account or rerolls a completed
image. Media app actions and character purchase offers require persistent
channel activity, bounded historical reply evidence, and current bot access.
Manager disable markers survive later normal activity. DM recovery remains
available; automatic purchase delivery remains scoped to DMs.

The owner clarified that server controls belong to members with Manage Server
or Administrator permission, rather than only the guild owner. Runtime checks
protect both text and slash controls. Additive SQLite tables store per-workload
defaults/ceilings, a 1–100% throttle reduction, and channel eligibility. Request
context variables isolate guild model resolution without changing saved personal
choices. Cheaper ceilings constrain primary built-in routes, explicit external
providers, retry consent, and selectable video configurations; operator-owned
classifier/verifier/tool models retain their existing configuration. Economy
image ceilings block new Sunburst paid jobs in that server without substituting
the product's model; cached downloads and DM recovery remain possible.

Server reduction applies to existing guild quotas and a separate member-in-guild
counter inside the same atomic limiter check. Cross-server personal/global
counters, premium multipliers, and provider capacity are retained. Global premium
bypass never bypasses server restrictions. General image actions use one render
attempt without an automatic provider switch. Native command visibility remains
under Discord integration permissions; runtime gates do not hide global menus.

## 2026-10-11: One saved character portrait per durable Discord purchase

The owner selected $0.99 for one rendered GFL2-inspired character image using
SKU 1558749059609665566 and explicitly requested Sunburst 2.5. Authenticated
inspection verified a durable SKU, so it is never consumed and repeated downloads
reuse the completed PNG. A separate private SQLite ledger stores the frozen
identity, image, leases, bounded provider attempts, and refund tombstones. Gateway
and REST discovery trigger delivery; closed DMs retain command recovery. Entitlement
ownership, application, SKU type, scope, validity, and deletion are rechecked.

Kitsu supplies free random identities from a catalog verified at 105,906 records;
named pre-purchase choices support subjects outside it. Sunburst low quality at
1024 square, downscaled reference inputs, and no chat/classifier calls bound cost.
Successful outputs are saved before delivery and never rerendered for download.
Source, rate-limit, and provider-capacity failures retain the purchase without
spending a provider attempt. The dedicated paid product uses shared provider
capacity directly to keep its promised model fixed across behavior overrides.
Failed image calls require an explicit retry, capped at three attempts;
no automatic provider switch or quality upgrade is authorized by this product.
The source count is over 100,000, not claimed to cover all possible characters.
The SKU was unavailable during initial inspection and available during deployment
verification. Publishing remains a Developer Portal action. This product changes
no subscription/donation benefits.

## 2026-10-11: Requester consent for Gemini retries and personal workload tiers

Failed OpenAI chat, vision, image generation, and image editing now propose a
Gemini retry instead of immediately sending the request. Video-chat rerouting
to Gemini also asks first. Only the requesting Discord account can accept;
cancellation, prompt delivery failure, and two-minute expiry deny the retry.
Image-provider services return a deferred `ProviderFallback` proposal so the
Discord layer waits for approval after releasing provider concurrency. Reference
inputs remain attached to the approved retry. Explicit Gemini image failures
also require approval before switching to OpenAI.

The self-service `/model_defaults` command persists explicit Auto, Economy,
Balanced, or Quality choices per account and workload in an additive SQLite
table. Text/vision choices map to Luna, Sol, and Astra; images/edits map to Mini
low or Sunburst medium/high. Research tools remain available, explicit provider
requests take priority, and inferred personalization cannot change these choices.
Same-model image retries preserve model and quality; saved text tiers disable
the prior automatic empty-answer upgrade. Video choices only preselect Sora
configuration and still require the existing cost confirmation. Auto restores
existing behavior. Weather images honor an explicit saved image tier.

SQLite connection context managers now explicitly close connections after
commit/rollback. Restart-persistence tests exposed the previous connection leak
as locked temporary databases on Windows.

## 2026-10-11: Preserve Claude image answers through presentation

The Claude route already passes attached and replied-to images as Anthropic
image blocks. Its generic post-draft reviewer receives only the latest text
request and draft, so it can replace a visual answer with a false request to
attach the image. Requests containing actual Claude image blocks now preserve
the provider's answer without this text-only rewrite. This follows the existing
main chat policy for grounded reverse-image tool answers. Text-only requests
retain draft review and research repair. Provider tools, safety constraints,
and final explicit personality overrides continue to apply.

## 2026-10-11: Connect actual fUnThrottled and support products

The owner replaced the earlier proposed offerings with fUnThrottled ($4.99),
Donate ($2.99), and Donate More ($9.99). Authenticated Discord SKU lookup verified
fUnThrottled as a user subscription and both donations as consumables. The owner
explicitly selected support-only donations with no throttle benefit and a
subscription bypass of bot-wide counters only. Its buyer still has personal
and guild limits. Requests continue contributing to aggregate counters; other
users receive no exemption. Provider capacity, provider quotas, media constraints,
and existing permission/safety/approval checks remain enforced.

The three supplied IDs are fresh-install defaults, verified against Discord
before access or purchase buttons are enabled. Subscriptions apply automatically
through verified gateway events and REST reconciliation; ended, deleted, foreign,
guild-scoped, or stale entitlements do not grant access. Donations reuse durable
confirmed consumable recording/recovery without creating passes or reset epochs.
The earlier settings schema migrates without losing saved mappings or records,
and existing saved owner configuration takes precedence over defaults.
Legacy guild/pass/reset mappings remain optional and disabled by default. The
additional unpublished Server Goon SKU is not configured because its benefits
and price were not selected. Actual checkout prices remain portal-managed.

## 2026-10-11: Initial premium proposal (superseded by actual products above)

The owner's selected products are a $10/month guild subscription, a $5
consumable 30-day personal pass, and a $1 personal throttle reset. A personal
pass is manually renewed because Discord currently allows one subscription
scope per application and the owner chose guild subscriptions. Both bonuses
default to 2x; they multiply to 4x personal limits inside a subscribed guild.

The existing central rate limiter remains the enforcement boundary. Guild
bonuses raise member and guild limits in that guild; personal passes raise the
user limit across conversations. Global limits, provider capacity, and all
permission/safety/approval controls remain enforced. Resets clear only the
buyer's personal buckets, with durable increasing generations preventing replay.

A managed Discord Cog exposes native SKU buttons, status, explicit confirmed
redemption, and owner-only SKU listing/configuration. SKU creation, pricing,
and publication remain Discord Developer Portal operations. Actual SKU types,
application ownership, and guild flags are validated before configuration or
redemption; subscription-group IDs are not accepted.

Private SQLite state reserves a consumable entitlement before Discord consumes
it. The local grant and applied marker commit together. Ambiguous API outcomes
are recovered through the reserved entitlement, never by issuing a second
unrecorded grant. Passes use fixed sequential 30-day periods; refunding one
revokes that period without moving later grants. Gateway changes and periodic
REST reconciliation control subscriptions and pass revocations. Bonuses require
recent verification, expire locally after five minutes without it, and are
re-verified after restart. The design retains the single-process throttle model.

## 2026-10-04: Enforce explicit silence before Discord response work

Standalone shutdown and stop-replying commands terminate the current message
path before rate-limit notices, progress UI, intent classification, and provider
generation. These explicit commands are enforced locally, independently of the
model classifier's availability or verdict. Full matching keeps quoted language,
negations, technical questions, and commands targeting another system out of
this local rule. Broader wording retains the existing model-based classifier.

Normal Discord requests carry their completed gate check into dispatch so the
decision is evaluated once. Direct dispatch callers retain a defensive gate.
Silence applies only to the current request; it does not terminate the process
or persist a mute. TypeSafe's Jev is a separate decision model; this correction
does not add or activate a TypeSafe provider integration.

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
