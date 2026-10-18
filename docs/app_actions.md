# Discord app actions and server controls

Right-click or long-press a Discord user profile (including MultiVAC) or a
message, then open **Apps**:

- **Character portrait $0.99** opens the repeatable character purchase
  and saved-image recovery. Choose character opens a subject-name form. The
  purchase keeps Sunburst 2.5 low quality at 1024 square and one saved PNG per
  purchase. Earlier one-time purchases stay recoverable.
- **Generate image** opens an editable description form. This is ordinary image
  generation with normal quotas, separate from the paid
  character product. Message actions prefill the description from the message;
  profile actions start with a blank description. Submitting opens a priced
  API/quality/size picker shared with `/generate_image` and chat image requests.
  Choosing an option starts generation. After 30 seconds without a selection,
  it uses OpenAI GPT Image Mini, low quality, 1024×1024 (about $0.005 output
  plus inputs), regardless of the saved tier highlighted in the menu. Cancel
  sends no image request. Only the requester can choose or cancel;
  merely opening it does not call the image provider.
- **Model defaults** opens your own workload and tier selectors, regardless of
  whose profile or message you selected. Only the requesting account can change
  that view.

All three actions operate on the account invoking them. Selecting another user
does not purchase for that account, edit their settings, or make their avatar an
image reference. The character action retains its explicit subject-name form.

Slash commands remain available: `/character`, `/generate_image prompt:...`,
and `/model_defaults`. Images and slash/menu settings responses are private.
Mention shortcuts reply in their originating channel, so a recovered paid
image and named choice can be visible to other channel members.

## Text invocation

Use an explicit mention or reply to MultiVAC:

```text
@MultiVAC $1 random plz
@MultiVAC $.99 Sonic
@MultiVAC model defaults image economy
@MultiVAC model defaults
@MultiVAC server settings
@MultiVAC server model chat economy
@MultiVAC server cap image balanced
@MultiVAC server throttle 50
@MultiVAC server channel off
@MultiVAC server channel on
```

`$1`, `$1.00`, `$.99`, and `$0.99` are aliases for the existing $0.99 SKU.
The bot offers Discord checkout if you do not own it, or returns your saved
image if you do. Typing a dollar amount never charges your account. A completed
image is reused; another shortcut does not buy or reroll it. The saved-image
response offers another checkout. An explicit new character choice after a
completed image opens the next offer without changing that saved image. A named selection
is fixed when the purchase is reserved. Ordinary mentions such as `imagine a
cat` retain their existing image route.

These shortcuts use explicit local parsing before intent classification. A
normal sentence containing a dollar amount does not become a purchase command.
The existing shutdown/silence check still runs before shortcuts.

## Active channels

Media app actions and paid shortcuts in a server require a channel where the
bot has previously answered a normal request. Eligibility persists across
restarts. Normal response completion records activity; historical channels can
also be recognized by inspecting at most 100 messages and fetching at most
eight referenced originals. Only server/channel IDs, activity time, and enabled
state are saved by this check. Purchase prompts, settings shortcuts, and denial
messages do not establish activity.

MultiVAC must still be installed in the server and able to view and send in the
channel. Threads have their own activity records and require thread-send access.
Disabled channels remain disabled even if the bot continues answering text
requests there. Re-enabling restores eligibility only where previous activity
exists; it does not create activity. Normal image/edit/video text routes also
respect a manager's channel disable setting.

DM commands retain private purchase recovery and normal quotas. Server policy
does not affect a user's DMs or other servers. Automatic paid-image delivery
remains a DM operation under personal/global limits. A cached image can be
downloaded without another model call. A server image maximum of Economy
blocks new Sunburst character generation and purchase offers in that server;
the buyer can still use DM recovery.

These gates check execution and purchase-button offers. Discord may still show
globally registered commands in inactive channels. Server managers can use
Discord's **Server Settings → Integrations** to control native command
visibility; the bot does not rewrite user-authorized command permissions.
See [Discord application command permissions](https://docs.discord.com/developers/interactions/application-commands#permissions).

## Server defaults, maximums, and stricter throttling

**Manage Server or Administrator permission is required at runtime** for both
slash and text server controls. App-owner status or guild-owner ID is not a
substitute. Ordinary members can change their own model preferences.

Examples:

```text
/server_settings workload:chat tier:economy max_tier:balanced
/server_settings workload:image max_tier:balanced
/server_settings throttle_percent:50
/server_settings channel_enabled:false
```

The optional channel argument selects another text channel in the same server;
otherwise the current channel is used. Workloads are chat, research, deep,
vision, image, edit, and video. All applies model changes to every workload.
Auto removes that server override. Defaults are used when a member's saved
choice is Auto; explicit personal tiers apply within the server maximum. A
maximum also caps Auto and more expensive saved choices without overwriting
personal settings. Model selectors show both the saved tier and effective tier
in the current server.

Maximum tiers apply to built-in primary response/generation models. Cheaper
maximums restrict explicit external providers and provider retries for that
workload, preserving the maximum even if the requester approves a fallback.
Internal classification, verification, safety, and support-tool models remain
operator controlled. Image options above the maximum are removed and checked
again before submission. Economy permits Mini low; Balanced adds Mini medium
and Sunburst low/medium. Quality/Auto allow all configured choices, including
Gemini. Video options above the maximum are removed and the
configuration is checked again after confirmation. The fixed paid product is
never silently switched away from Sunburst.

Throttle percentage accepts **1–100**, where 100 restores normal. It reduces
each action's existing shared server limit and a separate per-member limit
inside that server, rounding down with a minimum of one. It never raises base
limits, changes other servers, or clears existing counters. Normal personal and
global limits, premium rules, provider capacity, and provider quotas still
apply. fUnThrottled bypasses only bot-wide counters, so it does not bypass these
server caps. Existing premium multipliers apply before the percentage reduction.

Settings and channel eligibility use additive `guild_app_settings` and
`bot_channel_activity` tables in `conversation_history.db`. Include them in
backups and privacy handling. There is no automatic expiry. No extra API keys
or services are required. General image app actions make one OpenAI render
attempt and do not automatically switch providers or upgrade quality.
