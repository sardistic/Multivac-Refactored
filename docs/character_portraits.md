# Purchased GFL2 character images

The active SKU `1558875787321679992` belongs to MultiVAC application
`1087777433459118100`. Authenticated Discord inspection on October 11, 2026
verified its name as `random gfl2 3d character model`, consumable type (`3`),
and checkout availability. The operator's listing shows **$0.99 per purchase**.
Checkout supplies the actual localized price; the SKU API does not return prices.
The earlier durable SKU `1558749059609665566` is no longer available for new
checkout, but its existing purchases remain recoverable.

The delivered artifact is a PNG showing a character in a GFL2-inspired 3D game
render style. It is not a downloadable mesh or an official game asset. Each
verified entitlement earns one completed saved image; later downloads return
that image. After saving the completed image, the bot consumes the new SKU's
entitlement to allow another purchase. Each new purchase gets its own selection
and saved image. No throttle benefit is included. The earlier durable SKU is
never consumed or converted into an additional render.

Use `/character` before buying to view the offer and native purchase button.
`/character character_name:NAME` saves any named subject before purchase;
`/character character_name:random` restores random selection. A selection is
fixed when the purchase is reserved. A named subject does not need a catalog
entry; fidelity depends on the image model's knowledge of that subject.
Without a named choice, purchases use a random catalog character.

The **Apps → Character portrait $0.99** profile or message action opens the same flow.
`@MultiVAC $1 random plz` is a shortcut to this $0.99 SKU, not an immediate
charge or another render. Server purchase offers and media app actions require
previous bot activity in that channel and respect server-manager restrictions;
DM recovery remains available. See [app actions](app_actions.md).

`/character` prioritizes an unfinished purchase, then an unused purchase, then
the most recent saved image. Replies show the image with a short subject caption.
The optional `/character purchase_id:ID` selector remains supported for recovery
of a known older purchase. Selecting another member's purchase is
denied. Saved downloads offer the next checkout where the channel and server
model maximum allow it. Choosing a new name after completion changes only the
next purchase's choice, not any previously saved selection.

Purchase gateway events generate and DM the image automatically. A two-minute
REST reconciliation also discovers purchases missed while the bot was offline.
If DMs are closed, `/character` retrieves the saved result privately. The
`/premium` shop links this offer and includes its native purchase button when the
SKU is verified and available. Checkout availability and price are controlled
in the Discord Developer Portal; deployment does not publish a SKU or set its
price.

## Character source and rendering costs

The free [Kitsu character API](https://kitsu.docs.apiary.io/) reported **105,906
character records** in a live read-only query on October 11, 2026. Selection
samples a random offset across its reported entire catalog, caches the size for
a day, and spaces requests by at least 1.1 seconds. It does not download a bulk
character list or promise a catalog containing every person. Named selections
cover identities outside that anime/manga catalog without a lookup charge.
The current count is over 100,000, not verified as hundreds of thousands.

The image prompt is exactly `[character name] as a gfl2 3d character model`.
Only the selected name is used. Catalog descriptions and portrait images are
stored as selection metadata but are not sent to the image provider. Generation
uses the text generation endpoint, matching an ordinary `imagine` request with
that wording. No portrait pose, background, panel, or text restrictions are added.

At the operator's explicit request, generation uses
[GPT Image 2.5 Sunburst](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)
with `quality=low`, `size=1024x1024`, and one output. It reuses the existing
OpenAI API key. Character selection is free; image inference is paid. Actual
cost depends on text/output tokens; see
[OpenAI pricing](https://developers.openai.com/api/docs/pricing). Low quality,
no reference inputs, no chat/classifier calls, and saved results keep costs down.
This product does not honor a more expensive personal image default and does
not automatically retry through Gemini or upgrade quality.

Alternatives researched: Jikan has a free random-character API, but requests
timed out from both the desktop and production during evaluation. AniList's
[commercial-use terms](https://docs.anilist.co/guide/terms-of-use) require
permission at $150/month or more in revenue, so it was not selected for this
paid feature. GPT Image Mini lists a $0.005 low-quality square output charge
plus inputs, but [its announced shutdown](https://developers.openai.com/api/docs/deprecations)
is December 1, 2026. A self-hosted image model removes per-call API charges but
still needs compute. No reliable unlimited free hosted renderer was verified.

## Recovery and operations

`character_renders.db` in application state stores named choices, entitlement
and buyer IDs, the fixed character/source record, render attempt/lease state,
the saved PNG, consumption confirmation, and refund tombstones. Include it in backups, privacy access,
and full-data deletion procedures. It has no automatic expiry. Refund/deletion
events remove the stored PNG and deny access; retrieval rechecks the actual
Discord entitlement before returning an image.

An additive `consumption_confirmed` column preserves existing ledger rows. A
consumption timeout or uncertain response keeps ready image bytes; recovery
checks Discord and retries only consumption, not generation. A consumed
entitlement without an existing ready image is rejected. A crash after Discord
consumes but before local confirmation is recovered from the saved image.
Refunds during consumption still remove the image and block delivery.

A SQLite transaction claims each entitlement so concurrent gateway events,
command invocations, and processes cannot start duplicate completed jobs.
Failed source lookups, app rate limits, and full provider queues do not spend a
provider attempt.
Failed image calls release their lease, preserve the selection, and require an
explicit `/character` retry. At most three provider attempts are permitted;
afterward the operator must resolve delivery with the purchaser. A timed-out
provider may have billed work without returning an image, so a retry can incur
another API charge. Saved successful images are never regenerated for downloads.

Automatic work is bounded to eight queued jobs, and provider concurrency plus
personal/guild/global image limits remain enforced. Image generation times out
after ten minutes; interrupted jobs can be retried after their lease expires.
Delivery attempts are recorded before DM sends to avoid repeating failed DMs
on every poll; customers retain private command recovery.

`MULTIVAC_CHARACTER_SKU_ID=0` disables purchase processing. Its default is the
new consumable ID; alternate primary products must be consumable and belong to
this app. The known earlier durable SKU is accepted for recovery while processing
is enabled. No new API key or infrastructure service is needed.
