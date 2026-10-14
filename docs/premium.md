# Discord premium products

The owner selected the products below for MultiVAC application
`1087777433459118100`. Names, IDs, and actual types were verified using Discord's
authenticated SKU API on October 11, 2026. Prices were supplied by the owner;
the SKU API does not expose checkout prices. Listing copy is in
[premium_skus.json](premium_skus.json).

| Product | SKU ID | Type | USD price | Delivery |
| --- | --- | --- | --- | --- |
| fUnThrottled | `1558736446351872112` | User subscription (`5`, user flag) | $4.99/month | Automatic bypass of shared bot-wide throttle counters while active |
| Donate | `1558738728720007219` | Consumable (`3`) | $2.99 once | `/premium_redeem donate` records one support purchase; no throttle benefit |
| Donate More | `1558739372851859466` | Consumable (`3`) | $9.99 once | `/premium_redeem donate_more` records one support purchase; no throttle benefit |
| GFL2-inspired character portrait | `1558749059609665566` | Durable (`2`) | $0.99 once | One saved PNG, generated on purchase; `/character` recovers/downloads it |
| Unattractive | `1558762748324876308` | Consumable (`3`) | $1.99 once | `/unattractive activate` adds 10 minutes of unattractive chat and new visuals for everyone in the selected server |

The new character product uses a separate durable delivery ledger and fixed
Sunburst 2.5 low-quality renderer. See [character portraits](character_portraits.md)
for source, checkout availability, and recovery details. It grants no throttle benefit.

Unattractive uses a separate timed server-effect ledger within `premium.db`.
See [Unattractive](unattractive.md) for activation, stacking, manager stop,
expiry and refund handling. It is separate from `/premium_configure` mappings.

This supersedes the earlier proposed $10 guild subscription, $5 personal pass,
and $1 reset. The additional Server Goon SKU found in Discord is not configured
or offered by default; its price and benefits were not selected. System-created
subscription-group SKUs are never used for purchases or entitlements.
[Discord SKU types and flags](https://docs.discord.com/developers/resources/sku).

## Configuration

The three selected IDs are the default for a fresh installation. They grant
nothing until Discord verifies application ownership, product types, and active
entitlements. Existing saved owner configuration takes precedence. After
deployment, the application owner can replace a saved mapping with:

```text
/premium_configure unthrottled_sku_id:1558736446351872112 donate_sku_id:1558738728720007219 donate_more_sku_id:1558739372851859466
```

Omitted products are disabled by this command. `/premium_skus` lists actual SKUs.
Configuration validates distinct IDs and actual user-subscription versus
consumable types. It does not create or publish products or set prices.
Mappings cannot change after a consumable has been recorded without an operator
migration. The private database migrates the earlier settings schema while
preserving saved mappings and redemption history.

Environment overrides are available before startup:

```text
MULTIVAC_UNTHROTTLED_SKU_ID
MULTIVAC_DONATE_SKU_ID
MULTIVAC_DONATE_MORE_SKU_ID
```

Set each to `0` to disable its default. Earlier guild/pass/reset settings remain
for compatibility and default to disabled. Do not configure or advertise the
earlier benefits without separately confirming the intended offering.
Discord checkout supplies the final localized price, taxes, and renewal terms.
Use Discord test entitlements to verify access, removal, expiry, and consumable
recording before relying on delivery. Automated local tests mock external calls
and do not create purchases or send Discord messages.

## Throttle behavior

fUnThrottled exempts only its current verified buyer from MultiVAC's shared
bot-wide sliding-window limits for request, URL, image, compute, and code actions.
It applies across servers and DMs. It does not raise personal or guild limits.
Subscriber requests still count toward those limits and shared counters; other
users receive no exemption.

The subscription does not bypass provider queue/concurrency limits, provider
rate limits or quotas, media-size restrictions, content safety, or permission
and approval checks. A paid request can still be limited by any of these.

The two donation products are support-only purchases. Neither changes limits,
grants a pass, resets counters, nor enables fUnThrottled. Confirmation records
and consumes one unused entitlement so it cannot be recorded twice.
Neither support product renews automatically, and no charitable
tax-deductibility representation is made.

`/shop` (also `/premium`) displays configured products with native Discord purchase buttons
when the SKUs are available and verified. `/premium_status` reports subscription
status, the current entitlement end time when supplied by Discord, and effective
limits. Subscription access is automatic; no redemption is needed.

Throttle notices retain the retry time and offer a persistent **Open shop** button.
Anyone clicking it opens their own private shop, even while request counters
are capped. The navigation survives restarts and recovers older, expired shop
buttons. Bot-wide cap notices suggest fUnThrottled; personal and server cap
notices explain that the subscription does not remove those caps. Browsing never
spends a purchase or retries the denied request. Checkout buttons still require
verified, available SKUs. Use `/shop` or `/premium` any time. Plain-text `/shop`,
`@MultiVAC shop`, “your shop” and “where can I access the shop” open the same
verified storefront directly. A reply asking to link a bot-owned shop notice
again reopens it. These controls do not need generation quota or a chat-model
response. Text requests show the product listing in the requesting channel;
button and native command requests show it privately.

## Verification and recovery

Private `premium.db` stores mappings, entitlement/user IDs, consumable recording
state, and revocations. It also preserves prior local pass/reset records if any
exist. It stores no card details or prompts. Include it in protected application
state access, deletion, and backup procedures.

Gateway events update subscriptions and remove deleted/refunded entitlements.
REST reconciliation runs every two minutes. Events arriving during a REST read
take precedence over its older snapshot. Access expires locally after five
minutes without verification, and is re-verified after restart. Entitlement
start/end times determine access; cancellation alone does not end an otherwise
valid paid period. Donations never qualify for subscription access.

Consumable recording reserves an entitlement durably before Discord consumes it.
The recording and applied marker commit together. If Discord's response is
lost or local commit fails, retrying recovers the same reserved purchase without
consuming it twice. A consumed purchase with no local reservation cannot be
replayed. Confirmation is bound to the purchasing account.
[Discord entitlement lifecycle](https://docs.discord.com/developers/resources/entitlement).

This implementation assumes the existing single bot process and private state
directory. Multiple replicas would require shared throttle and redemption
coordination.
