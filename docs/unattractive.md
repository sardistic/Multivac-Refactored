# Unattractive server effect

SKU `1558762748324876308` is a **$1.99 consumable**. One confirmed activation
adds **10 minutes** of intentionally unappealing chat and newly generated
visuals for **everyone in the selected server**. It grants no throttle benefit.
The SKU was verified as type 3 and available for checkout during live
post-deployment verification. Discord checkout provides the final localized
price; publishing and availability are controlled in the Developer Portal.

Use `/unattractive` or `@MultiVAC unattractive` for checkout. After buying,
use `/unattractive action:activate` or `@MultiVAC unattractive activate` in
the intended server and confirm the requester-only button. The server channel
must already be eligible for media app actions and still permit the bot to
view/send. A purchase alone does not pick a server or start the timer. Purchases
made outside the server can be activated here later; DMs cannot activate it.

Any member with access to MultiVAC can activate their own purchase. Repeated
purchases queue another 10-minute window after existing active/queued time.
`/unattractive action:status` shows the current server timer. Members with
**Manage Server or Administrator** can use `/unattractive action:stop` or
`@MultiVAC unattractive stop` to cancel active and queued time. A used purchase
cannot be activated again. Stopping does not disable future purchases.

Chat uses a dour, dry, cantankerous style while preserving helpfulness,
accurate facts, citations and code. Images, edits and videos request drab,
clashing colors, awkward composition, unflattering light and shabby styling,
while preserving the requested subject and reference inputs. This does not
instruct harassment or alter classifiers, tool arguments, permissions, model
selection, costs, server ceilings, quotas, or fallback consent. Effects change
generation context, so exact visual outcomes are model-dependent.
The final chat reviewer receives the temporary style flag so its factual
checks do not inadvertently remove the purchased voice.

The modifier is applied when the response/render provider call is prepared.
Requests already sent cannot be changed; a result may finish after the timer
expires. Another server's requests and DMs are unaffected. New paid character
renders manually requested in the affected server receive the visual style
while retaining the fixed Sunburst renderer. Automatic character delivery is
a DM operation and unaffected. Cached images are never rerendered or modified.

Consumption is confirmed by Discord before starting the timer. Transactional
records prevent duplicate grants across restart, freeze a pending activation
to its original server, and recover a consumed purchase whose local grant was
interrupted. Uncertain consumption starts no local timer; retry activation in
the original server. Gateway refund/removal events revoke access; REST checks
reconcile pending and still-active purchases every two minutes. Network outages
can delay revocation discovery until the next successful check, but cannot
extend the stored expiry. Refunded windows are removed; later queued windows
retain their original schedule. Manager stop cannot recall requests already
submitted, and cancels granted time rather than pending confirmations.

`server_effect_purchases` and `server_effect_revocations` are additive tables
in `premium.db`. They store buyer, server, SKU and entitlement IDs, pending/
applied state, activation windows and stop/revocation state without storing
prompts or outputs. The timer expires automatically; the transaction ledger
has no automatic deletion deadline and belongs in backups/privacy handling.
Set `MULTIVAC_UNATTRACTIVE_SKU_ID=0` to disable the feature. This SKU is separate
from `/premium_configure` mappings, preserving existing subscription/donation
configuration. [Discord consumable fulfillment](https://docs.discord.com/developers/monetization/implementing-one-time-purchases).
