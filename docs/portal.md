# MultiVAC website and owner portal

The public page preserves the existing MultiVAC landing page. `/request` uses
Discord sign-in to submit invite requests and retrieve approved server invites.
`/admin` shows current installations, requests, decisions and recorded provider
usage by server/model/day for today or the last 7, 30 or 90 days. Costs are ledger
estimates, not provider invoices; missing attribution is explicitly separate.
Public visitors see only the existing aggregate token metrics.

Run `python -m services.portal` for the aiohttp server on port 8080. Production
uses `ops/docker-compose.portal.yml` layered over the normal release files, with
no published host ports. The canonical external infrastructure runbook owns host
paths, DNS, tunnel configuration, release activation and rollback.

## Operator configuration

The separate, ignored `.portal.env` must be readable only by the deployment user.
Set `PORTAL_ORIGIN` to the canonical HTTPS origin. Set `DISCORD_TOKEN` to this
bot's token and `DISCORD_CLIENT_SECRET` to this application's OAuth secret. An
optional `DISCORD_CLIENT_ID` must match the bot application; otherwise the portal
resolves it from Discord. No AI/provider keys belong in the portal environment.
Register `<PORTAL_ORIGIN>/auth/discord/callback` as an exact redirect in Discord's
Developer Portal. The portal requests only `identify guilds` for website sign-in.
It resolves the application owner from Discord; for a team application it uses
the team's `owner_user_id`, not the synthetic team account in `owner.id`.

Sessions and OAuth access tokens live in server memory for at most one hour;
restarts require another sign-in. Cookies are Secure, HttpOnly and SameSite=Lax.
OAuth state is one-use and expires after ten minutes. Mutations require the exact
configured Origin and a session CSRF token. Access logging is disabled to avoid
recording authorization codes. Only the owner can retrieve the admin APIs.

Before enabling the bot admission gate, explicitly seed existing installations
once using `PortalStore.seed_existing(guilds, owner_id)` with current Discord
membership. Existing servers retain access; no boot-time auto-approval occurs.
Set the bot's `INVITE_APPROVAL_DB_PATH` to the shared private portal database to
enable enforcement. Keep that database separate from public metrics. Back it up
before deployment and restore it together with the matching admission settings
if rolling back. Database permissions must permit the bot and portal runtime UID.

Managers submit the actual server ID and an intended-use description. Discord
permissions are rechecked at submission; client names and permissions are ignored.
Pending requests cannot be duplicated; rejected requests have a 24-hour cooldown.
Revoked servers need owner intervention before resubmitting. Approved invites
request normal message permissions, never Administrator. The bot refuses new
messages/application commands in unapproved servers and leaves unapproved or
revoked installations on join or within a minute. Database read failures block
new work but defer removal until the database is available. DMs remain available.
Disabling the gate is an explicit operator action, not an automatic failure mode.

There are no automatic Discord DMs or emails. Managers check `/request` for their
status and link; the owner checks `/admin` for the review queue. Review notes are
visible to the requesting server's current managers. Decisions are audited.

The bot publishes private, atomic aggregates every minute to `PORTAL_REPORT_PATH`.
Its usage ledger uses WAL, so a file-only database mount would miss recent writes.
The portal mounts only this aggregate directory and the existing public metrics
directory, both read-only. It has no usage ledger, conversation, memory, purchase
or provider-key mount. The exporter uses `mode=ro` and `query_only` against the
bot's own ledger and never runs migrations. Set `PORTAL_REPORT_PATH` in both
services to their respective paths for the same protected aggregate file.
Reporting windows use `USAGE_TZ` (default America/New_York); daily bars use UTC.
The bot's protected admission code and portal files cannot be rewritten through
the bot's ordinary generated-code proposal mechanism.

Validate with `python -m unittest discover -s tests -p test_portal.py` and the
repository's full unittest discovery. Browser/OAuth end-to-end verification needs
the real redirect registration, secret and an interactive Discord sign-in.
