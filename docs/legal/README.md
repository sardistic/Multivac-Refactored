# MultiVAC policy drafts

[Terms of Service](TERMS_OF_SERVICE.md) and [Privacy Policy](PRIVACY_POLICY.md)
were drafted on October 11, 2026 from the repository's current bot, memory,
personalization, reflection, tool-run, and usage behavior. They are drafts for
operator review, not confirmation of legal or Discord platform compliance.

Before publishing:

1. Confirm the supplied details: Intracoast Network Services is the operator,
   [redacted] is the contact person, `bot@sardistic.com` is the public
   support/privacy email, and North Carolina, United States is the operator's
   location and governing jurisdiction. The effective date is October 11, 2026;
   update it if the policies are adopted on a different date. Ensure the email
   is monitored for support, abuse, and privacy requests.
2. Confirm which providers and optional features are enabled in production,
   their data-use and retention settings, and the actual retention and backup
   schedules. The drafts intentionally disclose that some stores have no
   automatic expiry; do not substitute a deletion promise that is not enforced.
   Premium terms reflect fUnThrottled at $4.99/month, Donate at $2.99 once, and
   Donate More at $9.99 once. Donations grant no throttle benefit; fUnThrottled
   bypasses only bot-wide throttle counters. Match actual Discord checkout pricing and
   published benefits before enabling sales; see [premium setup](../premium.md).
3. Establish and verify an operator procedure for full access, correction, and
   deletion requests across memory indexes, SQLite state, usage, reflection,
   logs, backups, and providers. `/forget` deletes matching facts only;
   `/reflection_off` is not a permanent opt-out after another invocation.
4. Review automatic user-profile distillation against
   [Discord Developer Policy, item 16](https://support-dev.discord.com/hc/en-us/articles/8563934450327-Discord-Developer-Policy),
   which restricts profiling of Discord users. Disclosure alone does not resolve
   whether the current feature is permitted. Also review the notice and lawful
   basis for processing nearby participants' messages and reflection sessions.
5. Verify actual security and international-transfer arrangements. The code
   review does not establish hosting encryption, provider agreements, or lawful
   transfer safeguards. Discord's
   [Developer Terms, section 5](https://support-dev.discord.com/hc/en-us/articles/8562894815383-Discord-Developer-Terms-of-Service)
   address accessible privacy policies, deletion, and data security, including
   encryption at rest. The FTC's
   [data protection guidance](https://www.ftc.gov/business-guidance/resources/protecting-personal-information-guide-business)
   also recommends a defined retention and disposal policy.
6. Review the completed documents for the operator's jurisdiction and audience.
   Publish both at stable public URLs, replace their relative cross-links with
   the public URLs as needed, and add them to the Discord Developer Portal and
   the bot's public information. If using GitHub-rendered documents, they must
   be accessible without a repository login.

This change creates documentation only. It does not publish a website, configure
Discord policy URLs, change bot behavior, or implement a full-data deletion
command.
