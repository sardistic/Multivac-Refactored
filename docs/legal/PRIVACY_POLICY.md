# MultiVAC Privacy Policy

**Draft prepared:** October 11, 2026  
**Effective date:** October 11, 2026  
**Operator:** Intracoast Network Services  
**Contact person:** [redacted]  
**Contact:** bot@sardistic.com

Intracoast Network Services (“we,” “us,” or “our”) operates the MultiVAC Discord bot.
This Policy describes information processed by MultiVAC, including information
received through Discord and information sent to external providers. It applies
to the bot and its associated features. Discord and external services also
process information under their own policies.

## 1. Information we process

Depending on the enabled features and your interactions, MultiVAC processes:

- **Discord identifiers and context:** user IDs, display names, server and
  channel IDs, message IDs, timestamps, reply relationships, and permissions
  needed to handle or authorize requests.
- **Requests and conversation content:** messages addressed to the bot, its
  responses, replied-to messages, and a bounded window of nearby channel
  messages. Nearby messages can include people who did not directly address
  the bot.
- **Attachments and tool inputs:** images, videos, supported files, attachment
  URLs, links you ask us to open, search queries, and locations you provide for
  weather or other location-based requests. A saved location may persist for
  later requests; it is not continuous device location tracking.
- **Personalization and memory:** remembered facts, saved instructions,
  preferences (including model tiers you choose), conversation persona settings,
  and automatically distilled
  summaries of a user's indexed interactions where that feature is available.
- **Operational records:** provider/model names, request status, tool activity,
  timing, errors, public evidence links, token counts, estimated costs,
  attribution to a user, and feature approval or change-history records.
- **Support information:** information you provide when asking for help,
  reporting abuse, or making a privacy request.
- **Premium access and redemption:** Discord entitlement and SKU identifiers,
  the entitled user or guild, subscription validity periods, donation recording
  and consumption state, dates, and revocation status. Earlier local pass/reset
  records, if any, also contain grant periods and reset generations. Payments occur through
  Discord checkout; the bot does not collect your payment card details.

Do not send passwords, access tokens, payment card details, private medical
records, or other sensitive personal information. MultiVAC does not need your
Discord password to operate.

## 2. Why we process information

We use information to answer requests, analyze or generate media, retrieve
information, maintain conversation context, apply saved preferences, provide
requested memory features, manage usage and costs, enforce permissions and
safety controls, diagnose failures, respond to support and privacy requests,
verify subscription access, record support purchases, prevent duplicate consumable redemption,
and meet applicable legal obligations.

Where privacy law requires a legal basis, we rely on the applicable basis for
each purpose, such as providing a service you request, legitimate interests in
secure operation, consent where required, or compliance with law. A server
administrator's installation of the bot does not replace any individual consent
required by applicable law.

## 3. Conversation context, memory, and reflection

When invoked by a mention or reply, MultiVAC can read up to 24 preceding messages
in the same channel, subject to a 12,000-character limit. Display names and user
IDs help distinguish speakers. This live context window is processed
temporarily; it does not itself create another transcript archive.

Separately, when the Elasticsearch/OpenSearch memory backend is configured,
messages addressed to the bot and bot replies can be indexed persistently with
their content and Discord identifiers. Referenced messages may also be indexed.
Members with the required moderation permission can explicitly index recent
channel history, including messages from people who did not invoke the bot.
Personalization can retain user facts,
instructions, and a distilled profile. Some personalization is keyed by user ID
and may apply in later conversations; conversation context and persona settings
use their own channel or conversation scope.

If reflection is enabled by the operator, invoking the bot opens a bounded
interaction-analysis session. Defaults include a ten-minute lookback, continued
processing of messages in that channel while the session is active, and closure
after five minutes without activity. Other participants' messages may supply
context. Temporary text can be sent to the configured AI provider. The reflection
store retains derived observations, hashed actor references, evidence/message
identifiers, improvement ideas, and operational or budget records, rather than a
second raw conversation archive. Hashed references are pseudonymous and are not
necessarily anonymous.

## 4. Who receives information

- **Discord and channel participants:** responses are sent through Discord.
  People who can access the destination channel can normally see them. Direct
  messages still involve Discord and relevant processing providers.
- **AI and media providers:** configured features may send prompts, relevant
  context, saved personalization, attachments, and requested generation or
  analysis inputs to OpenAI, Google/Gemini, Anthropic, or other configured media
  providers. Context may include Discord display names or identifiers.
- **Search and information providers:** relevant queries, image data or public
  image URLs, locations, and other tool inputs may be sent to services such as
  Google Search, Google Cloud Vision, SerpApi, OpenWeather, or Finnhub when their
  feature is used. Opening a supplied link involves the destination website.
- **Hosting and storage providers and authorized operators:** these parties may
  process stored information or access it when necessary to maintain the
  Service, investigate incidents, or fulfill requests.
- **Legal or safety recipients:** we may disclose information where required by
  law or reasonably necessary to address unlawful activity or protect rights
  and safety, subject to applicable law.

Server members with the Discord `Manage Messages` permission can view per-user
usage rankings through the usage command, including user IDs, call counts,
and estimated costs. This is distinct from public status statistics,
which contain aggregate call and token counts without user IDs or prompts.

We do not sell Discord API data or share it with advertising networks or data
brokers. MultiVAC does not operate a model-training pipeline. Sending content
to a provider for a response is still disclosure to that provider; its retention
and permitted uses depend on its applicable service terms and account settings.
This Policy does not promise zero provider retention or that every provider
processes data identically. Relevant provider notices include
[Discord](https://discord.com/privacy), [OpenAI](https://openai.com/policies/privacy-policy/),
[Google](https://policies.google.com/privacy), and
[Anthropic](https://www.anthropic.com/legal/privacy).

## 5. Retention and deletion

Temporary conversation windows and attachment-processing inputs are used for
the relevant request or active session. This does not delete copies already
held by Discord, an external provider, or a separately enabled memory feature.

Indexed conversation memory, saved facts and profiles, preference or instruction
history, agent-run records, expanded responses, saved locations, and usage
records, including premium redemption records, do not currently have one
universal automatic deletion deadline. Some
can remain until removed by a feature control or an operator. Deleting the
original Discord message or removing the bot does not automatically remove all
of these copies.

Reflection's default cleanup removes completed sessions after 30 days and
selected run, budget, and frequency metadata after 90 days. Active observations
and ideas have separate lifecycles and may remain until removed or superseded.
Deployment settings can change those defaults; contact us for the schedule
applicable to your data.

You can request deletion using the contact method below. We will delete data we
are required to delete under applicable law and platform requirements. Any
permitted retention for legal obligations or security must be limited to the
necessary data and purpose. Copies held by external providers and any backups
require separate handling; we will explain applicable limitations rather than
promise immediate deletion from every system.

## 6. Your controls and privacy requests

- Use `/memories` to inspect remembered facts and `/profile` to view a distilled
  profile when available. These commands do not provide a complete data export.
- Use `/forget <text>` to delete saved facts matching that text. It does **not**
  erase indexed messages, profiles, usage records, or every other stored record.
- Use `/reflection_off` to remove your pending sessions and associated derived
  reflection observations. Invoking the bot again re-enables reflection for a
  new session if the operator has enabled that subsystem.
- Stop invoking the bot, or ask a server administrator to restrict or remove
  it. Nearby messages may still be processed when another person invokes it
  in a channel it can access.

For access, correction, deletion, export, objection, restriction, or withdrawal
of consent where applicable, contact **bot@sardistic.com**, attention [redacted]. Identify your Discord user ID and the relevant feature or server
privately. We may request
proportionate verification to protect your data; never send a Discord password
or access token. We will respond within the deadlines applicable to your
request. You may also have the right to complain to your local privacy regulator.

## 7. Security and international processing

MultiVAC uses application permissions, scoped data access, and bounded processing
controls. No online system can guarantee complete security. If you suspect
unauthorized access, contact us promptly.

The operator and providers may process information in countries other than
where you live. Those countries may have different privacy laws. Where law
requires transfer safeguards, the applicable safeguards must be in place for
the processing concerned. Contact us for information about the providers and
transfer arrangements applicable to your request.

## 8. Children

MultiVAC is not intended for anyone under 13 or under Discord's higher minimum
age where applicable. If you believe a child below the applicable minimum age
has submitted personal information, contact us so we can investigate and take
appropriate action, including deletion where required.

## 9. Policy changes and contact

We will publish updates with a revised effective date and provide reasonable
notice of material changes through the Service or its public policy location.
Where required, we will obtain consent before new processing begins.

**Privacy contact:** bot@sardistic.com  
**Contact person:** [redacted]  
**Operator:** Intracoast Network Services  
**Operator location:** North Carolina, United States
