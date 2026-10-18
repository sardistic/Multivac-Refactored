# Personal model defaults and Gemini retries

Use `/model_defaults` to view your saved choices. To set a cheaper default for
every supported workload, use `/model_defaults workload:All tier:Economy`.
Set individual workloads separately, for example
`/model_defaults workload:Image tier:Economy` or
`/model_defaults workload:Chat tier:Balanced`.
`/model_defaults workload:All tier:Auto` clears your overrides.

Choices belong to the account using the command, apply across servers and DMs,
and survive bot restarts. Command responses are private. Workloads are chat,
research, deep chat, vision analysis, image generation, image editing, and video.
Research and reverse-image tools remain available when changing text tiers.

The command also provides workload/tier dropdowns. **Apps → Model defaults**
opens the same controls. Text invocation supports `@MultiVAC model defaults
image economy`; that reply is visible in its channel. Server defaults fill in
Auto, and server maximums cap effective tiers while retaining your saved
choices. See [app actions and server controls](app_actions.md).

| Tier | Text and vision | Image picker highlight / edit default | Video preselection |
| --- | --- | --- | --- |
| Auto | Server default, then workload default | Server default, then Mini low highlight / existing edit route | Server default, then existing default |
| Economy | GPT-6 Luna | GPT Image Mini, low quality | Veo 3.1 Fast, 4 seconds |
| Balanced | GPT-6.1 Sol | GPT Image 2.5 Sunburst, medium quality | Veo 3.1 Fast, 8 seconds |
| Quality | GPT-6 Astra | GPT Image 2.5 Sunburst, high quality | Veo 3.1, 8 seconds |

Economy trades quality and capability for lower provider costs. OpenAI describes
[GPT Image Mini](https://developers.openai.com/api/docs/models/gpt-image-1-mini)
as its cost-efficient image model; see the
[provider pricing](https://developers.openai.com/api/docs/pricing) for current
rates. These choices do not set a fixed per-request price or change subscription
benefits, personal/guild limits, or provider capacity. Explicit Gemini and Claude
requests keep their own chat/edit route where the server maximum
allows it. Image generation always uses the priced picker. Cheaper server maximums block external provider routes and retries
for that workload. Automatic retries of a
saved OpenAI tier retain that model and quality. An empty Economy chat response
does not silently upgrade to the bot's more expensive chat default.

Image generation in chat, `/generate_image`, Apps forms, and `imagine weather`
offers a requester-only menu before rendering. OpenAI Mini and Sunburst offer
low/medium/high quality at 1024×1024, 1536×1024, or 1024×1536. Gemini 3 Pro Image
offers 2K/4K at square, landscape (16:9), or portrait (9:16), using its native
standard quality. Only configured providers and server-permitted settings appear.
The saved image tier highlights an available square choice; it does not start
generation. Choosing any menu option starts that configuration immediately.
**After a fixed 30 seconds without a choice, use Mini low at 1024×1024**,
even when a more expensive tier is highlighted. Cancel sends no image request.
If OpenAI is unavailable, timeout cancels rather than selecting another API.
Recheck availability, server maximum and channel access before submission.
Failed selected generations do not switch providers; start a new priced request.
Image edits retain their existing defaults and retry confirmation.

The menu labels API output estimates in USD, with prompt/reference inputs extra.
Mini output estimates range from $0.005 (low square) to $0.052 (high rectangle).
Sunburst is explicitly variable at $0.030 per 1K output tokens, rather than
claiming an unverified per-image total. Gemini output estimates are $0.134 for
2K and $0.24 for 4K; its existing cost overrides affect both menu and ledger.
Prices were checked against the official
[OpenAI image guide](https://developers.openai.com/api/docs/guides/image-generation),
[OpenAI pricing](https://developers.openai.com/api/docs/pricing), and
[Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3-pro-image)
on October 18, 2026. The fixed $0.99 character product retains its own settings.

Video defaults only preselect a configuration: users still
choose in the existing cost confirmation screen before any generation starts.
Server maximums also restrict those choices and are rechecked after confirmation.
Economy video maximums allow only Veo Fast at 4 seconds; Balanced allows Veo
Fast at 4, 6, or 8 seconds. Quality and Auto allow all available Veo choices.
Video generation requires the runtime's Gemini client. If it is unavailable,
the bot explains that video generation is unavailable without opening a picker.
Sora generation and remixing have been retired following the
[September 24, 2026 API shutdown](https://developers.openai.com/api/docs/deprecations#2026-03-24-sora-2-video-generation-models-and-videos-api).
Existing usage records are retained. Remixing/editing existing videos is unavailable;
new prompt-to-video and image-to-video requests continue through Veo.

When OpenAI chat, editing, or vision analysis needs a Gemini
retry, the bot first offers **Retry with Gemini** and **Cancel**. Sending a video
to Gemini because the selected chat model cannot read it also requires this
confirmation. The prompt identifies the model and explains that the request
and attachments will go to the provider and may use additional API credits.
Only the original requester can answer; cancellation, expiry after two minutes,
or failure to deliver the prompt sends no fallback request. Approval applies to
that retry only and does not change defaults. Reference images are retained.

The bot registers the command during normal startup/command synchronization.
Preferences are stored in `conversation_history.db` in the additive
`user_model_preferences` table, keyed by Discord user ID and workload. Choosing
Auto removes the applicable rows. Model IDs are selected from a fixed allowlist;
inferred personality and memory never change these settings. No additional
configuration is required; production needs the updated bot release and a
successful command sync before the command appears.
