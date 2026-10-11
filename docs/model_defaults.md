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

| Tier | Text and vision | Images and edits | Video preselection |
| --- | --- | --- | --- |
| Auto | Existing workload default | Existing route and quality | Existing default |
| Economy | GPT-6 Luna | GPT Image Mini, low quality | Sora 2, 4 seconds |
| Balanced | GPT-6.1 Sol | GPT Image 2.5 Sunburst, medium quality | Sora 2, 8 seconds |
| Quality | GPT-6 Astra | GPT Image 2.5 Sunburst, high quality | Sora 2 Pro, 8 seconds |

Economy trades quality and capability for lower provider costs. OpenAI describes
[GPT Image Mini](https://developers.openai.com/api/docs/models/gpt-image-1-mini)
as its cost-efficient image model; see the
[provider pricing](https://developers.openai.com/api/docs/pricing) for current
rates. These choices do not set a fixed per-request price or change subscription
benefits, personal/guild limits, or provider capacity. Explicit Gemini, Claude,
and Stability requests keep their own provider route. Automatic retries of a
saved OpenAI tier retain that model and quality. An empty Economy chat response
does not silently upgrade to the bot's more expensive chat default.

Weather images also use a saved image tier. Auto retains the existing Gemini
weather renderer. Video defaults only preselect a configuration: users still
choose in the existing cost confirmation screen before any generation starts.

When OpenAI chat, image generation, editing, or vision analysis needs a Gemini
retry, the bot first offers **Retry with Gemini** and **Cancel**. Sending a video
to Gemini because the selected chat model cannot read it also requires this
confirmation. The prompt identifies the model and explains that the request
and attachments will go to the provider and may use additional API credits.
Only the original requester can answer; cancellation, expiry after two minutes,
or failure to deliver the prompt sends no fallback request. Approval applies to
that retry only and does not change defaults. Reference images are retained.
An unsuccessful explicit Gemini image generation similarly offers an OpenAI
retry for confirmation.

The bot registers the command during normal startup/command synchronization.
Preferences are stored in `conversation_history.db` in the additive
`user_model_preferences` table, keyed by Discord user ID and workload. Choosing
Auto removes the applicable rows. Model IDs are selected from a fixed allowlist;
inferred personality and memory never change these settings. No additional
configuration is required; production needs the updated bot release and a
successful command sync before the command appears.
