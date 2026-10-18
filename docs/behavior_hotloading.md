# Live Behavior Components

Multivac can replace reviewed event adapters, intent handlers, provider entry
points, and runtime settings without reconnecting Discord. The permanent shell
continues to own the gateway, process bootstrap, databases, and the hotloader.

## Component contract

Each standalone `live_components/*.py` module exports async `setup(context)` and
at least one entry in `BEHAVIOR_HANDLERS` or `BEHAVIOR_SETTINGS`.

```python
BEHAVIOR_VERSION = "1"

async def chat(ctx):
    await ctx.message.reply("This generation owns chat requests.")
    return True

async def openai_chat(messages, **kwargs):
    return "provider response"

BEHAVIOR_HANDLERS = {
    "intents": {"chat": chat},
    "providers": {"chat.openai": openai_chat},
}
BEHAVIOR_SETTINGS = {
    "intent.model.chat": "gpt-5.6-terra",
    "intent.duration.chat": 6,
}

async def setup(context):
    return {"ready": True}

async def healthcheck(context):
    return bool(context.state["ready"])

async def teardown(context):
    pass
```

Handler namespaces:

- `events`: `message(message)`, `raw_reaction_add(payload)`, and
  `command_error(ctx, error)` replace the stable-shell fallback.
- `intents`: an exact classifier intent receives one `DispatchContext`.
- `providers`: receives the same arguments as the provider entry point it
  replaces. Current keys include `chat.openai`, `chat.openai_plain`,
  `chat.gemini`, `chat.claude`, `image.generate`, `image.gemini.generate`,
  `image.gemini.edit`, `image.openai.edit`, `vision.openai`,
  and `video.veo.generate`.
- `settings`: declared through `BEHAVIOR_SETTINGS`. The built-in router reads
  `intent.model.<intent>` and `intent.duration.<intent>`.

Routes already owned by another live component require an exact declaration:

```python
BEHAVIOR_OVERRIDES = {
    "intents": ["chat"],
    "providers": ["chat.openai"],
}
```

The supervisor authorizes declared overrides after review. Undeclared
collisions fail activation.

## Lifecycle and consistency

Every Discord event captures one immutable behavior generation. Nested intent,
provider, and setting lookups use that same generation. A replacement becomes
visible atomically to new events while old requests retain their callbacks and
settings.

After the swap, the loader waits for calls using the prior source generation to
finish. It then sets `context.stop_event`, calls async `teardown(context)`,
closes resources registered with `context.track_resource(...)`, and drains or
cancels tasks created with `context.create_task(...)`. Discord Views can be
tracked as resources; their `stop()` method is called during teardown.

Optional async `healthcheck(context)` runs before publication. Returning
`False` aborts activation. Batch failure restores earlier mutations in reverse
order. Active sources persist in `active-behaviors.json` and are restored after
a process restart.

## Reviewed activation

Behavior proposals may contain only standalone `live_components/*.py` files.
The host supervisor runs the full networkless suite, imports components against
the real bot composition root in a restricted container, publishes a signed
content-addressed artifact, and sends exact paths and SHA-256 values to the
running bot. Activation, deletion, and proposal rollback do not recreate it.

For local development without supervisor control:

```text
/behavior_hotload status
/behavior_hotload load live_components/chat.py
/behavior_hotload unload live_components/chat.py
/behavior_hotload rollback live_components/chat.py
```

The optional final `true` authorizes declared overrides. Direct mutation is
disabled when supervisor control is configured.

## Restart-only boundary

Dependency installation, native extensions, database migrations, Discord token
or intent changes, logging/event-loop bootstrap, mounts, hotload authority, and
the stable gateway shell remain normal releases. Arbitrary module reload cannot
safely replace existing class identities, unmanaged tasks, or native state.

This infrastructure was deployed to production on 2026-07-22. A harmless
owner-approved behavior proposal is still needed to verify live activation,
in-flight generation draining, teardown, and rollback against the running bot.
