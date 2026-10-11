"""Process-wide behavior registry and stable dispatch helpers."""

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Callable

from services.behavior_runtime import BehaviorRegistry
from services.security_limits import provider_capacity
from services.usage_costs import get_request_context


BEHAVIOR_REGISTRY = BehaviorRegistry()
_ACTIVE_SNAPSHOT: ContextVar[Any] = ContextVar(
    "multivac_behavior_snapshot", default=None
)


def _snapshot():
    return _ACTIVE_SNAPSHOT.get() or BEHAVIOR_REGISTRY.snapshot()


@asynccontextmanager
async def behavior_request_scope():
    existing = _ACTIVE_SNAPSHOT.get()
    if existing is not None:
        yield existing
        return
    snapshot = BEHAVIOR_REGISTRY.snapshot()
    token = _ACTIVE_SNAPSHOT.set(snapshot)
    try:
        yield snapshot
    finally:
        _ACTIVE_SNAPSHOT.reset(token)


async def dispatch_event(name: str, fallback: Callable[..., Any], *args, **kwargs):
    async with behavior_request_scope() as snapshot:
        if snapshot.get("events", name) is not None:
            return await BEHAVIOR_REGISTRY.invoke(
                "events", name, *args, snapshot=snapshot, **kwargs
            )
        result = fallback(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


async def dispatch_intent_override(name: str, context: Any) -> tuple[bool, Any]:
    snapshot = _snapshot()
    if snapshot.get("intents", name) is None:
        return False, None
    return True, await BEHAVIOR_REGISTRY.invoke(
        "intents", name, context, snapshot=snapshot
    )


async def invoke_provider(
    name: str, fallback: Callable[..., Any], *args, **kwargs
) -> Any:
    user_id = get_request_context().get("user_id")
    async with provider_capacity(user_id):
        from services.server_effects import decorate_provider_call
        args, kwargs = decorate_provider_call(name, args, kwargs)
        snapshot = _snapshot()
        if snapshot.get("providers", name) is not None:
            return await BEHAVIOR_REGISTRY.invoke(
                "providers", name, *args, snapshot=snapshot, **kwargs
            )
        result = fallback(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def get_runtime_setting(name: str, default: Any = None) -> Any:
    value = _snapshot().get("settings", name)
    return default if value is None else value


__all__ = [
    "BEHAVIOR_REGISTRY",
    "behavior_request_scope",
    "dispatch_event",
    "dispatch_intent_override",
    "invoke_provider",
    "get_runtime_setting",
]
