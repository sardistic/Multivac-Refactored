"""A provider retry proposal; constructing it never starts a provider call."""

from dataclasses import dataclass
from typing import Awaitable, Callable, Any


@dataclass
class ProviderFallback:
    provider: str
    model: str
    reason: str
    retry: Callable[[], Awaitable[Any]]
