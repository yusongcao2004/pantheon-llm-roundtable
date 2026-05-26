"""LLM adapters.

Each adapter implements :class:`LLMAdapter` and is registered here. Adding a
new adapter type requires (a) implementing it in a new module and (b) adding
its key to :data:`ADAPTER_REGISTRY`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pantheon.adapters.base import LLMAdapter, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from pantheon.config import LLMConfig, SummarizerConfig

__all__ = ["ADAPTER_REGISTRY", "LLMAdapter", "LLMMessage", "LLMResponse", "build_adapter"]


def _build_openai(cfg: LLMConfig | SummarizerConfig) -> LLMAdapter:
    from pantheon.adapters.openai_compat import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter.from_config(cfg)


def _build_google(cfg: LLMConfig | SummarizerConfig) -> LLMAdapter:
    from pantheon.adapters.google import GoogleAdapter

    return GoogleAdapter.from_config(cfg)


ADAPTER_REGISTRY: dict[str, Callable[[LLMConfig | SummarizerConfig], LLMAdapter]] = {
    "openai": _build_openai,
    "google": _build_google,
}


def build_adapter(cfg: LLMConfig | SummarizerConfig) -> LLMAdapter:
    """Instantiate the right adapter for a config block."""
    try:
        builder = ADAPTER_REGISTRY[cfg.adapter]
    except KeyError as exc:
        raise ValueError(
            f"Unknown adapter {cfg.adapter!r}. "
            f"Registered adapters: {sorted(ADAPTER_REGISTRY)}"
        ) from exc
    return builder(cfg)
