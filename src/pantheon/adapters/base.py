"""Abstract base for LLM adapters.

An adapter takes a normalised list of :class:`LLMMessage` plus a system prompt
and returns an :class:`LLMResponse`. It must also report whether a prompt
cache hit occurred (for cost accounting) when the provider exposes that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from pantheon.config import LLMConfig, SummarizerConfig


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    """A single message in conversation history.

    Used uniformly across adapters; each adapter translates to its own provider
    schema.
    """

    role: Role
    content: str
    # Free-form name tag for context (e.g. which LLM previously said this).
    # Adapters MAY surface it in the wire format if the provider supports it.
    speaker: str | None = None


@dataclass
class LLMResponse:
    """Normalised response from any adapter."""

    text: str
    model: str
    # Token accounting (best effort; some providers don't return all fields).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # Provider-specific raw payload, kept for debugging.
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_ratio(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens


class LLMAdapter(ABC):
    """Adapter contract.

    Implementations must be safe to use from multiple asyncio tasks
    concurrently (i.e. don't share mutable state across :meth:`chat` calls).
    """

    name: str
    """Human-readable name for logs (typically the LLMConfig.name)."""

    model: str
    """Provider-specific model identifier."""

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: LLMConfig | SummarizerConfig) -> Self:
        """Construct from a config block."""

    @abstractmethod
    async def chat(
        self,
        *,
        system: str,
        history: list[LLMMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            system: System prompt (persona, instructions, position-label rule).
            history: Conversation messages in chronological order.
            max_output_tokens: Override the default output cap.
            temperature: Override the default sampling temperature.
            cache_key: Stable identifier for the (system + early-history)
                prefix. Adapters that support explicit caching (e.g. Gemini)
                use this as the cache content's name/id.

        Returns:
            Normalised :class:`LLMResponse`.
        """

    async def aclose(self) -> None:  # noqa: B027 — intentional default no-op
        """Optional teardown. Override for adapters holding heavy resources."""
