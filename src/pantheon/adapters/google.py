"""Google adapter for Gemini.

Uses the new ``google-genai`` SDK (unified AI Studio / Vertex AI).

Caching note: Gemini exposes prompt caching via the ``CachedContent`` API,
not as an automatic prefix match. The adapter creates a CachedContent per
``cache_key`` on first use and reuses it for subsequent calls until the TTL
expires. Cache creation may fail (model too small, content under the minimum
token threshold, etc.) — in that case we fall back to non-cached calls.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Self

import structlog
from google import genai
from google.genai import types as genai_types

from pantheon.adapters.base import LLMAdapter, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from pantheon.config import LLMConfig, SummarizerConfig

log = structlog.get_logger(__name__)


# Gemini's role names differ from OpenAI's: it uses 'user' and 'model'.
def _to_gemini_role(role: str) -> str:
    return "model" if role == "assistant" else "user"


def _build_contents(history: list[LLMMessage]) -> list[genai_types.Content]:
    """Translate our normalised history into Gemini's Content list."""
    contents: list[genai_types.Content] = []
    for msg in history:
        contents.append(
            genai_types.Content(
                role=_to_gemini_role(msg.role),
                parts=[genai_types.Part.from_text(text=msg.content)],
            )
        )
    return contents


class GoogleAdapter(LLMAdapter):
    """Gemini adapter (google-genai SDK)."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        default_temperature: float = 0.7,
        default_max_tokens: int = 800,
        cache_ttl_seconds: int = 1800,
    ) -> None:
        self.name = name
        self.model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = genai.Client(api_key=api_key)
        self._cache_ttl_seconds = cache_ttl_seconds
        # Maps cache_key -> (cache_name, created_at_unix)
        self._cache_index: dict[str, tuple[str, float]] = {}

    @classmethod
    def from_config(cls, cfg: LLMConfig | SummarizerConfig) -> Self:
        name = getattr(cfg, "name", "summarizer")
        temperature = getattr(cfg, "temperature", 0.5)
        max_tokens = getattr(cfg, "max_output_tokens", 600)
        return cls(
            name=name,
            model=cfg.model,
            api_key=cfg.api_key,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    async def _get_or_create_cache(
        self,
        cache_key: str,
        system: str,
        prefix_contents: list[genai_types.Content],
    ) -> str | None:
        """Return a usable cache name, creating one if necessary.

        Returns ``None`` if cache creation fails (we'll fall back to a
        normal call).
        """
        now = time.time()

        cached = self._cache_index.get(cache_key)
        if cached is not None:
            cache_name, created_at = cached
            if now - created_at < self._cache_ttl_seconds:
                return cache_name
            # TTL expired — drop the entry and create fresh.
            self._cache_index.pop(cache_key, None)

        try:
            cache = await self._client.aio.caches.create(
                model=self.model,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=system,
                    contents=prefix_contents,
                    ttl=f"{self._cache_ttl_seconds}s",
                ),
            )
            assert cache.name is not None
            self._cache_index[cache_key] = (cache.name, now)
            log.info(
                "google.cache_created",
                adapter=self.name,
                cache_key=cache_key,
                cache_name=cache.name,
            )
            return cache.name
        except Exception as exc:
            log.warning(
                "google.cache_create_failed",
                adapter=self.name,
                cache_key=cache_key,
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def chat(
        self,
        *,
        system: str,
        history: list[LLMMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        contents = _build_contents(history)
        temp = temperature if temperature is not None else self._default_temperature
        max_tokens = max_output_tokens or self._default_max_tokens

        # Attempt to use cached prefix when a cache_key is supplied and
        # there's enough content to make caching worthwhile.
        cache_name: str | None = None
        if cache_key and len(contents) >= 2:
            # Use everything except the last message as the cacheable prefix.
            prefix = contents[:-1]
            tail = contents[-1:]
            cache_name = await self._get_or_create_cache(cache_key, system, prefix)
            if cache_name is not None:
                contents = tail

        if self.model.startswith("gemini-3"):
            config_kwargs: dict[str, Any] = {
                "max_output_tokens": max_tokens,
                "thinking_config": genai_types.ThinkingConfig(thinking_level="low"),
            }
        else:
            config_kwargs = {
                "temperature": temp,
                "max_output_tokens": max_tokens,
            }
        if cache_name is not None:
            config_kwargs["cached_content"] = cache_name
        else:
            # Without a cache, system instruction goes inline.
            config_kwargs["system_instruction"] = system

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            log.exception(
                "google.chat_failed",
                adapter=self.name,
                model=self.model,
                error=str(exc),
            )
            raise

        text = response.text or ""
        usage = response.usage_metadata
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)

        log.info(
            "google.chat_done",
            adapter=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cache_used=cache_name is not None,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            raw={"cache_name": cache_name} if cache_name else {},
        )

    async def aclose(self) -> None:
        # google-genai SDK manages its own connections; nothing to close.
        pass
