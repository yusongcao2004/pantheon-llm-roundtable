"""OpenAI-compatible adapter.

Reused for any provider that speaks OpenAI's chat-completions wire protocol:

- OpenAI (ChatGPT, https://api.openai.com/v1)
- DeepSeek (https://api.deepseek.com)
- Doubao / 字节豆包 (https://ark.cn-beijing.volces.com/api/v3)
- Kimi / Moonshot (https://api.moonshot.cn/v1)
- Qwen (https://dashscope.aliyuncs.com/compatible-mode/v1)
- Local Ollama, Groq, Together, etc.

Prompt caching is automatic on these providers when the prefix is long enough
(~1024 tokens for OpenAI). We surface cache hit counts from ``usage`` when the
provider reports them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import structlog
from openai import AsyncOpenAI

from pantheon.adapters.base import LLMAdapter, LLMMessage, LLMResponse

if TYPE_CHECKING:
    from pantheon.config import LLMConfig, SummarizerConfig

log = structlog.get_logger(__name__)


def _extract_cached_tokens(usage: Any) -> int:
    """Pull cached-token count out of a provider's usage object.

    OpenAI: ``usage.prompt_tokens_details.cached_tokens``
    DeepSeek: ``usage.prompt_cache_hit_tokens``
    Doubao: mirrors OpenAI's shape.

    Returns 0 if the provider doesn't report cache stats.
    """
    if usage is None:
        return 0
    # OpenAI / Doubao style
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            return int(cached)
    # DeepSeek style
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is not None:
        return int(cached)
    return 0


class OpenAICompatibleAdapter(LLMAdapter):
    """Adapter for any OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        api_key: str,
        base_url: str | None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 800,
    ) -> None:
        self.name = name
        self.model = model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_config(cls, cfg: LLMConfig | SummarizerConfig) -> Self:
        # Both LLMConfig and SummarizerConfig share the fields we need.
        name = getattr(cfg, "name", "summarizer")
        temperature = getattr(cfg, "temperature", 0.5)
        max_tokens = getattr(cfg, "max_output_tokens", 600)
        return cls(
            name=name,
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            default_temperature=temperature,
            default_max_tokens=max_tokens,
        )

    async def chat(
        self,
        *,
        system: str,
        history: list[LLMMessage],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        cache_key: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        token_limit = max_output_tokens or self._default_max_tokens
        token_kwargs = (
            {"max_completion_tokens": token_limit}
            if self.name == "chatgpt" and self.model.startswith("gpt-5")
            else {"max_tokens": token_limit}
        )
        provider_kwargs: dict[str, Any] = {}
        if self.name == "doubao":
            provider_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=cast(Any, messages),  # SDK uses TypedDict; runtime is fine
                temperature=temperature if temperature is not None else self._default_temperature,
                **token_kwargs,
                **provider_kwargs,
            )
        except Exception as exc:
            log.exception(
                "openai_compat.chat_failed",
                adapter=self.name,
                model=self.model,
                error=str(exc),
            )
            raise

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = completion.usage
        prompt_tokens = int(usage.prompt_tokens) if usage else 0
        completion_tokens = int(usage.completion_tokens) if usage else 0
        cached_tokens = _extract_cached_tokens(usage)

        log.info(
            "openai_compat.chat_done",
            adapter=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            finish_reason=choice.finish_reason,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            raw={"finish_reason": choice.finish_reason},
        )

    async def aclose(self) -> None:
        await self._client.close()
