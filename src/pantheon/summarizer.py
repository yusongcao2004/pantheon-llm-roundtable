"""Discussion summariser.

Compresses early-round utterances into a tight structured digest. Runs once
per completed round (not every turn). The output replaces the corresponding
utterances in subsequent context-building cycles.
"""

from __future__ import annotations

import structlog

from pantheon.adapters import build_adapter
from pantheon.adapters.base import LLMAdapter, LLMMessage
from pantheon.config import SummarizerConfig
from pantheon.models import DiscussionState, Utterance
from pantheon.prompts import (
    CONCLUSION_SYSTEM,
    CONCLUSION_USER,
    SUMMARIZER_SYSTEM,
    SUMMARIZER_USER,
    format_utterance_for_summarizer,
)

log = structlog.get_logger(__name__)


class Summarizer:
    """Wraps a cheap LLM adapter and produces structured discussion digests."""

    def __init__(self, cfg: SummarizerConfig) -> None:
        self._cfg = cfg
        self._adapter: LLMAdapter = build_adapter(cfg)

    @property
    def adapter(self) -> LLMAdapter:
        return self._adapter

    async def summarize(
        self,
        *,
        state: DiscussionState,
        utterances_to_compress: list[Utterance],
    ) -> str:
        """Produce a digest of the given utterances.

        Returns the summary text on success, or the current state.summary
        (unchanged) on failure — summary failures are non-fatal.
        """
        if not utterances_to_compress:
            return state.summary

        start_round = utterances_to_compress[0].round_index + 1
        end_round = utterances_to_compress[-1].round_index + 1

        system = SUMMARIZER_SYSTEM.format(
            start_round=start_round,
            end_round=end_round,
            max_tokens_hint=self._cfg.max_output_tokens,
        )
        utterances_block = "\n".join(
            format_utterance_for_summarizer(u) for u in utterances_to_compress
        )
        user_msg = SUMMARIZER_USER.format(
            topic=state.topic,
            utterances_block=utterances_block,
        )

        try:
            response = await self._adapter.chat(
                system=system,
                history=[LLMMessage(role="user", content=user_msg)],
                max_output_tokens=self._cfg.max_output_tokens,
                temperature=0.3,  # Low temp for faithful summarisation
            )
        except Exception as exc:
            log.warning(
                "summarizer.failed",
                error=str(exc),
                start_round=start_round,
                end_round=end_round,
            )
            return state.summary

        new_summary = response.text.strip()
        log.info(
            "summarizer.done",
            start_round=start_round,
            end_round=end_round,
            input_utterances=len(utterances_to_compress),
            output_chars=len(new_summary),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

        # If we already had a summary, prepend it so historical digests don't
        # vanish on each compression cycle.
        if state.summary:
            return f"{state.summary}\n\n{new_summary}"
        return new_summary

    async def conclude(
        self,
        *,
        state: DiscussionState,
    ) -> tuple[str, int, int, int]:
        """Produce a concise final conclusion after the discussion ends."""
        if not state.utterances:
            return "【最终结论】\n没有足够发言形成结论。", 0, 0, 0

        utterances_block = "\n".join(
            format_utterance_for_summarizer(u) for u in state.utterances
        )
        user_msg = CONCLUSION_USER.format(
            topic=state.topic,
            utterances_block=utterances_block,
        )

        try:
            response = await self._adapter.chat(
                system=CONCLUSION_SYSTEM,
                history=[LLMMessage(role="user", content=user_msg)],
                max_output_tokens=min(self._cfg.max_output_tokens, 400),
                temperature=0.2,
            )
        except Exception as exc:
            log.warning("conclusion.failed", error=str(exc))
            return "【最终结论】\n最终结论生成失败，请查看上方讨论。", 0, 0, 0

        conclusion = response.text.strip() or "【最终结论】\n未生成明确结论。"
        log.info(
            "conclusion.done",
            output_chars=len(conclusion),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        return (
            conclusion,
            response.prompt_tokens,
            response.completion_tokens,
            response.cached_tokens,
        )

    async def aclose(self) -> None:
        await self._adapter.aclose()
