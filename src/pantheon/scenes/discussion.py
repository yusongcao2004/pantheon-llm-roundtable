"""Discussion scene — Phase 1's only scene.

Orchestrates: rotation, context building, LLM calls, position parsing,
anti-sycophancy, termination checks, and periodic summarisation.

The scene is intentionally decoupled from Telegram. It emits utterances
via async callbacks supplied at construction time, so it can be tested
without any bot infrastructure.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from pantheon.adapters import build_adapter
from pantheon.adapters.base import LLMAdapter
from pantheon.anti_sycophancy import parse_position
from pantheon.config import DiscussionConfig, LLMConfig, SummarizerConfig
from pantheon.context_builder import build_context
from pantheon.models import DiscussionState, Utterance
from pantheon.summarizer import Summarizer

log = structlog.get_logger(__name__)


OnUtterance = Callable[[Utterance], Awaitable[None]]
OnStatus = Callable[[str], Awaitable[None]]


@dataclass
class DiscussionResult:
    """Summary of a completed discussion."""

    state: DiscussionState
    rounds_completed: int
    total_utterances: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    conclusion: str = ""

    @property
    def cache_savings_ratio(self) -> float:
        if self.total_prompt_tokens == 0:
            return 0.0
        return self.total_cached_tokens / self.total_prompt_tokens


class DiscussionScene:
    """The discussion scene.

    Construct once; call :meth:`run` for each new topic. The scene maintains
    no cross-discussion state in Phase 1.
    """

    def __init__(
        self,
        *,
        llm_configs: list[LLMConfig],
        discussion_cfg: DiscussionConfig,
        summarizer_cfg: SummarizerConfig,
        on_utterance: OnUtterance,
        on_status: OnStatus,
    ) -> None:
        self._llm_configs = llm_configs
        self._discussion_cfg = discussion_cfg
        self._summarizer_cfg = summarizer_cfg
        self._on_utterance = on_utterance
        self._on_status = on_status

        # Build adapter instances lazily on first run to avoid pulling SDK
        # connections during import / config validation.
        self._adapters: dict[str, LLMAdapter] = {}
        self._summarizer: Summarizer | None = None

        # In-flight discussion state (None when idle).
        self._state: DiscussionState | None = None
        self._stop_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Public commands (called from BotPool when /commands arrive)
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._state is not None and not self._state.terminated

    def stop(self) -> None:
        if self._state is not None and not self._state.terminated:
            self._state.terminate("/stop by god")
            # Release any paused awaiter so the loop can exit cleanly.
            self._state.resume()
        if self._stop_event is not None:
            self._stop_event.set()

    def pause(self) -> None:
        if self._state is not None:
            self._state.pause()

    def resume(self) -> None:
        if self._state is not None:
            self._state.resume()

    def skip(self, llm_name: str) -> bool:
        """Skip a specific LLM in this round. Returns True if accepted."""
        if self._state is None:
            return False
        valid = {cfg.name for cfg in self._llm_configs}
        if llm_name not in valid:
            return False
        self._state.skip_this_round.add(llm_name)
        return True

    def inject(self, text: str) -> bool:
        if self._state is None or not text.strip():
            return False
        self._state.pending_injections.append(text.strip())
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self, topic: str, starting_llm: str | None = None
    ) -> DiscussionResult:
        """Execute one full discussion to termination."""
        if self.is_active:
            raise RuntimeError("A discussion is already in progress.")

        self._ensure_adapters()
        state = DiscussionState(topic=topic)
        self._state = state
        self._stop_event = asyncio.Event()

        ordered_llm_configs = list(self._llm_configs)
        if starting_llm is not None:
            llm_names = [cfg.name for cfg in ordered_llm_configs]
            if starting_llm not in llm_names:
                raise ValueError(f"Unknown starting LLM: {starting_llm!r}.")
            start_index = llm_names.index(starting_llm)
            ordered_llm_configs = (
                ordered_llm_configs[start_index:]
                + ordered_llm_configs[:start_index]
            )

        total_prompt = 0
        total_completion = 0
        total_cached = 0
        conclusion = ""

        await self._on_status(
            f"🏛️ Discussion started: 《{topic}》\n"
            f"首发：{ordered_llm_configs[0].display_name}"
        )

        try:
            while not state.terminated and state.current_round < self._discussion_cfg.max_rounds:
                # Honour /pause by awaiting the resume event (no busy-wait).
                if state.is_paused:
                    log.info("discussion.paused")
                    await state.resume_event().wait()  # type: ignore[attr-defined]
                    log.info("discussion.resumed")
                if state.terminated:
                    break

                # Walk through every LLM in the configured order.
                for llm_cfg in ordered_llm_configs:
                    if state.terminated:
                        break
                    if llm_cfg.name in state.skip_this_round:
                        log.info(
                            "discussion.skip_speaker",
                            speaker=llm_cfg.name,
                            round=state.current_round,
                        )
                        continue

                    utterance, p_tok, c_tok, cache_tok = await self._take_turn(
                        state=state, speaker=llm_cfg
                    )
                    total_prompt += p_tok
                    total_completion += c_tok
                    total_cached += cache_tok

                    # Pending injections are consumed once per turn boundary.
                    state.pending_injections.clear()

                    await self._on_utterance(utterance)

                    if self._should_terminate_after(state):
                        break

                if state.terminated:
                    break

                # End of round — check consensus, maybe summarise, advance.
                if self._all_checked(state):
                    state.terminate("consensus: all participants signalled check")
                    break

                await self._maybe_summarise(state)
                state.reset_round()

            if not state.terminated:
                state.terminate(f"max_rounds reached ({self._discussion_cfg.max_rounds})")

            (
                conclusion,
                conclusion_prompt,
                conclusion_completion,
                conclusion_cached,
            ) = await self._summarizer.conclude(state=state)
            total_prompt += conclusion_prompt
            total_completion += conclusion_completion
            total_cached += conclusion_cached

        finally:
            await self._on_status(
                f"🛑 Discussion ended: {state.termination_reason or 'unknown'} "
                f"after {state.current_round} rounds, "
                f"{len(state.utterances)} utterances."
            )
            self._state = None

        return DiscussionResult(
            state=state,
            conclusion=conclusion,
            rounds_completed=state.current_round,
            total_utterances=len(state.utterances),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
        )

    # ------------------------------------------------------------------
    # Per-turn logic
    # ------------------------------------------------------------------

    async def _take_turn(
        self,
        *,
        state: DiscussionState,
        speaker: LLMConfig,
    ) -> tuple[Utterance, int, int, int]:
        ctx = build_context(
            state=state,
            speaker=speaker,
            all_participants=self._llm_configs,
            discussion_cfg=self._discussion_cfg,
        )

        adapter = self._adapters[speaker.name]
        response = await adapter.chat(
            system=ctx.system_prompt,
            history=ctx.history,
            max_output_tokens=speaker.max_output_tokens,
            temperature=speaker.temperature,
            cache_key=ctx.cache_key,
        )

        position, body = parse_position(response.text)
        utterance = Utterance(
            speaker=speaker.name,
            display_name=speaker.display_name,
            content=body if body else response.text,
            position=position,
            round_index=state.current_round,
        )
        state.add(utterance)

        log.info(
            "discussion.turn_done",
            speaker=speaker.name,
            position=position.value,
            round=state.current_round,
            convergence_triggered=ctx.convergence_triggered,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached_tokens=response.cached_tokens,
        )

        return (
            utterance,
            response.prompt_tokens,
            response.completion_tokens,
            response.cached_tokens,
        )

    # ------------------------------------------------------------------
    # Termination checks
    # ------------------------------------------------------------------

    def _all_checked(self, state: DiscussionState) -> bool:
        if not self._discussion_cfg.require_all_check:
            return False
        all_names = {cfg.name for cfg in self._llm_configs}
        # Exclude skipped participants from the consensus requirement.
        required = all_names - state.skip_this_round
        return required.issubset(state.checked_this_round) and len(required) > 0

    def _should_terminate_after(self, state: DiscussionState) -> bool:
        # Currently only handled by /stop (sets state.terminated directly).
        # Hook for future per-utterance termination logic (e.g. safety filters).
        return state.terminated

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    async def _maybe_summarise(self, state: DiscussionState) -> None:
        completed_rounds = state.current_round + 1
        if completed_rounds < self._discussion_cfg.summary_trigger_rounds:
            return
        # Pick everything older than the sliding window for compression.
        keep = self._discussion_cfg.window_size_turns
        if len(state.utterances) <= keep:
            return

        to_compress = state.utterances[:-keep]
        if not to_compress:
            return

        assert self._summarizer is not None
        new_summary = await self._summarizer.summarize(
            state=state, utterances_to_compress=to_compress
        )
        if new_summary != state.summary:
            state.summary = new_summary
            log.info(
                "discussion.summary_updated",
                round=state.current_round,
                summary_chars=len(new_summary),
            )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _ensure_adapters(self) -> None:
        if self._adapters:
            return
        for cfg in self._llm_configs:
            self._adapters[cfg.name] = build_adapter(cfg)
        self._summarizer = Summarizer(self._summarizer_cfg)

    async def aclose(self) -> None:
        for adapter in self._adapters.values():
            await adapter.aclose()
        if self._summarizer is not None:
            await self._summarizer.aclose()
