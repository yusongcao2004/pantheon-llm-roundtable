"""Application layer — glues DiscussionScene and BotPool together.

Provides the command handlers that translate Telegram commands into
DiscussionScene method calls, and routes the scene's utterance callbacks
back into BotPool sends.
"""

from __future__ import annotations

import asyncio

import structlog

from pantheon.bot_pool import BotPool, CommandHandlers
from pantheon.config import PantheonConfig
from pantheon.models import Position, Utterance
from pantheon.scenes.discussion import DiscussionScene

log = structlog.get_logger(__name__)


class PantheonApp:
    """The whole application, one instance per process."""

    def __init__(self, config: PantheonConfig) -> None:
        self._config = config
        self._scene = DiscussionScene(
            llm_configs=config.llms,
            discussion_cfg=config.discussion,
            summarizer_cfg=config.summarizer,
            on_utterance=self._on_utterance,
            on_status=self._on_status,
        )
        self._bot_pool = BotPool(
            llm_configs=config.llms,
            telegram_cfg=config.telegram,
            handlers=CommandHandlers(
                on_discuss=self._handle_discuss,
                on_stop=self._handle_stop,
                on_pause=self._handle_pause,
                on_resume=self._handle_resume,
                on_skip=self._handle_skip,
                on_inject=self._handle_inject,
            ),
        )
        self._discussion_task: asyncio.Task[object] | None = None

    # ------------------------------------------------------------------
    # Permission gate
    # ------------------------------------------------------------------

    def _is_god(self, user_id: int) -> bool:
        god = self._config.telegram.god_user_id
        return god is None or user_id == god

    # ------------------------------------------------------------------
    # Telegram → Scene
    # ------------------------------------------------------------------

    async def _handle_discuss(
        self, topic: str, user_id: int, starting_llm: str
    ) -> None:
        if not self._is_god(user_id):
            await self._bot_pool.send_status(
                "⛔ Only the configured god may start a discussion."
            )
            return
        if not topic:
            await self._bot_pool.send_status(
                "Usage: `/discuss <your topic here>`"
            )
            return
        if self._scene.is_active:
            await self._bot_pool.send_status(
                "A discussion is already in progress. Send `/stop` first."
            )
            return

        # Run the discussion as a background task so command handlers stay responsive.
        self._discussion_task = asyncio.create_task(
            self._run_discussion(topic, starting_llm)
        )

    async def _run_discussion(self, topic: str, starting_llm: str) -> None:
        try:
            result = await self._scene.run(topic, starting_llm=starting_llm)
            saved_pct = result.cache_savings_ratio * 100
            await self._bot_pool.send_status(f"🏁 {result.conclusion}")
            await self._bot_pool.send_status(
                f"📊 Stats — rounds: {result.rounds_completed}, "
                f"utterances: {result.total_utterances}, "
                f"input tokens: {result.total_prompt_tokens} "
                f"(cached: {result.total_cached_tokens}, ratio: {saved_pct:.1f}%), "
                f"output tokens: {result.total_completion_tokens}."
            )
        except Exception as exc:
            log.exception("app.discussion_failed", error=str(exc))
            await self._bot_pool.send_status(f"❌ Discussion failed: {exc}")

    async def _handle_stop(self, user_id: int) -> None:
        if not self._is_god(user_id):
            return
        if not self._scene.is_active:
            await self._bot_pool.send_status("No discussion is running.")
            return
        self._scene.stop()

    async def _handle_pause(self, user_id: int) -> None:
        if not self._is_god(user_id):
            return
        if not self._scene.is_active:
            await self._bot_pool.send_status("No discussion is running.")
            return
        self._scene.pause()
        await self._bot_pool.send_status("⏸️ Paused. Send `/resume` to continue.")

    async def _handle_resume(self, user_id: int) -> None:
        if not self._is_god(user_id):
            return
        if not self._scene.is_active:
            return
        self._scene.resume()
        await self._bot_pool.send_status("▶️ Resumed.")

    async def _handle_skip(self, llm_name: str, user_id: int) -> None:
        if not self._is_god(user_id):
            return
        if not llm_name:
            await self._bot_pool.send_status("Usage: `/skip <llm_name>`")
            return
        ok = self._scene.skip(llm_name)
        if ok:
            await self._bot_pool.send_status(f"⏭️ Will skip `{llm_name}` this round.")
        else:
            await self._bot_pool.send_status(
                f"❌ Unknown LLM `{llm_name}` (or no discussion active)."
            )

    async def _handle_inject(self, text: str, user_id: int) -> None:
        if not self._is_god(user_id):
            return
        if not text:
            await self._bot_pool.send_status("Usage: `/inject <text>`")
            return
        ok = self._scene.inject(text)
        if ok:
            await self._bot_pool.send_status(
                f"💉 Injected into the next speaker's context: _{text[:80]}_"
            )
        else:
            await self._bot_pool.send_status("No discussion is running.")

    # ------------------------------------------------------------------
    # Scene → Telegram
    # ------------------------------------------------------------------

    async def _on_utterance(self, utterance: Utterance) -> None:
        body = _format_utterance_for_telegram(utterance)
        try:
            await self._bot_pool.send_as(utterance.speaker, body)
        except Exception as exc:
            log.warning(
                "app.send_utterance_failed",
                speaker=utterance.speaker,
                error=str(exc),
            )

    async def _on_status(self, text: str) -> None:
        await self._bot_pool.send_status(text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await self._bot_pool.start()

    async def stop(self) -> None:
        if self._discussion_task is not None and not self._discussion_task.done():
            self._scene.stop()
            try:
                await asyncio.wait_for(self._discussion_task, timeout=10.0)
            except TimeoutError:
                log.warning("app.discussion_task_did_not_finish")
        await self._scene.aclose()
        await self._bot_pool.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POSITION_EMOJI: dict[Position, str] = {
    Position.SUPPORT: "👍",
    Position.OPPOSE: "👎",
    Position.NEUTRAL: "➖",
    Position.QUESTION: "❓",
    Position.CHECK: "✅",
    Position.UNKNOWN: "❔",
}


def _format_utterance_for_telegram(utterance: Utterance) -> str:
    """Pretty-format an utterance for display in the Telegram group."""
    emoji = _POSITION_EMOJI[utterance.position]
    # Telegram-safe: avoid stray underscores/asterisks tripping Markdown.
    body = utterance.content
    return f"{emoji} *【立场: {utterance.position.value}】*\n\n{body}"
