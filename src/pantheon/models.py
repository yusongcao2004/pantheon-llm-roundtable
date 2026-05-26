"""Shared dataclasses for the discussion scene."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class Position(StrEnum):
    """The 【立场】 label every LLM must prefix its reply with."""

    SUPPORT = "支持"
    OPPOSE = "反对"
    NEUTRAL = "中立"
    QUESTION = "质疑"
    CHECK = "check"
    UNKNOWN = "未知"  # When parsing failed (e.g. LLM forgot the label)

    @classmethod
    def parse(cls, raw: str) -> Position:
        """Map a string (Chinese or English shortcut) to a Position."""
        value = raw.strip().lower()
        mapping = {
            "支持": cls.SUPPORT,
            "support": cls.SUPPORT,
            "反对": cls.OPPOSE,
            "oppose": cls.OPPOSE,
            "中立": cls.NEUTRAL,
            "neutral": cls.NEUTRAL,
            "质疑": cls.QUESTION,
            "question": cls.QUESTION,
            "check": cls.CHECK,
        }
        return mapping.get(value, cls.UNKNOWN)


@dataclass
class Utterance:
    """One LLM's contribution to a discussion."""

    speaker: str               # LLM name (LLMConfig.name)
    display_name: str          # Human-readable name shown to other LLMs
    content: str               # Full text of the message
    position: Position         # Parsed from 【立场】 prefix
    round_index: int           # 0-based round counter
    timestamp: float = field(default_factory=time.time)

    @property
    def is_check(self) -> bool:
        return self.position == Position.CHECK


@dataclass
class DiscussionState:
    """Mutable state of an ongoing discussion."""

    topic: str
    utterances: list[Utterance] = field(default_factory=list)
    summary: str = ""                       # Compressed early-discussion digest
    current_round: int = 0
    terminated: bool = False
    termination_reason: str | None = None
    # Tracks which LLMs have signalled check in the current round.
    checked_this_round: set[str] = field(default_factory=set)
    # User interventions buffered for the next speaker's context.
    pending_injections: list[str] = field(default_factory=list)
    # Set of LLM names the user wants skipped in the current round.
    skip_this_round: set[str] = field(default_factory=set)
    # When clear, processing pauses until /resume sets the event.
    # Lazily created so dataclass default_factory stays cheap.
    _resume_event: object = None  # type: ignore[assignment]  # populated on first access

    def add(self, utterance: Utterance) -> None:
        self.utterances.append(utterance)
        if utterance.is_check:
            self.checked_this_round.add(utterance.speaker)

    def reset_round(self) -> None:
        """Called when a round completes (last speaker spoken)."""
        self.checked_this_round.clear()
        self.skip_this_round.clear()
        self.current_round += 1

    def terminate(self, reason: str) -> None:
        self.terminated = True
        self.termination_reason = reason

    # ----- Pause / resume via asyncio.Event -------------------------------

    def resume_event(self) -> object:
        """Return the asyncio.Event used to gate paused execution.

        Created lazily so the dataclass can be constructed outside a running
        loop. Set ('resumed') by default; clear() to pause.
        """
        import asyncio

        if self._resume_event is None:
            ev = asyncio.Event()
            ev.set()  # Start in resumed state.
            self._resume_event = ev
        return self._resume_event

    def pause(self) -> None:
        ev = self.resume_event()
        ev.clear()  # type: ignore[attr-defined]

    def resume(self) -> None:
        ev = self.resume_event()
        ev.set()  # type: ignore[attr-defined]

    @property
    def is_paused(self) -> bool:
        if self._resume_event is None:
            return False
        return not self._resume_event.is_set()  # type: ignore[attr-defined]
