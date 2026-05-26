"""Anti-sycophancy (light): three layers.

Layer 1 — challenge injection: appended to the system prompt of every speaker.
Layer 2 — convergence detection: keyword-based scan of recent utterances; when
            agreement signals dominate the recent window, the next speaker
            receives an override instruction to take an opposing position.
Layer 3 — position-label enforcement: every reply must start with 【立场: X】.
            Parsing happens here too.

All three are zero-LLM-call (no extra inference cost).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pantheon.config import AntiSycophancyConfig
from pantheon.models import Position, Utterance

# Keywords that signal "I agree with what was just said".
# Order-insensitive; case-folded comparison; substring match.
AGREEMENT_KEYWORDS_ZH: tuple[str, ...] = (
    "我同意",
    "完全同意",
    "完全正确",
    "深以为然",
    "没错",
    "确实如此",
    "说得对",
    "说得好",
    "言之有理",
    "我赞同",
    "你说得对",
)

AGREEMENT_KEYWORDS_EN: tuple[str, ...] = (
    "i agree",
    "you're right",
    "you are right",
    "exactly",
    "well said",
    "spot on",
    "couldn't agree more",
    "absolutely right",
)

POSITIONS_THAT_SIGNAL_AGREEMENT: frozenset[Position] = frozenset(
    {Position.SUPPORT, Position.CHECK}
)


@dataclass
class ConvergenceCheck:
    """Output of the convergence watcher."""

    triggered: bool
    agreement_count: int
    window_size: int

    @property
    def ratio(self) -> float:
        if self.window_size == 0:
            return 0.0
        return self.agreement_count / self.window_size


def _has_agreement_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in AGREEMENT_KEYWORDS_EN) or any(
        kw in text for kw in AGREEMENT_KEYWORDS_ZH
    )


def check_convergence(
    recent_utterances: list[Utterance],
    cfg: AntiSycophancyConfig,
) -> ConvergenceCheck:
    """Decide whether the next speaker should be forced into opposition.

    Combines two signals:
    - Agreement keywords appearing in the text.
    - 【立场】 labels falling in {支持, check}.

    A signal counts if EITHER is present in that utterance.
    """
    if not cfg.enabled:
        return ConvergenceCheck(triggered=False, agreement_count=0, window_size=0)

    window = recent_utterances[-cfg.convergence_window :]
    count = 0
    for u in window:
        if u.position in POSITIONS_THAT_SIGNAL_AGREEMENT:
            count += 1
        elif _has_agreement_keyword(u.content):
            count += 1
    triggered = count >= cfg.convergence_threshold
    return ConvergenceCheck(
        triggered=triggered,
        agreement_count=count,
        window_size=len(window),
    )


# ---------------------------------------------------------------------------
# Layer 3 — position label parsing
# ---------------------------------------------------------------------------

# Matches:
#   【立场: 支持】   → SUPPORT
#   【立场：反对】   → OPPOSE (full-width colon)
#   【立场: check】  → CHECK
#   【position: support】 → SUPPORT (lenient)
_LABEL_PATTERN = re.compile(
    r"【\s*(?:立场|position)\s*[:：]\s*([^\]】]+?)\s*[】\]]",
    flags=re.IGNORECASE,
)


def parse_position(reply_text: str) -> tuple[Position, str]:
    """Extract the position label and return (position, body_without_label).

    If no label is found, returns (Position.UNKNOWN, original_text).
    """
    match = _LABEL_PATTERN.search(reply_text)
    if not match:
        return Position.UNKNOWN, reply_text.strip()

    position = Position.parse(match.group(1))
    # Strip the matched label and any leading whitespace/newlines.
    body = reply_text[: match.start()] + reply_text[match.end() :]
    return position, body.strip()
