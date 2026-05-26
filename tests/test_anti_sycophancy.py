"""Tests for the anti-sycophancy layer.

Covers:
- Position label parsing (Chinese + English, half-width + full-width colons)
- Convergence detection on synthetic utterance sequences
"""

from __future__ import annotations

import pytest

from pantheon.anti_sycophancy import (
    ConvergenceCheck,
    check_convergence,
    parse_position,
)
from pantheon.config import AntiSycophancyConfig
from pantheon.models import Position, Utterance

# ---------------------------------------------------------------------------
# Position parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_position, expected_body",
    [
        # Basic Chinese labels
        ("【立场: 支持】\n\nI think this is good.", Position.SUPPORT, "I think this is good."),
        ("【立场: 反对】\n这不对", Position.OPPOSE, "这不对"),
        ("【立场: 中立】\n\nIt depends.", Position.NEUTRAL, "It depends."),
        ("【立场: 质疑】\n\nWhy though?", Position.QUESTION, "Why though?"),
        ("【立场: check】\nDone.", Position.CHECK, "Done."),
        # Full-width colon (common Chinese input)
        ("【立场：支持】\nGood point.", Position.SUPPORT, "Good point."),
        # Whitespace tolerance
        ("【立场:  反对  】\n\nNo.", Position.OPPOSE, "No."),
        # English variant (lenient parsing)
        ("【position: support】\nYes.", Position.SUPPORT, "Yes."),
        # Embedded mid-text (only the first match counts; body strips the label)
        ("Preface 【立场: 反对】\nNo way.", Position.OPPOSE, "Preface \nNo way."),
    ],
)
def test_parse_position_valid_labels(
    raw: str, expected_position: Position, expected_body: str
) -> None:
    position, body = parse_position(raw)
    assert position == expected_position
    # Body comparison: collapse extra whitespace for the embedded-mid-text case
    assert body.strip() == expected_body.strip()


def test_parse_position_no_label_returns_unknown() -> None:
    position, body = parse_position("I forgot to add a label.")
    assert position == Position.UNKNOWN
    assert body == "I forgot to add a label."


def test_parse_position_unrecognised_value_is_unknown() -> None:
    position, _ = parse_position("【立场: 完全不在乎】\nWhatever.")
    assert position == Position.UNKNOWN


def test_parse_position_empty_string() -> None:
    position, body = parse_position("")
    assert position == Position.UNKNOWN
    assert body == ""


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


def _make_utterance(
    *,
    position: Position = Position.OPPOSE,
    content: str = "filler text",
    speaker: str = "speakerX",
    round_index: int = 0,
) -> Utterance:
    return Utterance(
        speaker=speaker,
        display_name=speaker.upper(),
        content=content,
        position=position,
        round_index=round_index,
    )


@pytest.fixture
def default_cfg() -> AntiSycophancyConfig:
    return AntiSycophancyConfig(
        enabled=True,
        convergence_window=4,
        convergence_threshold=3,
    )


def test_convergence_disabled_returns_false(default_cfg: AntiSycophancyConfig) -> None:
    default_cfg.enabled = False
    history = [_make_utterance(position=Position.SUPPORT) for _ in range(4)]
    result = check_convergence(history, default_cfg)
    assert result.triggered is False
    assert result.window_size == 0


def test_convergence_triggers_on_majority_support(
    default_cfg: AntiSycophancyConfig,
) -> None:
    history = [_make_utterance(position=Position.SUPPORT) for _ in range(4)]
    result = check_convergence(history, default_cfg)
    assert isinstance(result, ConvergenceCheck)
    assert result.triggered is True
    assert result.agreement_count == 4
    assert result.window_size == 4


def test_convergence_below_threshold_does_not_trigger(
    default_cfg: AntiSycophancyConfig,
) -> None:
    history = [
        _make_utterance(position=Position.SUPPORT),
        _make_utterance(position=Position.OPPOSE),
        _make_utterance(position=Position.SUPPORT),
        _make_utterance(position=Position.OPPOSE),
    ]
    result = check_convergence(history, default_cfg)
    assert result.triggered is False
    assert result.agreement_count == 2  # only the two SUPPORT turns


def test_convergence_keyword_signal_counts(default_cfg: AntiSycophancyConfig) -> None:
    """Agreement keywords count even when the position label is neutral."""
    history = [
        _make_utterance(position=Position.NEUTRAL, content="我同意上面的观点"),
        _make_utterance(position=Position.NEUTRAL, content="you're right about that"),
        _make_utterance(position=Position.NEUTRAL, content="完全同意"),
        _make_utterance(position=Position.OPPOSE, content="Actually, no."),
    ]
    result = check_convergence(history, default_cfg)
    assert result.agreement_count == 3
    assert result.triggered is True


def test_convergence_window_limits_lookback(
    default_cfg: AntiSycophancyConfig,
) -> None:
    """Only the last `convergence_window` utterances are considered."""
    history = [
        # These shouldn't be counted (outside window).
        *[_make_utterance(position=Position.SUPPORT) for _ in range(10)],
        # Within window of 4.
        _make_utterance(position=Position.OPPOSE),
        _make_utterance(position=Position.OPPOSE),
        _make_utterance(position=Position.OPPOSE),
        _make_utterance(position=Position.OPPOSE),
    ]
    result = check_convergence(history, default_cfg)
    assert result.window_size == 4
    assert result.agreement_count == 0
    assert result.triggered is False


def test_convergence_empty_history(default_cfg: AntiSycophancyConfig) -> None:
    result = check_convergence([], default_cfg)
    assert result.triggered is False
    assert result.window_size == 0
    assert result.ratio == 0.0


def test_convergence_check_ratio() -> None:
    check = ConvergenceCheck(triggered=True, agreement_count=3, window_size=4)
    assert check.ratio == 0.75
    empty = ConvergenceCheck(triggered=False, agreement_count=0, window_size=0)
    assert empty.ratio == 0.0
