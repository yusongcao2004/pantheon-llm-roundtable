"""Tests for config loading and env interpolation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from pantheon.config import _interpolate_env, load_config

# ---------------------------------------------------------------------------
# Env interpolation
# ---------------------------------------------------------------------------


def test_interpolate_replaces_known_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TOKEN", "secret123")
    assert _interpolate_env("${MY_TOKEN}") == "secret123"


def test_interpolate_blank_for_missing_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABSENT_VAR", raising=False)
    assert _interpolate_env("${ABSENT_VAR}") == ""


def test_interpolate_preserves_literal_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "hello")
    assert _interpolate_env("prefix-${X}-suffix") == "prefix-hello-suffix"


def test_interpolate_nested_structures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("V", "world")
    raw = {"a": "${V}", "b": ["plain", "${V}!"], "c": {"d": "${V}"}}
    assert _interpolate_env(raw) == {
        "a": "world",
        "b": ["plain", "world!"],
        "c": {"d": "world"},
    }


def test_interpolate_non_string_passthrough() -> None:
    assert _interpolate_env(42) == 42
    assert _interpolate_env(None) is None
    assert _interpolate_env(True) is True


# ---------------------------------------------------------------------------
# Full load_config
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG = dedent(
    """\
    telegram:
      group_chat_id: ${TG_GROUP}
      god_user_id: null

    llms:
      - name: chatgpt
        display_name: "ChatGPT"
        adapter: openai
        model: gpt-4o-mini
        bot_token: ${BOT_TOKEN_A}
        api_key: ${OPENAI_KEY}
        base_url: https://api.openai.com/v1
      - name: gemini
        display_name: "Gemini"
        adapter: google
        model: gemini-3.5-flash
        bot_token: ${BOT_TOKEN_B}
        api_key: ${GEMINI_KEY}

    summarizer:
      adapter: google
      model: gemini-3.1-flash-lite
      api_key: ${GEMINI_KEY}
    """
)


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "pantheon.yaml"
    cfg.write_text(_MINIMAL_CONFIG)
    return cfg


def test_load_config_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TG_GROUP", "-1001234567890")
    monkeypatch.setenv("BOT_TOKEN_A", "111:aaa")
    monkeypatch.setenv("BOT_TOKEN_B", "222:bbb")
    monkeypatch.setenv("OPENAI_KEY", "sk-test-openai")
    monkeypatch.setenv("GEMINI_KEY", "gemini-test")

    cfg = load_config(_write_config(tmp_path), env_path=None)
    assert cfg.telegram.group_chat_id == -1001234567890
    assert cfg.telegram.god_user_id is None
    assert len(cfg.llms) == 2
    assert cfg.llms[0].name == "chatgpt"
    assert cfg.llms[1].adapter == "google"
    assert cfg.summarizer.model == "gemini-3.1-flash-lite"


def test_load_config_missing_chat_id_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Don't set TG_GROUP — interpolation yields "" and validator should reject.
    monkeypatch.delenv("TG_GROUP", raising=False)
    monkeypatch.setenv("BOT_TOKEN_A", "111:aaa")
    monkeypatch.setenv("BOT_TOKEN_B", "222:bbb")
    monkeypatch.setenv("OPENAI_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_KEY", "gemini-test")

    with pytest.raises(ValidationError, match="group_chat_id"):
        load_config(_write_config(tmp_path), env_path=None)


def test_load_config_empty_api_key_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TG_GROUP", "-100")
    monkeypatch.setenv("BOT_TOKEN_A", "111")
    monkeypatch.setenv("BOT_TOKEN_B", "222")
    # OPENAI_KEY intentionally missing.
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.setenv("GEMINI_KEY", "gemini-test")

    with pytest.raises(ValidationError, match="api_key"):
        load_config(_write_config(tmp_path), env_path=None)


def test_load_config_duplicate_llm_names_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_yaml = _MINIMAL_CONFIG.replace("name: gemini", "name: chatgpt")
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(bad_yaml)

    for var in ("TG_GROUP", "BOT_TOKEN_A", "BOT_TOKEN_B"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("OPENAI_KEY", "k")
    monkeypatch.setenv("GEMINI_KEY", "k")

    with pytest.raises(ValidationError, match="Duplicate LLM names"):
        load_config(cfg_file, env_path=None)


def test_load_config_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/to/config.yaml", env_path=None)


def test_load_config_invalid_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_yaml = _MINIMAL_CONFIG.replace("adapter: google", "adapter: spaghetti")
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(bad_yaml)

    for var, val in [
        ("TG_GROUP", "-1"),
        ("BOT_TOKEN_A", "a"),
        ("BOT_TOKEN_B", "b"),
        ("OPENAI_KEY", "k"),
        ("GEMINI_KEY", "k"),
    ]:
        monkeypatch.setenv(var, val)

    with pytest.raises(ValidationError):
        load_config(cfg_file, env_path=None)
