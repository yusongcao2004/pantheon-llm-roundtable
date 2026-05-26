"""Configuration schema and loader for Pantheon.

YAML is the source of truth. Environment variables are interpolated via the
``${VAR_NAME}`` syntax. Pydantic validates the merged result.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Env interpolation
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interpolate_env(raw: Any) -> Any:
    """Recursively replace ``${VAR}`` markers with env values.

    Missing variables are replaced with an empty string. Pydantic validators
    downstream are responsible for catching required-but-empty fields.
    """
    if isinstance(raw, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), raw)
    if isinstance(raw, list):
        return [_interpolate_env(item) for item in raw]
    if isinstance(raw, dict):
        return {key: _interpolate_env(value) for key, value in raw.items()}
    return raw


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TelegramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_chat_id: int = Field(..., description="Telegram group chat ID (negative).")
    god_user_id: int | None = Field(
        default=None,
        description="If set, only this user can issue /discuss. Otherwise any group member.",
    )

    @field_validator("group_chat_id", mode="before")
    @classmethod
    def _coerce_chat_id(cls, v: Any) -> int:
        if v is None or v == "":
            raise ValueError(
                "telegram.group_chat_id is required (set TELEGRAM_GROUP_CHAT_ID)"
            )
        return int(v)

    @field_validator("god_user_id", mode="before")
    @classmethod
    def _coerce_god_id(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        return int(v)


class LLMConfig(BaseModel):
    """One participating LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Internal identifier, must be unique.")
    display_name: str = Field(..., description="Shown to humans in logs and prompts.")
    adapter: Literal["openai", "google"] = Field(
        ..., description="Which adapter implementation to use."
    )
    model: str = Field(..., description="Provider-specific model identifier.")
    bot_token: str = Field(..., description="Telegram bot token for this LLM's persona.")
    api_key: str = Field(..., description="Provider API key.")
    base_url: str | None = Field(
        default=None,
        description="Override base URL (required for non-OpenAI providers using openai adapter).",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, ge=64, le=8192)

    @field_validator("name")
    @classmethod
    def _name_must_be_identifier(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"LLM name {v!r} must be alphanumeric (underscores ok).")
        return v

    @field_validator("bot_token", "api_key")
    @classmethod
    def _no_empty_secrets(cls, v: str) -> str:
        if not v:
            raise ValueError("Empty bot_token or api_key — check .env interpolation.")
        return v


class AntiSycophancyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    convergence_window: int = Field(default=4, ge=2, le=10)
    convergence_threshold: int = Field(default=3, ge=1, le=10)
    challenge_injection: bool = True
    position_label_required: bool = True


class DiscussionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=10, ge=1, le=50)
    require_all_check: bool = True
    window_size_turns: int = Field(default=6, ge=2, le=30)
    summary_trigger_rounds: int = Field(default=2, ge=1, le=10)
    summary_max_tokens: int = Field(default=500, ge=100, le=2000)
    anti_sycophancy: AntiSycophancyConfig = Field(default_factory=AntiSycophancyConfig)


class SummarizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: Literal["openai", "google"] = "google"
    model: str = "gemini-3.1-flash-lite"
    api_key: str = Field(..., description="API key for the summarizer's provider.")
    base_url: str | None = None
    max_output_tokens: int = Field(default=600, ge=100, le=2000)

    @field_validator("api_key")
    @classmethod
    def _no_empty_key(cls, v: str) -> str:
        if not v:
            raise ValueError("Summarizer api_key is empty — check .env interpolation.")
        return v


class CachingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    gemini_cache_ttl_seconds: int = Field(default=1800, ge=300, le=3600)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    json_format: bool = False
    file_path: str | None = "logs/pantheon.log"


class PantheonConfig(BaseModel):
    """Top-level configuration aggregating all sections."""

    model_config = ConfigDict(extra="forbid")

    telegram: TelegramConfig
    llms: list[LLMConfig] = Field(..., min_length=2)
    discussion: DiscussionConfig = Field(default_factory=DiscussionConfig)
    summarizer: SummarizerConfig
    caching: CachingConfig = Field(default_factory=CachingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("llms")
    @classmethod
    def _llm_names_unique(cls, v: list[LLMConfig]) -> list[LLMConfig]:
        names = [llm.name for llm in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate LLM names: {set(duplicates)}")
        return v


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(
    yaml_path: str | Path = "config/pantheon.yaml",
    env_path: str | Path | None = ".env",
) -> PantheonConfig:
    """Load configuration from YAML + .env into a validated PantheonConfig.

    Args:
        yaml_path: Path to the main YAML file.
        env_path: Path to a .env file. Pass None to skip dotenv loading
            (env vars must already be exported).

    Raises:
        FileNotFoundError: if the YAML file is missing.
        pydantic.ValidationError: if any field fails validation.
    """
    if env_path is not None:
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(env_file, override=False)

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_file.resolve()}")

    with yaml_file.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    interpolated = _interpolate_env(raw)
    return PantheonConfig.model_validate(interpolated)
