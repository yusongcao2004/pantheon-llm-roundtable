"""Centralised logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from pantheon.config import LoggingConfig


def configure_logging(cfg: LoggingConfig) -> None:
    """Wire structlog + stdlib logging together.

    - Stdlib loggers (used by python-telegram-bot, openai, etc.) feed into
      structlog so everything ends up in one format.
    - JSON mode for production / file logs; pretty console mode for dev.
    """
    level = getattr(logging, cfg.level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if cfg.json_format:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Stdlib root logger — keep it modest so SDK chatter doesn't drown logs.
    root = logging.getLogger()
    root.setLevel(level)

    # File handler if configured.
    if cfg.file_path:
        log_path = Path(cfg.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(file_handler)

    # Console handler.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    # Tone down very chatty libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext._updater").setLevel(logging.WARNING)
