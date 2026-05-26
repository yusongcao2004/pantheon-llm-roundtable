"""Entry point: ``python -m pantheon``."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import structlog

from pantheon.app import PantheonApp
from pantheon.config import load_config
from pantheon.logging_setup import configure_logging


async def _run(config_path: Path, env_path: Path | None) -> int:
    config = load_config(config_path, env_path)
    configure_logging(config.logging)
    log = structlog.get_logger("pantheon.main")
    log.info(
        "pantheon.starting",
        n_llms=len(config.llms),
        llms=[c.name for c in config.llms],
        group_chat_id=config.telegram.group_chat_id,
    )

    app = PantheonApp(config)
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int) -> None:
        log.info("pantheon.signal_received", signal=sig)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig)
        except NotImplementedError:
            # Windows / restricted environments — fall back to default.
            pass

    try:
        await app.start()
        log.info("pantheon.ready")
        await shutdown_event.wait()
    finally:
        log.info("pantheon.shutting_down")
        await app.stop()
        log.info("pantheon.stopped")

    return 0


def main() -> None:
    """Synchronous wrapper for the ``pantheon`` console script."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="pantheon",
        description="A theatre of LLMs. One human god, N AI performers.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config/pantheon.yaml"),
        help="Path to the YAML config file (default: config/pantheon.yaml)",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file (default: .env; pass /dev/null to skip)",
    )
    args = parser.parse_args()

    env_path: Path | None = args.env if args.env.exists() else None
    exit_code = asyncio.run(_run(args.config, env_path))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
