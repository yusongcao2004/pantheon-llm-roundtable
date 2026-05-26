"""Telegram bot pool.

One Telegram bot per LLM. They all live in the same group chat.
Every bot can receive a /discuss command addressed to itself. The addressed
bot determines which LLM speaks first; subsequent speakers follow in cyclic
order. Operational commands remain on the default narrator bot.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from pantheon.config import LLMConfig, TelegramConfig

log = structlog.get_logger(__name__)


# Async handler signatures — supplied by the application layer.
DiscussCommandFn = Callable[[str, int, str], Awaitable[None]]  # (topic, user_id, starting_llm)
CommandFn = Callable[[str, int], Awaitable[None]]  # (arg_string, user_id) -> None
NoArgCommandFn = Callable[[int], Awaitable[None]]  # (user_id) -> None


@dataclass
class CommandHandlers:
    """Application-level callbacks for each /command.

    Each callback receives the user_id of the issuer so the application can
    enforce the god_user_id restriction.
    """

    on_discuss: DiscussCommandFn   # (topic, user_id, starting_llm)
    on_stop: NoArgCommandFn        # (user_id)
    on_pause: NoArgCommandFn
    on_resume: NoArgCommandFn
    on_skip: CommandFn             # (llm_name, user_id)
    on_inject: CommandFn           # (text, user_id)


class BotPool:
    """Owns N Telegram bots plus a polling Application for command intake."""

    def __init__(
        self,
        *,
        llm_configs: list[LLMConfig],
        telegram_cfg: TelegramConfig,
        handlers: CommandHandlers,
    ) -> None:
        if not llm_configs:
            raise ValueError("BotPool requires at least one LLM config.")
        self._telegram_cfg = telegram_cfg
        self._handlers = handlers

        # Standalone Bot instances for sending messages — one per LLM.
        self._bots: dict[str, Bot] = {
            cfg.name: Bot(token=cfg.bot_token) for cfg in llm_configs
        }
        self._display_names: dict[str, str] = {
            cfg.name: cfg.display_name for cfg in llm_configs
        }

        # Every bot listens for /discuss addressed to itself.
        # The first bot remains the narrator/default command listener.
        self._default_listener_name = llm_configs[0].name
        self._apps: dict[str, Application] = {}
        for cfg in llm_configs:
            app = ApplicationBuilder().token(cfg.bot_token).build()
            self._register_handlers(app, cfg.name)
            self._apps[cfg.name] = app

        # Backwards-compatible alias for the default narrator application.
        self._app = self._apps[self._default_listener_name]

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_as(self, llm_name: str, text: str) -> None:
        """Send `text` to the group as the bot belonging to `llm_name`."""
        bot = self._bots.get(llm_name)
        if bot is None:
            raise KeyError(f"No bot configured for LLM {llm_name!r}.")
        try:
            await bot.send_message(
                chat_id=self._telegram_cfg.group_chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception as exc:
            log.warning(
                "bot_pool.send_failed_md",
                llm=llm_name,
                error=str(exc),
            )
            # Retry without markdown in case of formatting issues.
            await bot.send_message(
                chat_id=self._telegram_cfg.group_chat_id,
                text=text,
            )

    async def send_status(self, text: str) -> None:
        """Send a status message as the first bot (the 'narrator')."""
        first_name = next(iter(self._bots))
        await self.send_as(first_name, text)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        for app in self._apps.values():
            await app.initialize()
            await app.start()
            assert app.updater is not None
            await app.updater.start_polling(drop_pending_updates=True)
        log.info(
            "bot_pool.started",
            n_bots=len(self._bots),
            group_chat_id=self._telegram_cfg.group_chat_id,
        )

    async def stop(self) -> None:
        for app in self._apps.values():
            if app.updater is not None:
                await app.updater.stop()
            await app.stop()
            await app.shutdown()
        log.info("bot_pool.stopped")

    # ------------------------------------------------------------------
    # Command handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self, app: Application, listener_name: str) -> None:
        async def cmd_discuss(
            update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            await self._cmd_discuss(update, context, listener_name)

        # Every bot may start a discussion when directly addressed.
        app.add_handler(CommandHandler("discuss", cmd_discuss))

        # Keep operational commands on the default bot to avoid duplicate handling.
        if listener_name != self._default_listener_name:
            return

        app.add_handler(CommandHandler("stop", self._cmd_stop))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("skip", self._cmd_skip))
        app.add_handler(CommandHandler("inject", self._cmd_inject))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("start", self._cmd_help))

    def _user_id_from(self, update: Update) -> int:
        if update.effective_user is None:
            return 0
        return update.effective_user.id

    def _is_target_chat(self, update: Update) -> bool:
        if update.effective_chat is None:
            return False
        return update.effective_chat.id == self._telegram_cfg.group_chat_id

    @staticmethod
    def _join_args(context: ContextTypes.DEFAULT_TYPE) -> str:
        if not context.args:
            return ""
        return " ".join(context.args)

    # ---- Command callbacks ----

    async def _cmd_discuss(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        starting_llm: str,
    ) -> None:
        if not self._is_target_chat(update):
            return

        message_text = (
            update.effective_message.text
            if update.effective_message is not None
            else ""
        ) or ""
        command_text = message_text.split(maxsplit=1)[0]

        # An unaddressed /discuss defaults to the narrator bot only.
        # Addressed commands such as /discuss@pantheon_deepseek_cao_bot
        # are handled by the bot that receives them.
        if "@" not in command_text and starting_llm != self._default_listener_name:
            return

        topic = self._join_args(context).strip()
        await self._handlers.on_discuss(
            topic, self._user_id_from(update), starting_llm
        )

    async def _cmd_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        await self._handlers.on_stop(self._user_id_from(update))

    async def _cmd_pause(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        await self._handlers.on_pause(self._user_id_from(update))

    async def _cmd_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        await self._handlers.on_resume(self._user_id_from(update))

    async def _cmd_skip(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        target = self._join_args(context).strip()
        await self._handlers.on_skip(target, self._user_id_from(update))

    async def _cmd_inject(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        text = self._join_args(context).strip()
        await self._handlers.on_inject(text, self._user_id_from(update))

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not self._is_target_chat(update):
            return
        help_text = (
            "🏛️ *Pantheon Commands*\n\n"
            "/discuss <topic> — start a new discussion\n"
            "/stop — terminate the current discussion\n"
            "/pause — pause the current discussion\n"
            "/resume — resume after /pause\n"
            "/skip <llm\\_name> — skip a specific LLM this round\n"
            "/inject <text> — inject context for the next speaker\n"
        )
        await self.send_status(help_text)
