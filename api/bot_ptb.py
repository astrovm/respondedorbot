from __future__ import annotations

import asyncio
import importlib
import logging
import os
from typing import Any, Dict, Optional, Sequence

from api.index import (
    admin_report,
    config_redis,
    handle_callback_query,
    handle_msg,
    handle_pre_checkout_query,
)


logger = logging.getLogger(__name__)


def _update_to_dict(update: Any) -> Dict[str, Any]:
    return update.to_dict()


async def _run_sync(func: Any, *args: Any) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


async def _async_handle_message(update: Any, context: Any) -> None:
    del context
    if update.message is None:
        return

    update_dict = _update_to_dict(update)

    try:
        await _run_sync(handle_msg, update_dict.get("message", {}))
    except Exception as error:
        logger.exception("Error handling PTB message update")
        try:
            await _run_sync(admin_report, "PTB message handler error", error)
        except Exception:
            logger.exception("Failed to report PTB message handler error")


async def _async_handle_callback_query(update: Any, context: Any) -> None:
    del context
    if update.callback_query is None:
        return

    update_dict = _update_to_dict(update)

    try:
        await _run_sync(handle_callback_query, update_dict.get("callback_query", {}))
    except Exception as error:
        logger.exception("Error handling PTB callback query update")
        try:
            await _run_sync(admin_report, "PTB callback handler error", error)
        except Exception:
            logger.exception("Failed to report PTB callback handler error")


async def _async_handle_pre_checkout_query(update: Any, context: Any) -> None:
    del context
    if update.pre_checkout_query is None:
        return

    update_dict = _update_to_dict(update)

    try:
        await _run_sync(
            handle_pre_checkout_query,
            update_dict.get("pre_checkout_query", {}),
        )
    except Exception as error:
        logger.exception("Error handling PTB pre-checkout update")
        try:
            await _run_sync(admin_report, "PTB pre-checkout handler error", error)
        except Exception:
            logger.exception("Failed to report PTB pre-checkout handler error")


async def _post_init(application: Any) -> None:
    del application
    logger.info("PTB application initialized")


async def _error_handler(update: object, context: Any) -> None:
    update_id = getattr(update, "update_id", "unknown")
    logger.exception(
        "Unhandled PTB exception while processing update_id=%s",
        update_id,
        exc_info=context.error,
    )

    try:
        await _run_sync(
            admin_report,
            f"PTB unhandled exception (update_id={update_id})",
            context.error,
        )
    except Exception:
        logger.exception("Failed to report PTB unhandled exception")


def create_application(
    token: Optional[str] = None,
    redis_client: Optional[Any] = None,
) -> Any:
    resolved_token = token or os.environ.get("TELEGRAM_TOKEN")
    if not resolved_token:
        raise ValueError("TELEGRAM_TOKEN not provided")

    resolved_redis = redis_client if redis_client is not None else config_redis()

    telegram_ext = importlib.import_module("telegram.ext")
    ApplicationBuilder = getattr(telegram_ext, "ApplicationBuilder")
    CallbackQueryHandler = getattr(telegram_ext, "CallbackQueryHandler")
    MessageHandler = getattr(telegram_ext, "MessageHandler")
    PreCheckoutQueryHandler = getattr(telegram_ext, "PreCheckoutQueryHandler")
    filters = getattr(telegram_ext, "filters")

    application = (
        ApplicationBuilder()
        .token(resolved_token)
        .concurrent_updates(True)
        .post_init(_post_init)
        .build()
    )
    application.bot_data["redis"] = resolved_redis

    application.add_handler(MessageHandler(filters.ALL, _async_handle_message), group=0)
    application.add_handler(
        CallbackQueryHandler(_async_handle_callback_query),
        group=1,
    )
    application.add_handler(
        PreCheckoutQueryHandler(_async_handle_pre_checkout_query),
        group=2,
    )
    application.add_error_handler(_error_handler)

    return application


def run_polling(
    token: Optional[str] = None,
    redis_client: Optional[Any] = None,
    drop_pending_updates: bool = True,
    allowed_updates: Optional[Sequence[str]] = None,
) -> None:
    application = create_application(token=token, redis_client=redis_client)
    logger.info("Starting PTB polling runtime")
    application.run_polling(
        drop_pending_updates=drop_pending_updates,
        allowed_updates=list(allowed_updates) if allowed_updates is not None else None,
    )
