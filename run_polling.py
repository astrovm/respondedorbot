#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import importlib
import threading
import time
from typing import Optional


def refresh_price_caches() -> None:
    from api.index import refresh_price_caches as _refresh_price_caches

    _refresh_price_caches()


def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return

    try:
        dotenv = importlib.import_module("dotenv")
        load_dotenv = dotenv.load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    with open(env_path, encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value


def _price_refresh_loop() -> None:
    while True:
        try:
            refresh_price_caches()
        except Exception as e:
            print(f"Price cache refresh error: {e}", file=sys.stderr)
        time.sleep(1800)  # 30 minutes


def main() -> int:
    _load_dotenv()

    token: Optional[str] = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        print("FATAL: TELEGRAM_TOKEN not set", file=sys.stderr)
        return 1

    from api.core.logging import setup_logging

    setup_logging()

    from api import index
    from api.bot.general_commands import gen_random
    from api.bot.ptb import run_polling
    from api.tasks.scheduler import (
        backfill_canonical_task_records,
        get_scheduler,
        init_scheduler,
        rebuild_scheduler_from_canonical_records,
    )

    runtime = index.app_runtime
    threading.Thread(target=_price_refresh_loop, daemon=True).start()
    runtime.summary.start_background_worker()
    runtime.billing_reconciler.start()
    exit_code = 0
    try:
        init_scheduler(
            redis_factory=runtime.config.redis,
            task_executor_deps={
                "ask_ai": runtime.ai.ask,
                "send_msg": runtime.telegram.send_message,
                "admin_report": runtime.admin.report,
                "credits_db_service": runtime.billing.credits,
                "gen_random_fn": gen_random,
                "build_insufficient_credits_message_fn": (
                    runtime.billing.build_insufficient_message
                ),
                "estimate_ai_base_reserve_credits": (runtime.estimate_ai_base_reserve_credits),
            },
        )
        get_scheduler()
        task_backfill = backfill_canonical_task_records()
        if task_backfill["unmatched"] or task_backfill["invalid"]:
            print(
                "Warning: scheduled-task canonical backfill is incomplete: "
                f"{task_backfill}",
                file=sys.stderr,
            )
        task_rebuild = rebuild_scheduler_from_canonical_records()
        if task_rebuild["missing_next"] or task_rebuild["invalid"]:
            print(
                "Warning: scheduled-task canonical rebuild is incomplete: "
                f"{task_rebuild}",
                file=sys.stderr,
            )
    except Exception as error:
        print(f"Warning: failed to initialize task scheduler: {error}", file=sys.stderr)

    try:
        index.update_telegram_bot_commands()
    except Exception as e:
        print(f"Warning: failed to update bot commands: {e}", file=sys.stderr)

    try:
        run_polling(
            token=token,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query", "pre_checkout_query"],
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        runtime.summary.stop_background_worker()
        runtime.billing_reconciler.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
