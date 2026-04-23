#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import importlib
import threading
import time
from typing import Optional


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
    from api.index import refresh_price_caches

    time.sleep(300)  # wait 5 min after startup before first run
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

    from api.logging_config import setup_logging
    setup_logging()

    from api import index
    from api.bot_ptb import run_polling
    from api.tools.task_scheduler import get_scheduler, init_scheduler

    threading.Thread(target=_price_refresh_loop, daemon=True).start()

    try:
        init_scheduler(
            redis_factory=index.config_redis,
            task_executor_deps={
                "ask_ai": index.ask_ai,
                "send_msg": index.send_msg,
                "admin_report": index.admin_report,
                "credits_db_service": index.credits_db_service,
                "gen_random_fn": index.gen_random,
                "build_insufficient_credits_message_fn": (
                    index.build_insufficient_credits_message
                ),
                "estimate_ai_base_reserve_credits": (
                    index.estimate_ai_base_reserve_credits
                ),
            },
        )
        get_scheduler()
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
        return 0
    except Exception as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
