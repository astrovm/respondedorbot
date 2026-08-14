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
    from api.markets.world_cup_goals import (
        WorldCupGoalMonitor,
        start_world_cup_goal_monitor,
    )
    from api.tasks.scheduler import get_scheduler, init_scheduler

    runtime = index.app_runtime
    threading.Thread(target=_price_refresh_loop, daemon=True).start()
    runtime.summary.start_background_worker()
    start_world_cup_goal_monitor(
        WorldCupGoalMonitor(
            list_chat_ids=index.list_world_cup_goal_chat_ids,
            ask_ai=runtime.ai.ask,
            send_message=runtime.telegram.send_message,
            credits_db_service=runtime.billing.credits,
            estimate_ai_base_reserve_credits=(
                runtime.estimate_ai_base_reserve_credits
            ),
            scoreboard=index._world_cup_scoreboard,
        )
    )

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
                "estimate_ai_base_reserve_credits": (
                    runtime.estimate_ai_base_reserve_credits
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
        index.app_runtime.summary.stop_background_worker()
        return 0
    except Exception as error:
        index.app_runtime.summary.stop_background_worker()
        print(f"FATAL: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
