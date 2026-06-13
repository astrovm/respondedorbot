from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from api.bot import ptb as bot_ptb


class _FakeUpdate:
    def __init__(self, payload):
        self._payload = dict(payload)
        self.message = payload.get("message")
        self.callback_query = payload.get("callback_query")
        self.pre_checkout_query = payload.get("pre_checkout_query")
        self.update_id = payload.get("update_id")

    def to_dict(self):
        return dict(self._payload)


class BotPtbTests(unittest.TestCase):
    def test_get_handler_executor_uses_env_worker_count(self):
        created = []

        class DummyExecutor:
            def __init__(self, max_workers, thread_name_prefix):
                created.append((max_workers, thread_name_prefix))

        with (
            patch.dict("os.environ", {"BOT_HANDLER_WORKERS": "24"}),
            patch("api.bot.ptb.concurrent.futures.ThreadPoolExecutor", DummyExecutor),
            patch.object(bot_ptb, "_HANDLER_EXECUTOR", None),
        ):
            executor = bot_ptb._get_handler_executor()

        self.assertIsNotNone(executor)
        self.assertEqual(created, [(24, "ptb-handler")])

    def test_create_application_requires_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TELEGRAM_TOKEN not provided"):
                bot_ptb.create_application()

    def test_create_application_registers_handlers_and_bot_data(self):
        builder = MagicMock()
        app = MagicMock()
        app.bot_data = {}
        telegram_ext = MagicMock()

        builder.token.return_value = builder
        builder.concurrent_updates.return_value = builder
        builder.post_init.return_value = builder
        builder.build.return_value = app
        telegram_ext.ApplicationBuilder = MagicMock(return_value=builder)
        telegram_ext.CallbackQueryHandler = MagicMock()
        telegram_ext.MessageHandler = MagicMock()
        telegram_ext.PreCheckoutQueryHandler = MagicMock()
        telegram_ext.filters = MagicMock()
        telegram_ext.filters.ALL = object()

        with patch("api.bot.ptb.importlib.import_module", return_value=telegram_ext):
            redis_client = MagicMock()
            result = bot_ptb.create_application(
                token="test-token",
                redis_client=redis_client,
            )

        self.assertIs(result, app)
        self.assertIs(app.bot_data["redis"], redis_client)
        self.assertEqual(app.add_handler.call_count, 3)
        app.add_error_handler.assert_called_once()

    def test_update_to_dict_uses_payload(self):
        payload = {
            "update_id": 123,
            "message": {
                "message_id": 1,
                "chat": {"id": 10, "type": "private"},
                "text": "hola",
            },
        }
        update = _FakeUpdate(payload)
        converted = bot_ptb._update_to_dict(update)

        self.assertEqual(converted["update_id"], 123)
        self.assertEqual(converted["message"]["text"], "hola")

    def test_run_polling_passes_args_to_application(self):
        app = MagicMock()
        with patch("api.bot.ptb.create_application", return_value=app):
            bot_ptb.run_polling(
                token="test-token",
                drop_pending_updates=False,
                allowed_updates=["message"],
            )

        app.run_polling.assert_called_once_with(
            drop_pending_updates=False,
            allowed_updates=["message"],
        )


class BotPtbAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_handle_message_delegates_sync_handler(self):
        update = _FakeUpdate({"message": {"chat": {"id": 1}, "text": "test"}})

        async def _run_sync_impl(func, *args):
            return func(*args)

        runtime = MagicMock()
        with (
            patch("api.bot.ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot.ptb.app_runtime", runtime),
        ):
            await bot_ptb._async_handle_message(update, MagicMock())

        runtime.handle_message.assert_called_once_with(
            {"chat": {"id": 1}, "text": "test"}
        )

    async def test_async_handle_callback_delegates_sync_handler(self):
        update = _FakeUpdate({"callback_query": {"id": "cbq1", "data": "cfg:foo"}})

        async def _run_sync_impl(func, *args):
            return func(*args)

        runtime = MagicMock()
        with (
            patch("api.bot.ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot.ptb.app_runtime", runtime),
        ):
            await bot_ptb._async_handle_callback_query(update, MagicMock())

        runtime.handle_callback_query.assert_called_once_with(
            {"id": "cbq1", "data": "cfg:foo"}
        )

    async def test_async_handle_pre_checkout_delegates_sync_handler(self):
        update = _FakeUpdate(
            {"pre_checkout_query": {"id": "pcq1", "invoice_payload": "topup:p50"}}
        )

        async def _run_sync_impl(func, *args):
            return func(*args)

        runtime = MagicMock()
        with (
            patch("api.bot.ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot.ptb.app_runtime", runtime),
        ):
            await bot_ptb._async_handle_pre_checkout_query(update, MagicMock())

        runtime.billing.handle_pre_checkout.assert_called_once_with(
            {"id": "pcq1", "invoice_payload": "topup:p50"}
        )

    async def test_polling_network_error_is_warning_only(self):
        class NetworkError(Exception):
            pass

        context = MagicMock()
        context.error = NetworkError("httpx.ConnectError: All connection attempts failed")

        with (
            patch("api.bot.ptb._run_sync") as run_sync,
            patch("api.bot.ptb.time.monotonic", return_value=1000.0),
            patch("api.bot.ptb._last_polling_network_report", 0.0),
            self.assertLogs("api.bot.ptb", level="WARNING") as logs,
        ):
            await bot_ptb._error_handler(object(), context)

        run_sync.assert_not_called()
        self.assertIn("PTB polling network error; polling will retry", logs.output[0])

    async def test_polling_network_error_warning_is_rate_limited(self):
        class NetworkError(Exception):
            pass

        context = MagicMock()
        context.error = NetworkError("httpx.RemoteProtocolError: disconnected")

        with (
            patch("api.bot.ptb._run_sync") as run_sync,
            patch("api.bot.ptb.time.monotonic", return_value=1001.0),
            patch("api.bot.ptb._last_polling_network_report", 900.0),
        ):
            await bot_ptb._error_handler(object(), context)

        run_sync.assert_not_called()


class PollingEntrypointTests(unittest.TestCase):
    def test_main_uses_hardcoded_defaults(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "TELEGRAM_TOKEN": "abc",
                },
                clear=True,
            ),
            patch("run_polling._load_dotenv"),
            patch("api.bot.ptb.run_polling") as mock_run_polling,
        ):
            import importlib
            import run_polling

            importlib.reload(run_polling)
            self.assertEqual(run_polling.main(), 0)

        mock_run_polling.assert_called_once_with(
            token="abc",
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query", "pre_checkout_query"],
        )

    def test_main_requires_token(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("run_polling._load_dotenv"),
        ):
            from run_polling import main

            self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
