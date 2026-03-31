from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from api import bot_ptb


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
    def test_create_application_requires_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TELEGRAM_TOKEN not provided"):
                bot_ptb.create_application()

    def test_create_application_registers_handlers_and_bot_data(self):
        builder = MagicMock()
        app = MagicMock()
        app.bot_data = {}

        builder.token.return_value = builder
        builder.post_init.return_value = builder
        builder.build.return_value = app

        with patch("telegram.ext.ApplicationBuilder", return_value=builder):
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
        with patch("api.bot_ptb.create_application", return_value=app):
            bot_ptb.run_polling(
                token="test-token",
                drop_pending_updates=False,
                allowed_updates=["message"],
            )

        app.run_polling.assert_called_once_with(
            drop_pending_updates=False,
            allowed_updates=["message"],
        )

    def test_run_webhook_requires_webhook_url(self):
        with self.assertRaisesRegex(
            ValueError, "webhook_url is required for webhook mode"
        ):
            bot_ptb.run_webhook(token="test-token", webhook_url=None)

    def test_run_webhook_passes_args_to_application(self):
        app = MagicMock()
        with patch("api.bot_ptb.create_application", return_value=app):
            bot_ptb.run_webhook(
                token="test-token",
                webhook_url="https://example.com/webhook",
                secret_token="secret",
                allowed_updates=["message", "callback_query"],
            )

        app.run_webhook.assert_called_once_with(
            listen="0.0.0.0",
            port=8443,
            webhook_url="https://example.com/webhook",
            secret_token="secret",
            allowed_updates=["message", "callback_query"],
        )


class BotPtbAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_handle_message_delegates_sync_handler(self):
        update = _FakeUpdate({"message": {"chat": {"id": 1}, "text": "test"}})

        async def _run_sync_impl(func, *args):
            return func(*args)

        with (
            patch("api.bot_ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot_ptb.handle_msg") as handle_msg,
        ):
            await bot_ptb._async_handle_message(update, MagicMock())

        handle_msg.assert_called_once_with({"chat": {"id": 1}, "text": "test"})

    async def test_async_handle_callback_delegates_sync_handler(self):
        update = _FakeUpdate({"callback_query": {"id": "cbq1", "data": "cfg:foo"}})

        async def _run_sync_impl(func, *args):
            return func(*args)

        with (
            patch("api.bot_ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot_ptb.handle_callback_query") as handle_callback_query,
        ):
            await bot_ptb._async_handle_callback_query(update, MagicMock())

        handle_callback_query.assert_called_once_with({"id": "cbq1", "data": "cfg:foo"})

    async def test_async_handle_pre_checkout_delegates_sync_handler(self):
        update = _FakeUpdate(
            {"pre_checkout_query": {"id": "pcq1", "invoice_payload": "topup:p50"}}
        )

        async def _run_sync_impl(func, *args):
            return func(*args)

        with (
            patch("api.bot_ptb._run_sync", side_effect=_run_sync_impl),
            patch("api.bot_ptb.handle_pre_checkout_query") as handle_pre_checkout_query,
        ):
            await bot_ptb._async_handle_pre_checkout_query(update, MagicMock())

        handle_pre_checkout_query.assert_called_once_with(
            {"id": "pcq1", "invoice_payload": "topup:p50"}
        )


class PollingEntrypointTests(unittest.TestCase):
    def test_main_uses_environment(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "TELEGRAM_TOKEN": "abc",
                    "PTB_ALLOWED_UPDATES": "message,callback_query",
                    "PTB_DROP_PENDING_UPDATES": "false",
                },
                clear=False,
            ),
            patch("run_polling._load_dotenv"),
            patch("api.bot_ptb.run_polling") as run_polling,
        ):
            from run_polling import main

            self.assertEqual(main(), 0)

        run_polling.assert_called_once_with(
            token="abc",
            drop_pending_updates=False,
            allowed_updates=["message", "callback_query"],
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
