from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from api import index


def test_admin_report_formats_error_traceback_from_exception():
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        captured_error = error

    with (
        patch.dict(
            "api.admin_reporting.environ",
            {"ADMIN_CHAT_ID": "1", "FRIENDLY_INSTANCE_NAME": "test"},
            clear=True,
        ),
        patch.object(index.app_runtime.telegram, "send_message") as send_msg,
    ):
        index.app_runtime.admin.report("failed", captured_error)

    message = send_msg.call_args.args[1]
    assert "tipo de error: RuntimeError" in message
    assert "mensaje de error: boom" in message
    assert "RuntimeError: boom" in message
    assert "NoneType: None" not in message


def test_admin_report_redacts_telegram_token_from_error_message_and_traceback():
    fake_bot_auth = "123456" + ":" + "ABC-secret"
    try:
        raise RuntimeError(
            f"https://api.telegram.org/bot{fake_bot_auth}/sendMessage failed"
        )
    except RuntimeError as error:
        captured_error = error

    with (
        patch.dict(
            "api.admin_reporting.environ",
            {"ADMIN_CHAT_ID": "1", "FRIENDLY_INSTANCE_NAME": "test"},
            clear=True,
        ),
        patch.object(index.app_runtime.telegram, "send_message") as send_msg,
    ):
        index.app_runtime.admin.report("failed", captured_error)

    message = send_msg.call_args.args[1]
    assert fake_bot_auth not in message
    assert "/bot<redacted>/sendMessage" in message


def test_telegram_request_redacts_token_in_error_log(capsys):
    fake_bot_auth = "123456" + ":" + "ABC-secret"
    error = requests.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): "
        f"Max retries exceeded with url: /bot{fake_bot_auth}/sendMessage"
    )

    with (
        patch.dict("api.index.environ", {"TELEGRAM_TOKEN": fake_bot_auth}, clear=True),
        patch("api.index.http_client.get", side_effect=error),
    ):
        payload, description = index._telegram_request("sendMessage")

    output = capsys.readouterr().out
    assert payload is None
    assert description == str(error)
    assert fake_bot_auth not in output
    assert "/bot<redacted>/sendMessage" in output


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/bot123456:ABC/sendMessage", "/bot<redacted>/sendMessage"),
        ("no token here", "no token here"),
    ],
)
def test_redact_telegram_tokens(raw, expected):
    assert index._redact_telegram_tokens(raw) == expected
