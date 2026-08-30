from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from api.bot import telegram


class _FakeRustTelegramHttp:
    def __init__(self, outcome: object, *, fail: bool = False) -> None:
        self.outcome = outcome
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []
        self.multipart_calls: list[tuple[object, ...]] = []

    def telegram_http_request(self, *arguments: object) -> str:
        self.calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust Telegram failure")
        return json.dumps(self.outcome)

    def telegram_multipart_request(self, *arguments: object) -> str:
        self.multipart_calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust Telegram multipart failure")
        return json.dumps(self.outcome)


def _python_response(payload: object) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


@pytest.mark.parametrize(
    ("method", "params", "json_payload"),
    [
        ("GET", {"chat_id": "42", "action": "typing"}, None),
        ("POST", None, {"chat_id": "42", "text": "hola"}),
    ],
)
def test_rust_non_multipart_request_is_authoritative(
    monkeypatch,
    method: str,
    params: dict[str, object] | None,
    json_payload: dict[str, object] | None,
) -> None:
    rust = _FakeRustTelegramHttp(
        {
            "status": "response",
            "status_code": 200,
            "body": '{"ok":true,"result":{"message_id":42}}',
        }
    )
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    python_request = MagicMock(
        side_effect=AssertionError("Python HTTP path must not run")
    )
    monkeypatch.setattr(telegram, "_send_telegram_request", python_request)

    actual = telegram.telegram_request(
        "sendMessage",
        method=method,
        params=params,
        json_payload=json_payload,
        token="synthetic-token",
    )

    assert actual == ({"ok": True, "result": {"message_id": 42}}, None)
    assert rust.calls == [
        (
            "synthetic-token",
            "sendMessage",
            method,
            (
                json.dumps(params, ensure_ascii=False, separators=(",", ":"))
                if params is not None
                else None
            ),
            (
                json.dumps(
                    json_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if json_payload is not None
                else None
            ),
            5,
        )
    ]
    python_request.assert_not_called()


def test_rust_transport_error_does_not_repeat_side_effect(monkeypatch) -> None:
    rust = _FakeRustTelegramHttp(
        {"status": "transport_error", "kind": "timeout"}
    )
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    python_request = MagicMock(
        side_effect=AssertionError("failed side effect must not be repeated")
    )
    monkeypatch.setattr(telegram, "_send_telegram_request", python_request)

    actual = telegram.telegram_request(
        "sendMessage",
        method="POST",
        json_payload={"chat_id": "42", "text": "hola"},
        token="synthetic-token",
        log_errors=False,
    )

    assert actual == (None, "Telegram request timed out")
    python_request.assert_not_called()


def test_invalid_rust_outcome_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustTelegramHttp({"status": "unknown"})
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    response = _python_response({"ok": True, "result": True})
    monkeypatch.setattr(telegram, "_send_telegram_request", MagicMock(return_value=response))

    with caplog.at_level(logging.ERROR, logger=telegram.__name__):
        actual = telegram.telegram_request("getMe", token="synthetic-token")

    assert actual == ({"ok": True, "result": True}, None)
    assert "using Python fallback" in caplog.text


def test_bridge_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustTelegramHttp({}, fail=True)
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    response = _python_response({"ok": True, "result": True})
    send = MagicMock(return_value=response)
    monkeypatch.setattr(telegram, "_send_telegram_request", send)

    with caplog.at_level(logging.ERROR, logger=telegram.__name__):
        actual = telegram.telegram_request("getMe", token="synthetic-token")

    assert actual == ({"ok": True, "result": True}, None)
    assert "using Python fallback" in caplog.text
    send.assert_called_once()


def test_rust_multipart_request_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustTelegramHttp(
        {
            "status": "response",
            "status_code": 200,
            "body": '{"ok":true,"result":{"message_id":7}}',
        }
    )
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    send = MagicMock(side_effect=AssertionError("Python upload must not run"))
    monkeypatch.setattr(telegram, "_send_telegram_request", send)

    actual = telegram.telegram_request(
        "sendPhoto",
        method="POST",
        data_payload={"chat_id": "42"},
        files={"photo": ("chart.png", b"synthetic", "image/png")},
        token="synthetic-token",
    )

    assert actual == ({"ok": True, "result": {"message_id": 7}}, None)
    assert rust.multipart_calls == [
        (
            "synthetic-token",
            "sendPhoto",
            '{"chat_id":"42"}',
            "photo",
            "chart.png",
            b"synthetic",
            "image/png",
            5,
        )
    ]
    send.assert_not_called()


def test_multipart_transport_error_does_not_repeat_upload(monkeypatch) -> None:
    rust = _FakeRustTelegramHttp(
        {"status": "transport_error", "kind": "connection"}
    )
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", lambda: rust)
    send = MagicMock(side_effect=AssertionError("failed upload must not be repeated"))
    monkeypatch.setattr(telegram, "_send_telegram_request", send)

    actual = telegram.telegram_request(
        "sendVideo",
        method="POST",
        data_payload={"chat_id": "42"},
        files={"video": ("video.mp4", b"synthetic", "video/mp4")},
        token="synthetic-token",
        log_errors=False,
    )

    assert actual == (None, "Telegram connection failed")
    send.assert_not_called()


@pytest.mark.parametrize(
    ("method", "params", "json_payload", "files"),
    [
        ("POST", None, None, {"document": ("file.bin", b"synthetic")}),
        (
            "GET",
            None,
            None,
            {"photo": ("chart.png", b"synthetic", "image/png")},
        ),
        (
            "POST",
            {"chat_id": "42"},
            None,
            {"photo": ("chart.png", b"synthetic", "image/png")},
        ),
        (
            "POST",
            None,
            {"chat_id": "42"},
            {"photo": ("chart.png", b"synthetic", "image/png")},
        ),
        (
            "POST",
            None,
            None,
            {"document": ("file.bin", b"synthetic", "application/octet-stream")},
        ),
    ],
)
def test_unsupported_multipart_shape_remains_on_python(
    monkeypatch,
    method: str,
    params: dict[str, object] | None,
    json_payload: dict[str, object] | None,
    files: dict[str, tuple[object, ...]],
) -> None:
    loader = MagicMock()
    monkeypatch.setattr(telegram, "_load_rust_telegram_http_adapter", loader)
    response = _python_response({"ok": True, "result": True})
    send = MagicMock(return_value=response)
    monkeypatch.setattr(telegram, "_send_telegram_request", send)

    actual = telegram.telegram_request(
        "customUpload",
        method=method,
        params=params,
        json_payload=json_payload,
        data_payload={"chat_id": "42"},
        files=files,
        token="synthetic-token",
    )

    assert actual == ({"ok": True, "result": True}, None)
    loader.assert_not_called()
    send.assert_called_once()


def test_rust_response_translation_matches_shared_contract(capsys) -> None:
    path = Path(__file__).parents[1] / "contracts" / "telegram_http_adapter.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        actual = telegram._rust_telegram_result(
            json.dumps(
                {
                    "status": "response",
                    "status_code": case["status_code"],
                    "body": case["body"],
                }
            ),
            "syntheticEndpoint",
            log_errors=False,
            expect_json=case["expect_json"],
        )
        assert list(actual) == [
            case["expected_payload"],
            case["expected_error"],
        ], case["name"]

    for case in contract["transport_errors"]:
        actual = telegram._rust_telegram_result(
            json.dumps(
                {"status": "transport_error", "kind": case["kind"]}
            ),
            "syntheticEndpoint",
            log_errors=False,
            expect_json=True,
        )
        assert actual == (None, case["expected_error"])
    assert capsys.readouterr().out == ""


def test_rust_error_logs_redact_embedded_bot_tokens(capsys) -> None:
    telegram._rust_telegram_result(
        json.dumps(
            {
                "status": "response",
                "status_code": 500,
                "body": "failure at https://api.telegram.org/botsecret-token/getMe",
            }
        ),
        "getMe",
        log_errors=True,
        expect_json=True,
    )

    output = capsys.readouterr().out
    assert "secret-token" not in output
    assert "/bot<redacted>/" in output
