from __future__ import annotations

import json
import logging
from pathlib import Path

from api.ai import request_runtime


class _FakeRustAIRequestSanitization:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def ai_sanitize_assistant_text(self, text: str) -> str:
        self.calls.append(text)
        if self.fail:
            raise ValueError("synthetic Rust request-sanitization failure")
        return f"rust:{text}"


def test_rust_ai_request_sanitization_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustAIRequestSanitization()
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_request_sanitization",
        lambda: rust,
    )

    actual = request_runtime.sanitize_bot_message(
        {"role": "assistant", "content": "PYTHON."}
    )

    assert actual == {"role": "assistant", "content": "rust:PYTHON."}
    assert rust.calls == ["PYTHON."]


def test_nested_text_parts_keep_legacy_mutation_and_container_identity(monkeypatch) -> None:
    rust = _FakeRustAIRequestSanitization()
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_request_sanitization",
        lambda: rust,
    )
    parts = [
        {"type": "text", "text": "FIRST."},
        {"type": "image_url", "image_url": "synthetic://image"},
        {"type": "text", "text": "SECOND."},
    ]
    message = {"role": "assistant", "content": parts, "id": "synthetic"}

    actual = request_runtime.sanitize_bot_message(message)

    assert actual is not message
    assert actual["content"] is parts
    assert message["content"] is parts
    assert parts[0]["text"] == "rust:FIRST."
    assert parts[1] == {"type": "image_url", "image_url": "synthetic://image"}
    assert parts[2]["text"] == "rust:SECOND."
    assert rust.calls == ["FIRST.", "SECOND."]


def test_non_assistant_message_preserves_identity_without_bridge(monkeypatch) -> None:
    rust = _FakeRustAIRequestSanitization()
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_request_sanitization",
        lambda: rust,
    )
    message = {"role": "user", "content": "UNCHANGED."}

    assert request_runtime.sanitize_bot_message(message) is message
    assert rust.calls == []


def test_rust_ai_request_sanitization_failure_uses_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAIRequestSanitization(fail=True)
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_request_sanitization",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=request_runtime.__name__):
        actual = request_runtime.sanitize_bot_message(
            {"role": "assistant", "content": "HOLA😀..."}
        )

    assert actual == {"role": "assistant", "content": "hola"}
    assert "using Python fallback" in caplog.text


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_request_sanitization",
        lambda: None,
    )
    path = Path(__file__).parents[1] / "contracts" / "ai_request_sanitization.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["text_cases"]:
        actual = request_runtime._sanitize_assistant_text(case["input"])
        assert actual == case["expected"], case["name"]
