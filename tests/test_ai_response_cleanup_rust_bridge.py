from __future__ import annotations

import logging

from api.ai import pipeline


class _FakeRustAIResponseCleanup:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def ai_cleanup_response(self, *arguments: object):
        self.calls.append(arguments)
        if self.fail:
            raise ValueError("synthetic Rust cleanup failure")
        return ("rust raw", "rust persona", "rust context", "rust identity", "rust final")


def _request() -> pipeline.AIResponseRequest:
    return pipeline.AIResponseRequest(
        chat_id="synthetic-chat",
        handler=lambda _messages: None,
        messages=[],
        context_texts=["context", None],
        user_identity="@user",
    )


def test_rust_ai_response_cleanup_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustAIResponseCleanup()
    monkeypatch.setattr(
        pipeline,
        "_load_rust_ai_response_cleanup",
        lambda: rust,
    )

    actual = pipeline._clean_response(_request(), "Python would keep this")

    assert actual == pipeline._CleanupStages(
        raw="rust raw",
        persona="rust persona",
        context="rust context",
        identity="rust identity",
        final="rust final",
    )
    assert rust.calls == [("Python would keep this", '["context", null]', "@user")]


def test_rust_ai_response_cleanup_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAIResponseCleanup(fail=True)
    monkeypatch.setattr(
        pipeline,
        "_load_rust_ai_response_cleanup",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=pipeline.__name__):
        actual = pipeline._clean_response(
            _request(),
            "Gordo: context: @user: **answer**",
        )

    assert actual.final == "answer"
    assert "using Python fallback" in caplog.text
