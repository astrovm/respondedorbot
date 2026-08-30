from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from api.ai import request_runtime


class _FakeRustAIImageContextPlanning:
    def __init__(
        self,
        result: tuple[str, str | None] = ("description_failed", None),
        *,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def ai_plan_image_context(self, *facts):
        self.calls.append(facts)
        if self.fail:
            raise ValueError("synthetic Rust image-context failure")
        return self.result


def test_rust_image_context_plan_is_authoritative_for_billing_and_prompt(
    monkeypatch,
) -> None:
    rust = _FakeRustAIImageContextPlanning()
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_image_context_planning",
        lambda: rust,
    )
    messages = [{"role": "user", "content": "synthetic prompt"}]
    segments: list[object] = []
    logs: list[str] = []

    request_runtime.inject_image_context(
        messages,
        b"synthetic-image",
        "synthetic-file",
        {},
        describe_image=lambda *_args: SimpleNamespace(text="synthetic description"),
        append_billing_segment=lambda _meta, segment: segments.append(segment),
        logger=SimpleNamespace(info=logs.append),
    )

    assert messages[-1]["content"] == "synthetic prompt"
    assert segments == []
    assert rust.calls[0][:3] == (
        True,
        "synthetic description",
        "synthetic prompt",
    )


def test_rust_image_context_failure_preserves_python_behavior(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAIImageContextPlanning(fail=True)
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_image_context_planning",
        lambda: rust,
    )
    messages = [{"role": "user", "content": "synthetic prompt"}]
    segments: list[object] = []

    with caplog.at_level(logging.ERROR, logger=request_runtime.__name__):
        request_runtime.inject_image_context(
            messages,
            b"synthetic-image",
            "synthetic-file",
            {},
            describe_image=lambda *_args: SimpleNamespace(
                text="synthetic description"
            ),
            append_billing_segment=lambda _meta, segment: segments.append(segment),
            logger=SimpleNamespace(info=lambda _message: None),
        )

    assert len(segments) == 1
    assert messages[-1]["content"].startswith("synthetic prompt\n\n")
    assert "using Python fallback" in caplog.text


def test_invalid_rust_image_context_plan_uses_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustAIImageContextPlanning(("description_ready", "orphan"))
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_image_context_planning",
        lambda: rust,
    )

    with caplog.at_level(logging.ERROR, logger=request_runtime.__name__):
        actual = request_runtime._plan_image_context(
            True,
            "description",
            None,
            "context",
        )

    assert actual == ("description_ready", None)
    assert "using Python fallback" in caplog.text


def test_python_image_context_planning_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        request_runtime,
        "_load_rust_ai_image_context_planning",
        lambda: None,
    )
    path = (
        Path(__file__).parents[1]
        / "contracts"
        / "ai_image_context_planning.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        actual = request_runtime._plan_image_context(*case["facts"])
        assert list(actual) == case["expected"], case["name"]
