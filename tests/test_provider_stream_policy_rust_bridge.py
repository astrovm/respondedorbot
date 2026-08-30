from __future__ import annotations

import logging
import json
from types import SimpleNamespace

import pytest

from api.providers import openrouter


class _FakeRustProviderStreamPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object):
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust stream-policy failure")
        return value

    def provider_stream_text_decision(self, *arguments: object):
        return self._result(
            "decision",
            ("held-by-rust", "emitted-by-rust", True),
            *arguments,
        )

    def provider_stream_could_be_pseudo_tool_call(self, *arguments: object) -> bool:
        return bool(self._result("candidate", True, *arguments))

    def provider_stream_accumulate_tool_calls(self, *arguments: object) -> str:
        return str(
            self._result(
                "tools",
                json.dumps(
                    [
                        {
                            "index": 4,
                            "id": "rust-id",
                            "type": "function",
                            "name": "rust_tool",
                            "arguments": "{\"rust\":true}",
                        }
                    ]
                ),
                *arguments,
            )
        )


def _client_with_content(content: str):
    chunk = SimpleNamespace(
        error=None,
        model=None,
        provider=None,
        service_tier=None,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content=content,
                    annotations=[],
                    tool_calls=[],
                ),
            )
        ],
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: iter([chunk]))
        )
    )


def test_rust_stream_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderStreamPolicy()
    monkeypatch.setattr(
        openrouter,
        "_load_rust_provider_stream_policy",
        lambda: rust,
    )
    provider = object.__new__(openrouter.OpenRouterProvider)

    stream = provider._consume_stream_round(
        _client_with_content("python would release this"),
        {},
        {"web_fetch"},
    )
    assert next(stream) == "emitted-by-rust"
    with pytest.raises(StopIteration) as stopped:
        next(stream)
    streamed_round, held_text, text_released = stopped.value.value

    assert streamed_round.text == "python would release this"
    assert held_text == "held-by-rust"
    assert text_released is True
    assert provider._could_be_pseudo_tool_call("plain", {"web_fetch"}) is True
    assert [call[0] for call in rust.calls] == ["decision", "candidate"]


def test_rust_stream_policy_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderStreamPolicy(fail=True)
    monkeypatch.setattr(
        openrouter,
        "_load_rust_provider_stream_policy",
        lambda: rust,
    )
    provider = object.__new__(openrouter.OpenRouterProvider)

    with caplog.at_level(logging.ERROR, logger=openrouter.__name__):
        stream = provider._consume_stream_round(
            _client_with_content("plain answer"),
            {},
            {"web_fetch"},
        )
        assert next(stream) == "plain answer"
        with pytest.raises(StopIteration):
            next(stream)
        assert provider._could_be_pseudo_tool_call("plain answer", {"web_fetch"}) is False

    assert "using Python fallback" in caplog.text


def test_rust_stream_tool_accumulation_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderStreamPolicy()
    monkeypatch.setattr(
        openrouter,
        "_load_rust_provider_stream_policy",
        lambda: rust,
    )
    streamed_round = openrouter._StreamRound()
    delta = SimpleNamespace(
        content="",
        annotations=[],
        tool_calls=[
            SimpleNamespace(
                index=0,
                id="python-id",
                type="function",
                function=SimpleNamespace(name="python_tool", arguments="{}"),
            )
        ],
    )

    openrouter.OpenRouterProvider._accumulate_stream_delta(streamed_round, delta)

    assert list(streamed_round.tool_calls) == [4]
    assert streamed_round.tool_calls[4].id == "rust-id"
    assert streamed_round.tool_calls[4].name == "rust_tool"
    assert rust.calls[0][0] == "tools"


def test_rust_stream_tool_accumulation_failure_uses_python(monkeypatch, caplog) -> None:
    rust = _FakeRustProviderStreamPolicy(fail=True)
    monkeypatch.setattr(
        openrouter,
        "_load_rust_provider_stream_policy",
        lambda: rust,
    )
    streamed_round = openrouter._StreamRound()
    delta = SimpleNamespace(
        content="",
        annotations=[],
        tool_calls=[
            SimpleNamespace(
                index="invalid",
                id="call_",
                type="function",
                function=SimpleNamespace(name="web_", arguments="{"),
            ),
            SimpleNamespace(
                index=0,
                id="suffix",
                type=None,
                function=SimpleNamespace(name="fetch", arguments="}"),
            ),
        ],
    )

    with caplog.at_level(logging.ERROR, logger=openrouter.__name__):
        openrouter.OpenRouterProvider._accumulate_stream_delta(streamed_round, delta)

    assert streamed_round.tool_calls[0].id == "call_suffix"
    assert streamed_round.tool_calls[0].name == "web_fetch"
    assert streamed_round.tool_calls[0].arguments == "{}"
    assert "using Python fallback" in caplog.text
