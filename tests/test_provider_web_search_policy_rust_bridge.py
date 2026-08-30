from __future__ import annotations

import logging
from types import SimpleNamespace

from api.providers import runtime as provider_runtime


class _FakeRustProviderWebSearchPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object):
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust web-search policy failure")
        return value

    def provider_web_search_max_uses(self, value_json: str | None) -> int:
        return int(self._result("limit", 91, value_json))

    def provider_web_search_round_metrics(self, *arguments: object):
        return self._result("round", (8, 9, True, 7), *arguments)

    def provider_web_search_remaining_budget(self, *arguments: object):
        return self._result("budget", 55, *arguments)

    def provider_web_search_source_urls(self, messages_json: str) -> list[str]:
        return list(self._result("sources", ["https://rust.example"], messages_json))

    def provider_web_search_outcome_is_grounded(self, *arguments: object) -> bool:
        return bool(self._result("outcome", False, *arguments))


def _runtime() -> provider_runtime.ProviderRuntime:
    runtime = object.__new__(provider_runtime.ProviderRuntime)
    runtime._deps = SimpleNamespace(
        extract_usage_map=lambda _response: {
            "server_tool_use": {"web_search_requests": 2}
        },
        primary_model="test-model",
    )
    return runtime


def _message():
    return SimpleNamespace(
        content="answer",
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name="web_search"))],
        annotations=[{"type": "url_citation"}],
    )


def test_rust_web_search_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderWebSearchPolicy()
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_web_search_policy",
        lambda: rust,
    )
    runtime = _runtime()

    assert runtime._web_search_max_uses({"parameters": {"max_uses": 3}}) == 91
    assert runtime._web_search_metadata(object(), _message()) == {
        "web_search_requests": 8,
        "web_search_citation_count": 9,
        "web_search_grounded": True,
    }
    assert runtime._web_search_request_count(object(), _message()) == 7
    assert runtime._remaining_web_search_uses(3, object(), _message()) == 55
    assert runtime._web_search_source_urls([]) == ["https://rust.example"]

    metadata = {"web_search_citation_count": 1}
    runtime._record_web_search_outcome(
        "answer",
        [],
        metadata,
        tool_context={},
        round_idx=0,
    )
    assert metadata["web_search_grounded"] is False
    assert [call[0] for call in rust.calls] == [
        "limit",
        "round",
        "round",
        "round",
        "budget",
        "sources",
        "sources",
        "outcome",
    ]


def test_rust_web_search_policy_failure_preserves_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderWebSearchPolicy(fail=True)
    monkeypatch.setattr(
        provider_runtime,
        "_load_rust_provider_web_search_policy",
        lambda: rust,
    )
    runtime = _runtime()
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "search", "function": {"name": "web_search"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "search",
            "content": '{"results":[{"url":"https://python.example"}]}',
        },
    ]

    with caplog.at_level(logging.ERROR, logger=provider_runtime.__name__):
        assert runtime._web_search_max_uses({"parameters": {"max_uses": 3}}) == 3
        assert runtime._web_search_request_count(object(), _message()) == 2
        assert runtime._remaining_web_search_uses(3, object(), _message()) == 1
        assert runtime._web_search_source_urls(messages) == ["https://python.example"]
        metadata = {"web_search_citation_count": 0}
        runtime._record_web_search_outcome(
            "answer",
            messages,
            metadata,
            tool_context={},
            round_idx=0,
        )

    assert metadata["web_search_grounded"] is True
    assert "using Python fallback" in caplog.text
