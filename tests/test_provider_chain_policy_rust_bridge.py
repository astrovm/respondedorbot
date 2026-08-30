from __future__ import annotations

import json
import logging
from pathlib import Path

from api.ai.pricing import AIUsageResult
from api.providers import base


class _FakeRustProviderChainPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[object, ...]] = []

    def _result(self, name: str, value: object, *arguments: object) -> object:
        self.calls.append((name, *arguments))
        if self.fail:
            raise ValueError("synthetic Rust provider-chain failure")
        return value

    def provider_chain_select(self, *arguments: object) -> list[int]:
        value = self._result("selection", [1], *arguments)
        assert isinstance(value, list)
        return value

    def provider_chain_outcome(self, *arguments: object) -> tuple[str, bool]:
        value = self._result("outcome", ("rust-selected", True), *arguments)
        assert isinstance(value, tuple)
        return value


class _Provider:
    def __init__(self, name: str, text: str, *, available: bool = True) -> None:
        self._name = name
        self._text = text
        self._available = available
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def complete(self, *_arguments, **_kwargs) -> AIUsageResult:
        self.calls += 1
        return AIUsageResult(
            kind="chat",
            text=self._text,
            model="synthetic-model",
            usage={},
            metadata={},
        )


def test_rust_provider_chain_policy_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderChainPolicy()
    monkeypatch.setattr(base, "_load_rust_provider_chain_policy", lambda: rust)
    first = _Provider("python-first", "first")
    second = _Provider("python-second", "second")
    chain = base.ProviderChain([first, second])

    result = chain.complete({"role": "system", "content": "synthetic"}, [])

    assert result.result is not None
    assert result.result.text == "second"
    assert result.provider_name == "rust-selected"
    assert result.fallback_used is True
    assert first.calls == 0
    assert second.calls == 1
    assert rust.calls == [
        ("selection", [True, True]),
        ("outcome", ["python-second"], 0),
    ]


def test_rust_provider_chain_failure_preserves_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustProviderChainPolicy(fail=True)
    monkeypatch.setattr(base, "_load_rust_provider_chain_policy", lambda: rust)
    first = _Provider("python-first", "first")
    second = _Provider("python-second", "second")
    chain = base.ProviderChain([first, second])

    with caplog.at_level(logging.ERROR, logger=base.__name__):
        result = chain.complete({"role": "system", "content": "synthetic"}, [])

    assert result.result is not None
    assert result.result.text == "first"
    assert result.provider_name == "python-first"
    assert result.fallback_used is False
    assert [call[0] for call in rust.calls] == ["selection", "outcome"]
    assert caplog.text.count("using Python fallback") == 2


def test_invalid_rust_provider_indices_use_python_fallback(monkeypatch, caplog) -> None:
    rust = _FakeRustProviderChainPolicy()
    rust.provider_chain_select = lambda *_arguments: [99]  # type: ignore[method-assign]
    monkeypatch.setattr(base, "_load_rust_provider_chain_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=base.__name__):
        selected = base.ProviderChain([_Provider("only", "result")]).available_providers

    assert [provider.name for provider in selected] == ["only"]
    assert "using Python fallback" in caplog.text


def test_python_fallback_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(base, "_load_rust_provider_chain_policy", lambda: None)
    path = Path(__file__).parents[1] / "contracts" / "provider_chain_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["selection_cases"]:
        assert base._select_available_indices(case["availability"]) == case["expected"], case[
            "name"
        ]
    for case in contract["outcome_cases"]:
        actual = base._provider_chain_outcome(
            case["provider_names"],
            case["successful_position"],
        )
        assert list(actual) == case["expected"], case["name"]
