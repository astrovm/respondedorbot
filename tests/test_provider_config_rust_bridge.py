from __future__ import annotations

import json
import logging
from pathlib import Path

from api.providers import config


class _FakeRustProviderConfigPolicy:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def _result(self, value):
        if self.fail:
            raise ValueError("synthetic Rust provider-config failure")
        return value

    def provider_groq_api_key(self, *_arguments):
        return self._result("rust-groq-key")

    def provider_configured_groq_accounts(self, *_arguments):
        return self._result(["paid"])

    def provider_openrouter_api_key(self, _value):
        return self._result("rust-openrouter-key")

    def provider_openrouter_base_url(self):
        return self._result("https://rust.invalid/v1")

    def provider_groq_backoff_key(self, _account, _scope):
        return self._result("rust:backoff")

    def provider_scope_is_available(self, _backoff_active):
        return self._result(False)

    def provider_web_search_tool(self, _max_results, _max_queries):
        return self._result(
            json.dumps(
                {
                    "type": "rust:web_search",
                    "parameters": {"engine": "synthetic"},
                }
            )
        )


def test_rust_provider_configuration_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustProviderConfigPolicy()
    monkeypatch.setattr(config, "_load_rust_provider_config_policy", lambda: rust)

    assert config.get_groq_api_key("free", environment={}) == "rust-groq-key"
    assert config.get_configured_groq_accounts(
        ("free", "paid"),
        get_api_key=lambda _account: "python-key",
    ) == ["paid"]
    assert config.get_openrouter_api_key(environment={}) == "rust-openrouter-key"
    assert config.get_openrouter_base_url() == "https://rust.invalid/v1"
    assert config.get_groq_backoff_key("free", "chat") == "rust:backoff"
    assert config.is_scope_available([]) is False
    assert config.build_web_search_tool(5, 3) == {
        "type": "rust:web_search",
        "parameters": {"engine": "synthetic"},
    }


def test_rust_provider_configuration_failure_preserves_python_behavior(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderConfigPolicy(fail=True)
    monkeypatch.setattr(config, "_load_rust_provider_config_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=config.__name__):
        assert config.get_groq_api_key(
            "free", environment={"GROQ_FREE_API_KEY": " free-key "}
        ) == "free-key"
        assert config.get_configured_groq_accounts(
            ("free", "paid"),
            get_api_key=lambda account: "key" if account == "free" else None,
        ) == ["free"]
        assert config.get_openrouter_api_key(
            environment={"OPENROUTER_API_KEY": " router-key "}
        ) == "router-key"
        assert config.get_openrouter_base_url() == config.DEFAULT_OPENROUTER_URL
        assert config.get_groq_backoff_key("FREE", "CHAT") == "groq:free:chat"
        assert config.is_scope_available([True, False]) is True
        assert config.build_web_search_tool(5, 3)["parameters"][
            "max_total_results"
        ] == 15

    assert caplog.text.count("using Python fallback") == 7


def test_invalid_rust_provider_configuration_uses_python_fallback(
    monkeypatch,
    caplog,
) -> None:
    rust = _FakeRustProviderConfigPolicy()
    rust.provider_configured_groq_accounts = lambda *_arguments: ["unknown"]
    rust.provider_openrouter_base_url = lambda: ""
    rust.provider_groq_backoff_key = lambda *_arguments: ""
    rust.provider_web_search_tool = lambda *_arguments: "[]"
    monkeypatch.setattr(config, "_load_rust_provider_config_policy", lambda: rust)

    with caplog.at_level(logging.ERROR, logger=config.__name__):
        assert config.get_configured_groq_accounts(
            ("free",), get_api_key=lambda _account: "key"
        ) == ["free"]
        assert config.get_openrouter_base_url() == config.DEFAULT_OPENROUTER_URL
        assert config.get_groq_backoff_key("free", "chat") == "groq:free:chat"
        assert config.build_web_search_tool(2, 4)["parameters"][
            "max_total_results"
        ] == 8

    assert caplog.text.count("using Python fallback") == 4


def test_python_provider_configuration_matches_shared_contract(monkeypatch) -> None:
    monkeypatch.setattr(config, "_load_rust_provider_config_policy", lambda: None)
    path = Path(__file__).parents[1] / "contracts" / "provider_config_policy.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["credential_cases"]:
        environment = {
            key: value
            for key, value in {
                "GROQ_FREE_API_KEY": case["free_api_key"],
                "GROQ_API_KEY": case["paid_api_key"],
            }.items()
            if value is not None
        }
        assert (
            config.get_groq_api_key(case["account"], environment=environment)
            == case["expected"]
        ), case["name"]
    for case in contract["openrouter_key_cases"]:
        environment = (
            {"OPENROUTER_API_KEY": case["value"]}
            if case["value"] is not None
            else {}
        )
        assert (
            config.get_openrouter_api_key(environment=environment) == case["expected"]
        ), case["name"]
    for case in contract["account_cases"]:
        configured = dict(zip(case["account_order"], case["configured"], strict=True))
        assert config.get_configured_groq_accounts(
            tuple(case["account_order"]),
            get_api_key=lambda account: "key" if configured[account] else None,
        ) == case["expected"], case["name"]
    for case in contract["backoff_cases"]:
        assert (
            config.get_groq_backoff_key(case["account"], case["scope"])
            == case["expected"]
        ), case["name"]
    for case in contract["scope_cases"]:
        assert (
            config.is_scope_available(case["backoff_active"]) is case["expected"]
        ), case["name"]
    for case in contract["web_search_cases"]:
        assert config.build_web_search_tool(
            case["max_results"], case["max_queries"]
        ) == case["expected"], case["name"]
    assert config.get_openrouter_base_url() == contract["openrouter_base_url"]
