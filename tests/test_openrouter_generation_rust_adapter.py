from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from api.billing import reconciliation


class _FakeRustGeneration:
    def __init__(self, outcome: object, *, fail: bool = False) -> None:
        self.outcome = outcome
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def openrouter_generation_fetch(self, api_key: str, generation_id: str) -> str:
        self.calls.append((api_key, generation_id))
        if self.fail:
            raise RuntimeError("synthetic Rust generation failure")
        return json.dumps(self.outcome)


def _response(status_code: int, body: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"synthetic HTTP {status_code}"
        )
    try:
        payload = json.loads(body)
    except ValueError:
        response.json.side_effect = ValueError("invalid JSON")
    else:
        response.json.return_value = payload
    return response


def test_rust_generation_success_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustGeneration(
        {
            "status": "success",
            "generation": {
                "id": "generation-1",
                "upstream_inference_cost": 0.0002,
            },
        }
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: rust,
    )
    get = MagicMock(side_effect=AssertionError("Python HTTP path must not run"))
    monkeypatch.setattr(reconciliation.requests, "get", get)

    actual = reconciliation.fetch_openrouter_generation("generation-1")

    assert actual == {
        "id": "generation-1",
        "upstream_inference_cost": 0.0002,
    }
    assert rust.calls == [("synthetic-key", "generation-1")]
    get.assert_not_called()


def test_rust_pending_is_authoritative(monkeypatch) -> None:
    rust = _FakeRustGeneration({"status": "pending"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: rust,
    )
    get = MagicMock(side_effect=AssertionError("Python HTTP path must not run"))
    monkeypatch.setattr(reconciliation.requests, "get", get)

    assert reconciliation.fetch_openrouter_generation("missing") is None
    get.assert_not_called()


@pytest.mark.parametrize(
    "outcome",
    [
        [],
        {"status": "unknown"},
        {"status": "success", "generation": []},
    ],
)
def test_invalid_rust_generation_uses_python_fallback(
    monkeypatch,
    caplog,
    outcome: object,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: _FakeRustGeneration(outcome),
    )
    get = MagicMock(return_value=_response(200, '{"data":{"id":"python"}}'))
    monkeypatch.setattr(reconciliation.requests, "get", get)

    with caplog.at_level(logging.ERROR, logger=reconciliation.__name__):
        actual = reconciliation.fetch_openrouter_generation("generation-1")

    assert actual == {"id": "python"}
    assert "using Python fallback" in caplog.text


def test_rust_transport_failure_uses_python_fallback(monkeypatch, caplog) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: _FakeRustGeneration({}, fail=True),
    )
    get = MagicMock(return_value=_response(404, "not-json"))
    monkeypatch.setattr(reconciliation.requests, "get", get)

    with caplog.at_level(logging.ERROR, logger=reconciliation.__name__):
        actual = reconciliation.fetch_openrouter_generation("generation-1")

    assert actual is None
    assert "using Python fallback" in caplog.text
    get.assert_called_once()


def test_missing_api_key_does_not_load_either_adapter(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loader = MagicMock()
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        loader,
    )

    assert reconciliation.fetch_openrouter_generation("generation-1") is None
    loader.assert_not_called()


def test_python_generation_lookup_matches_shared_contract(monkeypatch) -> None:
    path = (
        Path(__file__).parents[1]
        / "contracts"
        / "openrouter_generation_adapter.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: None,
    )

    for case in contract["cases"]:
        get = MagicMock(return_value=_response(case["status_code"], case["body"]))
        monkeypatch.setattr(reconciliation.requests, "get", get)
        actual = reconciliation.fetch_openrouter_generation("generation-id")
        expected = case["expected"]
        assert actual == (
            expected.get("generation")
            if expected["status"] == "success"
            else None
        ), case["name"]
        get.assert_called_once_with(
            "https://openrouter.ai/api/v1/generation",
            params={"id": "generation-id"},
            headers={"Authorization": "Bearer synthetic-key"},
            timeout=(5.0, 20.0),
        )


def test_python_generation_lookup_preserves_contract_failures(monkeypatch) -> None:
    path = (
        Path(__file__).parents[1]
        / "contracts"
        / "openrouter_generation_adapter.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        reconciliation,
        "_load_rust_openrouter_generation_adapter",
        lambda: None,
    )

    for case in contract["errors"]:
        monkeypatch.setattr(
            reconciliation.requests,
            "get",
            MagicMock(return_value=_response(case["status_code"], case["body"])),
        )
        with pytest.raises((requests.HTTPError, ValueError)):
            reconciliation.fetch_openrouter_generation("generation-id")
