from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from api.tools import web_search


class _FakeRustFirecrawl:
    def __init__(self, outcome: dict[str, Any] | None = None, *, fail: bool = False):
        self.outcome = outcome or {"status": "timeout"}
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def firecrawl_search(self, api_key: str, query: str) -> str:
        self.calls.append((api_key, query))
        if self.fail:
            raise RuntimeError("synthetic Rust Firecrawl failure")
        return json.dumps(self.outcome, ensure_ascii=False)


def _response(status_code: int, body: str) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.ok = status_code < 400
    response.text = body
    try:
        payload = json.loads(body)
    except ValueError:
        response.json.side_effect = ValueError("invalid JSON")
    else:
        response.json.return_value = payload
    return response


def test_rust_success_is_authoritative_and_preserves_accounting(monkeypatch) -> None:
    rust = _FakeRustFirecrawl(
        {
            "status": "success",
            "query": "consulta",
            "results": [
                {
                    "title": "Ejemplo",
                    "url": "https://example.com",
                    "description": "Descripción",
                }
            ],
            "credits_used": 2,
            "request_id": "request-1",
        }
    )
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-key")
    monkeypatch.setattr(web_search, "_load_rust_firecrawl_adapter", lambda: rust)
    post = MagicMock(side_effect=AssertionError("Python HTTP path must not run"))
    monkeypatch.setattr(web_search.requests, "post", post)

    result = web_search._execute_web_search({"query": " consulta "}, {})

    assert json.loads(result.output) == {
        "query": "consulta",
        "results": [
            {
                "title": "Ejemplo",
                "url": "https://example.com",
                "description": "Descripción",
            }
        ],
    }
    assert result.metadata == {
        "query": "consulta",
        "result_count": 1,
        "credits_used": 2,
        "request_id": "request-1",
    }
    assert rust.calls == [("synthetic-key", "consulta")]
    post.assert_not_called()


@pytest.mark.parametrize(
    ("outcome", "expected", "metadata"),
    [
        (
            {"status": "timeout"},
            "Error de búsqueda: Firecrawl agotó el tiempo de espera.",
            {},
        ),
        (
            {"status": "connection"},
            "Error de búsqueda: no se pudo conectar con Firecrawl.",
            {},
        ),
        (
            {"status": "invalid_json"},
            "Error de búsqueda: Firecrawl devolvió JSON inválido.",
            {},
        ),
        (
            {"status": "api_error", "detail": "rechazado"},
            "Error de búsqueda: rechazado",
            {},
        ),
        (
            {"status": "http_error", "status_code": 429, "detail": "límite"},
            "Error de búsqueda: Firecrawl HTTP 429: límite",
            {"status_code": 429},
        ),
    ],
)
def test_rust_typed_errors_keep_localized_python_contract(
    monkeypatch,
    outcome: dict[str, Any],
    expected: str,
    metadata: dict[str, Any],
) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        web_search,
        "_load_rust_firecrawl_adapter",
        lambda: _FakeRustFirecrawl(outcome),
    )

    result = web_search._execute_web_search({"query": "consulta"}, {})

    assert result.output == expected
    assert result.metadata == metadata


@pytest.mark.parametrize(
    "raw_result",
    [
        "[]",
        '{"status":"unknown"}',
        '{"status":"success","query":"other","results":[]}',
        '{"status":"success","query":"consulta","results":[{"title":1,"url":"u","description":"d"}]}',
    ],
)
def test_invalid_rust_outcome_uses_python_fallback(
    monkeypatch,
    caplog,
    raw_result: str,
) -> None:
    rust = MagicMock()
    rust.firecrawl_search.return_value = raw_result
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-key")
    monkeypatch.setattr(web_search, "_load_rust_firecrawl_adapter", lambda: rust)
    monkeypatch.setattr(
        web_search.requests,
        "post",
        MagicMock(
            return_value=_response(
                200,
                '{"success":true,"data":{"web":[]}}',
            )
        ),
    )

    with caplog.at_level(logging.ERROR, logger=web_search.__name__):
        result = web_search._execute_web_search({"query": "consulta"}, {})

    assert json.loads(result.output)["results"] == []
    assert "using Python fallback" in caplog.text


def test_rust_bridge_failure_uses_python_timeout_path(monkeypatch, caplog) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        web_search,
        "_load_rust_firecrawl_adapter",
        lambda: _FakeRustFirecrawl(fail=True),
    )
    monkeypatch.setattr(
        web_search.requests,
        "post",
        MagicMock(side_effect=requests.Timeout("synthetic timeout")),
    )
    monkeypatch.setattr(web_search.time, "sleep", lambda _delay: None)

    with caplog.at_level(logging.ERROR, logger=web_search.__name__):
        result = web_search._execute_web_search({"query": "consulta"}, {})

    assert result.output == "Error de búsqueda: Firecrawl agotó el tiempo de espera."
    assert "using Python fallback" in caplog.text


def test_python_response_parser_matches_shared_contract() -> None:
    path = Path(__file__).parents[1] / "contracts" / "firecrawl_adapter.json"
    contract = json.loads(path.read_text(encoding="utf-8"))

    for case in contract["cases"]:
        expected = case["expected"]
        result = web_search._parse_search_response(
            _response(case["status_code"], case["body"]),
            case["query"],
        )
        if expected["status"] == "success":
            assert json.loads(result.output) == {
                "query": expected["query"],
                "results": expected["results"],
            }, case["name"]
            assert result.metadata == {
                "query": expected["query"],
                "result_count": len(expected["results"]),
                "credits_used": expected["credits_used"],
                "request_id": expected["request_id"],
            }, case["name"]
        elif expected["status"] == "http_error":
            assert result.metadata == {"status_code": expected["status_code"]}
            assert expected["detail"] in result.output
        elif expected["status"] == "invalid_json":
            assert result.output == "Error de búsqueda: Firecrawl devolvió JSON inválido."
        else:
            assert expected["detail"] in result.output
