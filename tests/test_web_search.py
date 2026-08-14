import json
from unittest.mock import MagicMock

import requests

from api.tools.registry import execute_tool, get_all_tool_schemas
from api.tools import web_search


def _response(status_code, payload):
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 300
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def test_web_search_returns_compact_citable_results(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    post = MagicMock(
        return_value=_response(
            200,
            {
                "success": True,
                "id": "search-1",
                "creditsUsed": 1,
                "data": {
                    "web": [
                        {
                            "title": "Pablo Wasserman",
                            "url": "https://example.com/pablo",
                            "description": "Perfil público de Pablo Wasserman.",
                            "markdown": "contenido largo que no debe enviarse",
                        }
                    ]
                },
            },
        )
    )
    monkeypatch.setattr(web_search.requests, "post", post)

    result = execute_tool("web_search", {"query": "Pablo Wasserman"})

    payload = json.loads(result.output)
    assert payload["results"] == [
        {
            "title": "Pablo Wasserman",
            "url": "https://example.com/pablo",
            "description": "Perfil público de Pablo Wasserman.",
        }
    ]
    assert "markdown" not in result.output
    assert result.metadata["result_count"] == 1
    assert post.call_args.kwargs["json"] == {
        "query": "Pablo Wasserman",
        "limit": 5,
        "sources": ["web"],
        "timeout": 60_000,
    }
    assert post.call_args.kwargs["timeout"] == (10.0, 75.0)


def test_web_search_retries_transient_http_failures(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    post = MagicMock(
        side_effect=[
            _response(500, {"success": False, "error": "temporary"}),
            _response(200, {"success": True, "data": {"web": []}}),
        ]
    )
    sleep = MagicMock()
    monkeypatch.setattr(web_search.requests, "post", post)
    monkeypatch.setattr(web_search.time, "sleep", sleep)

    result = execute_tool("web_search", {"query": "consulta"})

    assert json.loads(result.output)["results"] == []
    assert post.call_count == 2
    sleep.assert_called_once_with(1)


def test_web_search_reports_timeout_after_retries(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(
        web_search.requests,
        "post",
        MagicMock(side_effect=requests.Timeout("slow")),
    )
    sleep = MagicMock()
    monkeypatch.setattr(web_search.time, "sleep", sleep)

    result = execute_tool("web_search", {"query": "consulta"})

    assert result.output == "Error de búsqueda: Firecrawl agotó el tiempo de espera."
    assert sleep.call_count == 2


def test_web_search_schema_requires_key_and_enabled_context(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    names = {
        tool["function"]["name"]
        for tool in get_all_tool_schemas({"web_search_enabled": True})
    }
    assert "web_search" not in names

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    disabled_names = {
        tool["function"]["name"]
        for tool in get_all_tool_schemas({})
    }
    enabled_names = {
        tool["function"]["name"]
        for tool in get_all_tool_schemas({"web_search_enabled": True})
    }
    assert "web_search" not in disabled_names
    assert "web_search" in enabled_names
