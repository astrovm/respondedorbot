"""Search the public web through Firecrawl."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping

import requests

from api.core.logging import get_logger
from api.tools.registry import ToolResult, register_tool


logger = get_logger(__name__)

_SEARCH_URL = "https://api.firecrawl.dev/v2/search"
_MAX_RESULTS = 5
_MAX_ATTEMPTS = 3
_API_TIMEOUT_MS = 60_000
_REQUEST_TIMEOUT_SECONDS = (10.0, 75.0)
_MAX_DESCRIPTION_CHARS = 1_200


def _clean_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}…"


def _response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return _clean_text(response.text, max_chars=500) or "respuesta sin detalles"
    if isinstance(payload, Mapping):
        return _clean_text(
            payload.get("error") or payload.get("message") or payload,
            max_chars=500,
        )
    return _clean_text(payload, max_chars=500)


def _search_request(query: str, api_key: str) -> requests.Response:
    payload = {
        "query": query,
        "limit": _MAX_RESULTS,
        "sources": ["web"],
        "timeout": _API_TIMEOUT_MS,
    }
    last_response: requests.Response | None = None
    started = time.monotonic()

    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = requests.post(
                _SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout):
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
            continue

        last_response = response
        if response.status_code not in {408, 409, 429} and response.status_code < 500:
            break
        if attempt == _MAX_ATTEMPTS - 1:
            break
        time.sleep(2**attempt)

    if last_response is None:
        raise RuntimeError("Firecrawl no devolvió una respuesta")

    logger.info(
        "web_search: firecrawl query=%r status=%d duration_ms=%d",
        query,
        last_response.status_code,
        round((time.monotonic() - started) * 1000),
    )
    return last_response


def _extract_results(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_data = payload.get("data")
    if isinstance(raw_data, Mapping):
        raw_results = raw_data.get("web")
    else:
        raw_results = raw_data
    if not isinstance(raw_results, list):
        return []

    results: list[dict[str, str]] = []
    for raw_result in raw_results[:_MAX_RESULTS]:
        if not isinstance(raw_result, Mapping):
            continue
        url = _clean_text(raw_result.get("url"), max_chars=2_000)
        if not url:
            continue
        results.append(
            {
                "title": _clean_text(raw_result.get("title"), max_chars=300),
                "url": url,
                "description": _clean_text(
                    raw_result.get("description"),
                    max_chars=_MAX_DESCRIPTION_CHARS,
                ),
            }
        )
    return results


def _parse_search_response(response: requests.Response, query: str) -> ToolResult:
    if not response.ok:
        return ToolResult(
            output=(
                f"Error de búsqueda: Firecrawl respondió HTTP {response.status_code}: "
                f"{_response_error(response)}"
            ),
            metadata={"status_code": response.status_code},
        )

    try:
        payload = response.json()
    except ValueError:
        return ToolResult(output="Error de búsqueda: Firecrawl devolvió JSON inválido.")
    if not isinstance(payload, Mapping) or payload.get("success") is not True:
        return ToolResult(output=f"Error de búsqueda: {_response_error(response)}")

    results = _extract_results(payload)
    logger.info(
        "web_search: firecrawl query=%r results=%d credits=%s request_id=%s",
        query,
        len(results),
        payload.get("creditsUsed"),
        payload.get("id"),
    )
    return ToolResult(
        output=json.dumps(
            {
                "query": query,
                "results": results,
                "instruction": "Usá estas fuentes para responder y citá sus URLs.",
            },
            ensure_ascii=False,
        ),
        metadata={
            "query": query,
            "result_count": len(results),
            "credits_used": payload.get("creditsUsed"),
            "request_id": payload.get("id"),
        },
    )


def _execute_web_search(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    query = _clean_text(params.get("query"), max_chars=500)
    if not query:
        return ToolResult(output="Error de búsqueda: falta la consulta.")

    api_key = str(os.environ.get("FIRECRAWL_API_KEY") or "").strip()
    if not api_key:
        return ToolResult(output="Error de búsqueda: Firecrawl no está configurado.")

    try:
        response = _search_request(query, api_key)
    except requests.Timeout:
        return ToolResult(output="Error de búsqueda: Firecrawl agotó el tiempo de espera.")
    except requests.ConnectionError:
        return ToolResult(output="Error de búsqueda: no se pudo conectar con Firecrawl.")
    return _parse_search_response(response, query)


register_tool(
    name="web_search",
    description=(
        "Search the public web with Firecrawl. Use it for current facts or when the user "
        "asks you to search. The result contains source URLs that you must cite."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise web search query",
                "maxLength": 500,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    executor=_execute_web_search,
    requires_env=["FIRECRAWL_API_KEY"],
    requires_context=["web_search_enabled"],
)
