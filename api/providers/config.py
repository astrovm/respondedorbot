"""Resolve provider models, credentials, and endpoint configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def get_groq_api_key(account: str, *, environment: Mapping[str, str]) -> str | None:
    variable = "GROQ_FREE_API_KEY" if account == "free" else "GROQ_API_KEY"
    return _clean_value(environment.get(variable))


def get_configured_groq_accounts(
    account_order: tuple[str, ...],
    *,
    get_api_key: Callable[[str], str | None],
) -> list[str]:
    return [account for account in account_order if get_api_key(account)]


def get_openrouter_api_key(*, environment: Mapping[str, str]) -> str | None:
    return _clean_value(environment.get("OPENROUTER_API_KEY"))


def get_openrouter_base_url() -> str:
    return DEFAULT_OPENROUTER_URL


def build_openrouter_client(
    *,
    get_api_key: Callable[[], str | None],
    get_base_url: Callable[[], str | None],
    client_factory: Callable[..., Any],
    default_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> Any | None:
    api_key = get_api_key()
    base_url = get_base_url()
    if not api_key or not base_url:
        return None
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
        # ProviderRuntime owns retries. Keeping one retry layer prevents the
        # SDK and the application from multiplying slow server-tool attempts.
        "max_retries": 0,
    }
    if default_headers:
        kwargs["default_headers"] = dict(default_headers)
    return client_factory(**kwargs)


def build_groq_native_client(
    account: str,
    *,
    get_api_key: Callable[[str], str | None],
    client_factory: Callable[..., Any],
) -> Any | None:
    api_key = get_api_key(account)
    if not api_key:
        print(f"Groq API key not configured for account={account}")
        return None
    return client_factory(api_key=api_key)


def build_web_search_tool(max_results: int, max_queries: int) -> dict[str, Any]:
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "firecrawl",
            "max_results": max_results,
            "max_uses": max_queries,
            "max_total_results": max_results * max_queries,
        },
    }


def _clean_value(value: str | None) -> str | None:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None
