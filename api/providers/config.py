from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlparse, urlunparse

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


def get_openrouter_base_url(*, environment: Mapping[str, str]) -> str:
    value = _clean_value(environment.get("CF_AIG_BASE_URL"))
    if not value or "gateway.ai.cloudflare.com" not in value:
        return DEFAULT_OPENROUTER_URL
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if not path:
        return DEFAULT_OPENROUTER_URL
    base_path = path.rsplit("/", 1)[0]
    openrouter_path = f"{base_path}/openrouter" if base_path else "/openrouter"
    return urlunparse(parsed._replace(path=openrouter_path))


def build_openrouter_client(
    *,
    get_api_key: Callable[[], str | None],
    get_base_url: Callable[[], str | None],
    environment: Mapping[str, str],
    client_factory: Callable[..., Any],
    default_headers: Mapping[str, str] | None = None,
) -> Any | None:
    api_key = get_api_key()
    base_url = get_base_url()
    if not api_key or not base_url:
        return None
    headers = dict(default_headers or {})
    gateway_token = environment.get("CF_AIG_TOKEN")
    if gateway_token:
        headers["cf-aig-authorization"] = f"Bearer {gateway_token}"
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": 60.0,
    }
    if headers:
        kwargs["default_headers"] = headers
    return client_factory(**kwargs)


def build_groq_openai_client(
    account: str,
    *,
    get_api_key: Callable[[str], str | None],
    environment: Mapping[str, str],
    client_factory: Callable[..., Any],
    default_headers: Mapping[str, str] | None = None,
) -> Any | None:
    api_key = get_api_key(account)
    if not api_key:
        print(f"Groq API key not configured for account={account}")
        return None
    headers = dict(default_headers or {})
    gateway_token = environment.get("CF_AIG_TOKEN")
    if gateway_token:
        headers["cf-aig-authorization"] = f"Bearer {gateway_token}"
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": environment.get(
            "CF_AIG_BASE_URL", "https://api.groq.com/openai/v1"
        ),
    }
    if headers:
        kwargs["default_headers"] = headers
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
            "max_total_results": max_results * max_queries,
        },
    }


def _clean_value(value: str | None) -> str | None:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None
