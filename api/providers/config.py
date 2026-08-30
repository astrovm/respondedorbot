"""Resolve provider models, credentials, and endpoint configuration."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from api.core.logging import get_logger
from api.core.rust_bridge import load_rust_bridge

DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"

logger = get_logger(__name__)


class _RustProviderConfigPolicy(Protocol):
    def provider_groq_api_key(
        self,
        account: str,
        free_api_key: str | None,
        paid_api_key: str | None,
    ) -> str | None: ...

    def provider_configured_groq_accounts(
        self,
        account_order: list[str],
        configured: list[bool],
    ) -> list[str]: ...

    def provider_openrouter_api_key(self, value: str | None) -> str | None: ...

    def provider_openrouter_base_url(self) -> str: ...

    def provider_groq_backoff_key(self, account: str, scope: str) -> str: ...

    def provider_scope_is_available(self, backoff_active: list[bool]) -> bool: ...

    def provider_web_search_tool(self, max_results: int, max_queries: int) -> str: ...


def _load_rust_provider_config_policy() -> _RustProviderConfigPolicy | None:
    module = load_rust_bridge("RUST_PROVIDER_CONFIG_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustProviderConfigPolicy, module)


def _rust_provider_config_failed(operation: str) -> None:
    logger.exception(
        "Rust provider configuration policy failed; using Python fallback: "
        "operation=%s",
        operation,
    )


def get_groq_api_key(account: str, *, environment: Mapping[str, str]) -> str | None:
    free_api_key = environment.get("GROQ_FREE_API_KEY")
    paid_api_key = environment.get("GROQ_API_KEY")
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            return rust.provider_groq_api_key(account, free_api_key, paid_api_key)
        except Exception:
            _rust_provider_config_failed("get_groq_api_key")
    value = free_api_key if account == "free" else paid_api_key
    return _clean_value(value)


def get_configured_groq_accounts(
    account_order: tuple[str, ...],
    *,
    get_api_key: Callable[[str], str | None],
) -> list[str]:
    configured = [bool(get_api_key(account)) for account in account_order]
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            accounts = rust.provider_configured_groq_accounts(
                list(account_order),
                configured,
            )
            if any(account not in account_order for account in accounts):
                raise ValueError("Rust returned an unknown Groq account")
            return accounts
        except Exception:
            _rust_provider_config_failed("get_configured_groq_accounts")
    return [
        account
        for account, is_configured in zip(account_order, configured, strict=True)
        if is_configured
    ]


def get_openrouter_api_key(*, environment: Mapping[str, str]) -> str | None:
    value = environment.get("OPENROUTER_API_KEY")
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            return rust.provider_openrouter_api_key(value)
        except Exception:
            _rust_provider_config_failed("get_openrouter_api_key")
    return _clean_value(value)


def get_openrouter_base_url() -> str:
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            value = str(rust.provider_openrouter_base_url())
            if not value:
                raise ValueError("Rust returned an empty OpenRouter base URL")
            return value
        except Exception:
            _rust_provider_config_failed("get_openrouter_base_url")
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
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            value = json.loads(
                rust.provider_web_search_tool(max_results, max_queries)
            )
            if not isinstance(value, dict):
                raise ValueError("Rust web-search tool must be an object")
            return value
        except Exception:
            _rust_provider_config_failed("build_web_search_tool")
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "firecrawl",
            "max_results": max_results,
            "max_uses": max_queries,
            "max_total_results": max_results * max_queries,
        },
    }


def get_groq_backoff_key(account: str, scope: str) -> str:
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            value = str(rust.provider_groq_backoff_key(account, scope))
            if not value:
                raise ValueError("Rust returned an empty Groq backoff key")
            return value
        except Exception:
            _rust_provider_config_failed("get_groq_backoff_key")
    return f"groq:{account}:{scope}".lower()


def is_scope_available(backoff_active: list[bool]) -> bool:
    rust = _load_rust_provider_config_policy()
    if rust is not None:
        try:
            return bool(rust.provider_scope_is_available(backoff_active))
        except Exception:
            _rust_provider_config_failed("is_scope_available")
    return not backoff_active or any(not active for active in backoff_active)


def _clean_value(value: str | None) -> str | None:
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None
