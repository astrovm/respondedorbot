"""AI provider clients, cooldowns, usage accounting, and provider selection."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextvars import ContextVar, Token
from logging import Logger
from typing import Any

from groq import Groq
from openai import OpenAI

from api.providers import config as provider_config
from api.providers import support as provider_support
from api.ai.pricing import AIUsageResult, calculate_billing_for_segments, ensure_mapping
from api.providers.backoff import (
    get_provider_cooldown_remaining,
    is_provider_cooled_down,
    mark_provider_cooldown,
)
from api.providers.errors import (
    extract_rate_limit_backoff_seconds,
    is_rate_limit_error,
)
from api.providers import OpenRouterProvider, ProviderChain
from api.tools.runtime import ToolRuntime


class ProviderService:
    """Own everything shared by calls to OpenRouter and Groq.

    In simple terms:
    1. Build authenticated clients from the environment.
    2. Skip accounts that recently hit a rate limit.
    3. Turn provider responses into the bot's common usage format.
    4. Add successful usage to the billing metadata.

    Keeping this state here prevents chat, media, and summary code from each
    implementing a slightly different provider policy.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str],
        admin_report: Any,
        logger: Logger,
        tool_runtime: ToolRuntime,
        primary_model: str,
        account_order: tuple[str, ...],
        default_backoff_seconds: int,
        web_search_max_results: int,
        web_search_max_queries: int,
        max_tool_rounds: int,
    ) -> None:
        self.environment = environment
        self.admin_report = admin_report
        self.logger = logger
        self.tool_runtime = tool_runtime
        self.primary_model = primary_model
        self.account_order = account_order
        self.default_backoff_seconds = default_backoff_seconds
        self.web_search_max_results = web_search_max_results
        self.web_search_max_queries = web_search_max_queries
        self.max_tool_rounds = max_tool_rounds
        self.openai_client_factory = OpenAI
        self.groq_client_factory = Groq
        # ContextVar keeps concurrent messages from sharing request counts.
        self._request_count: ContextVar[int] = ContextVar(
            "ai_provider_request_count",
            default=0,
        )
        self._chain: ProviderChain | None = None

    def reset_request_count(self) -> Token[int]:
        return self._request_count.set(0)

    def restore_request_count(self, token: Token[int]) -> None:
        self._request_count.reset(token)

    def increment_request_count(self) -> None:
        self._request_count.set(int(self._request_count.get() or 0) + 1)

    def get_request_count(self) -> int:
        return int(self._request_count.get() or 0)

    def set_backoff(self, provider: str, duration: int | None) -> None:
        if provider and duration:
            mark_provider_cooldown(provider.lower(), float(max(0, duration)))

    def get_backoff_remaining(self, provider: str) -> float:
        return get_provider_cooldown_remaining(provider.lower()) if provider else 0.0

    def is_backoff_active(self, provider: str) -> bool:
        return is_provider_cooled_down(provider.lower()) if provider else False

    def get_groq_api_key(self, account: str) -> str | None:
        return provider_config.get_groq_api_key(
            account,
            environment=self.environment,
        )

    def get_groq_accounts(self) -> list[str]:
        return provider_config.get_configured_groq_accounts(
            self.account_order,
            get_api_key=self.get_groq_api_key,
        )

    def get_groq_backoff_key(self, account: str, scope: str) -> str:
        return f"groq:{account}:{scope}".lower()

    def get_openrouter_api_key(self) -> str | None:
        return provider_config.get_openrouter_api_key(environment=self.environment)

    def get_openrouter_base_url(self) -> str | None:
        return provider_config.get_openrouter_base_url(environment=self.environment)

    def get_openrouter_client(
        self,
        *,
        default_headers: Mapping[str, str] | None = None,
    ) -> Any | None:
        return provider_config.build_openrouter_client(
            get_api_key=self.get_openrouter_api_key,
            get_base_url=self.get_openrouter_base_url,
            environment=self.environment,
            client_factory=self.openai_client_factory,
            default_headers=default_headers,
        )

    def get_groq_client(
        self,
        account: str,
        *,
        default_headers: Mapping[str, str] | None = None,
    ) -> Any | None:
        return provider_config.build_groq_openai_client(
            account,
            get_api_key=self.get_groq_api_key,
            environment=self.environment,
            client_factory=self.openai_client_factory,
            default_headers=default_headers,
        )

    def get_groq_native_client(self, account: str) -> Any | None:
        return provider_config.build_groq_native_client(
            account,
            get_api_key=self.get_groq_api_key,
            client_factory=self.groq_client_factory,
        )

    def build_web_search_tool(self) -> dict[str, Any]:
        return provider_config.build_web_search_tool(
            self.web_search_max_results,
            self.web_search_max_queries,
        )

    def invoke(
        self,
        provider_name: str,
        *,
        attempt: Any,
        rate_limit_backoff: int | None = None,
        label: str | None = None,
        backoff_key: str | None = None,
    ) -> Any | None:
        """Run one provider attempt with the shared cooldown policy."""

        return provider_support.invoke_provider(
            provider_name,
            attempt=attempt,
            rate_limit_backoff=rate_limit_backoff,
            label=label,
            backoff_key=backoff_key,
            is_backoff_active=self.is_backoff_active,
            get_backoff_remaining=self.get_backoff_remaining,
            is_rate_limit_error=is_rate_limit_error,
            extract_backoff_seconds=extract_rate_limit_backoff_seconds,
            set_backoff=self.set_backoff,
            default_backoff=self.default_backoff_seconds,
        )

    def append_billing_segment(
        self,
        response_meta: dict[str, Any] | None,
        result: AIUsageResult | None,
    ) -> None:
        provider_support.append_billing_segment(response_meta, result)

    def extract_usage_map(self, response: Any) -> dict[str, Any] | None:
        return provider_support.extract_usage_map(
            response,
            ensure_mapping=ensure_mapping,
        )

    def build_usage_result(
        self,
        *,
        kind: str,
        text: str,
        model: str,
        response: Any,
        audio_seconds: float | None = None,
        cached: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> AIUsageResult:
        return provider_support.build_usage_result(
            kind=kind,
            text=text,
            model=model,
            response=response,
            audio_seconds=audio_seconds,
            cached=cached,
            metadata=metadata,
            extract_usage=self.extract_usage_map,
        )

    def log_groq_request_result(
        self,
        *,
        label: str,
        scope: str,
        account: str,
        token_count: int,
        audio_seconds: float,
        result: AIUsageResult | None,
    ) -> None:
        provider_support.log_groq_request_result(
            label=label,
            scope=scope,
            account=account,
            token_count=token_count,
            audio_seconds=audio_seconds,
            result=result,
            calculate_billing=calculate_billing_for_segments,
            ensure_mapping=ensure_mapping,
            logger=self.logger,
        )

    def build_chain(
        self,
        *,
        model: str | None = None,
        max_tool_rounds: int | None = None,
    ) -> ProviderChain:
        """Build the ordered list of chat providers to try."""

        return ProviderChain(
            [
                self.build_provider(
                    model=model,
                    max_tool_rounds=max_tool_rounds,
                )
            ]
        )

    def build_provider(
        self,
        *,
        model: str | None = None,
        max_tool_rounds: int | None = None,
    ) -> OpenRouterProvider:
        return OpenRouterProvider(
            get_client=self.get_openrouter_client,
            admin_report=self.admin_report,
            increment_request_count=self.increment_request_count,
            build_web_search_tool=self.build_web_search_tool,
            build_usage_result=self.build_usage_result,
            extract_usage_map=self.extract_usage_map,
            primary_model=model or self.primary_model,
            max_tool_rounds=(
                self.max_tool_rounds
                if max_tool_rounds is None
                else max_tool_rounds
            ),
            tool_runtime=self.tool_runtime,
        )

    def get_chain(self) -> ProviderChain:
        # Provider objects are stateless for a request, so one chain can be reused.
        if self._chain is None:
            self._chain = self.build_chain()
        return self._chain

    def complete(
        self,
        system_message: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        response_meta: dict[str, Any] | None = None,
        enable_web_search: bool = True,
        extra_tools: list[dict[str, Any]] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> str | None:
        """Return the first successful provider text and record its usage."""

        provider_result = self.get_chain().complete(
            system_message,
            messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )
        if not provider_result.result:
            return None
        self.append_billing_segment(response_meta, provider_result.result)
        self.logger.info(
            "provider response provider=%s",
            provider_result.provider_name,
        )
        return provider_result.result.text

    def stream(
        self,
        system_message: dict[str, Any],
        messages: list[dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: list[dict[str, Any]] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> Iterator[tuple[str, str]]:
        return self.get_chain().stream(
            system_message,
            messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )

    def is_scope_available(self, scope: str) -> bool:
        """Check whether media/chat work should be attempted right now.

        No configured Groq account is treated as available because OpenRouter
        may still handle the request.
        """

        accounts = self.get_groq_accounts()
        return not accounts or any(
            not self.is_backoff_active(self.get_groq_backoff_key(account, scope))
            for account in accounts
        )

    def has_openrouter_fallback(self) -> bool:
        return self.get_openrouter_api_key() is not None


__all__ = ["ProviderService"]
