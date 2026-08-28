"""AI provider abstraction layer.

Provides a unified interface for multiple AI backends (OpenRouter, Groq, etc.)
with support for both completion and streaming modes.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

from api.ai.pricing import AIUsageResult
from api.billing.authorization import AIAuthorizationDenied


class AIProvider(Protocol):
    """Protocol for AI providers supporting completions."""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool:
        """Return whether this provider is configured and ready."""
        ...

    def complete(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        on_usage_result: Optional[Callable[[AIUsageResult], None]] = None,
    ) -> Optional[AIUsageResult]:
        """Execute a non-streaming completion with optional tool support."""
        ...


@runtime_checkable
class StreamingAIProvider(AIProvider, Protocol):
    """Provider that also supports token streaming."""

    def stream(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        on_usage_result: Optional[Callable[[AIUsageResult], None]] = None,
    ) -> Iterator[str]:
        """Stream response tokens."""
        ...


@dataclass(frozen=True)
class ProviderResult:
    """Result from a provider chain attempt."""

    result: Optional[AIUsageResult]
    provider_name: str
    fallback_used: bool = False


class ProviderChain:
    """Try multiple providers in order until one succeeds."""

    def __init__(self, providers: List[AIProvider]) -> None:
        self._providers = providers

    @property
    def available_providers(self) -> List[AIProvider]:
        return [p for p in self._providers if p.is_available()]

    def has_any_available(self) -> bool:
        return any(p.is_available() for p in self._providers)

    def complete(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        on_usage_result: Optional[Callable[[AIUsageResult], None]] = None,
    ) -> ProviderResult:
        """Try each provider in order until one returns a result."""
        available = self.available_providers
        if not available:
            return ProviderResult(result=None, provider_name="none")

        for idx, provider in enumerate(available):
            try:
                complete_kwargs: Dict[str, Any] = {
                    "enable_web_search": enable_web_search,
                    "extra_tools": extra_tools,
                    "tool_context": tool_context,
                }
                if on_usage_result is not None:
                    complete_kwargs["on_usage_result"] = on_usage_result
                result = provider.complete(
                    system_message,
                    messages,
                    **complete_kwargs,
                )
                if result is not None:
                    return ProviderResult(
                        result=result,
                        provider_name=provider.name,
                        fallback_used=idx > 0,
                    )
            except AIAuthorizationDenied:
                raise
            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue

        return ProviderResult(
            result=None,
            provider_name=available[-1].name if available else "none",
        )

    def stream(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        on_usage_result: Optional[Callable[[AIUsageResult], None]] = None,
    ) -> Iterator[Tuple[str, str]]:
        """Stream from the first available provider.

        All responses are streamed token-by-token. Tool rounds are handled
        internally; only the final assistant text is yielded.
        """
        for provider in self.available_providers:
            try:
                if not isinstance(provider, StreamingAIProvider):
                    continue
                stream_kwargs: Dict[str, Any] = {
                    "enable_web_search": enable_web_search,
                    "extra_tools": extra_tools,
                    "tool_context": tool_context,
                }
                if on_usage_result is not None:
                    stream_kwargs["on_usage_result"] = on_usage_result
                token_iterator = provider.stream(
                    system_message,
                    messages,
                    **stream_kwargs,
                )
                try:
                    first_token = next(token_iterator)
                except StopIteration:
                    continue
                yield provider.name, ""
                yield provider.name, first_token
                for token in token_iterator:
                    yield provider.name, token
                return
            except AIAuthorizationDenied:
                raise
            except Exception as e:
                print(f"Streaming provider {provider.name} failed: {e}")
                continue

        raise RuntimeError("No streaming providers available")
