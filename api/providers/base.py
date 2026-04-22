"""AI provider abstraction layer.

Provides a unified interface for multiple AI backends (OpenRouter, Groq, etc.)
with support for both completion and streaming modes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable

from api.ai_pricing import AIUsageResult


class AIProvider(Protocol):
    """Protocol for AI providers supporting completions."""

    @property
    def name(self) -> str:
        ...

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
    ) -> Iterator[str]:
        """Stream response tokens. Tool calls are disabled during streaming."""
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
    ) -> ProviderResult:
        """Try each provider in order until one returns a result."""
        available = self.available_providers
        if not available:
            return ProviderResult(result=None, provider_name="none")

        for idx, provider in enumerate(available):
            try:
                result = provider.complete(
                    system_message,
                    messages,
                    enable_web_search=enable_web_search,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
                )
                if result is not None:
                    return ProviderResult(
                        result=result,
                        provider_name=provider.name,
                        fallback_used=idx > 0,
                    )
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
    ) -> Iterator[Tuple[str, str]]:
        """Stream from the first available streaming provider.

        Yields (provider_name, token) tuples.
        Falls back to non-streaming if no streaming provider is available.
        """
        for provider in self.available_providers:
            if not isinstance(provider, StreamingAIProvider):
                continue
            try:
                yield provider.name, ""
                for token in provider.stream(
                    system_message,
                    messages,
                    enable_web_search=enable_web_search,
                ):
                    yield provider.name, token
                return
            except Exception as e:
                print(f"Streaming provider {provider.name} failed: {e}")
                continue

        yield "none", ""
