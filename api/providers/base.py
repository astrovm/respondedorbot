"""AI provider abstraction layer.

Provides a unified interface for multiple AI backends (OpenRouter, Groq, etc.)
with support for both completion and streaming modes.
"""

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    cast,
    runtime_checkable,
)

from api.ai.pricing import AIUsageResult
from api.billing.authorization import AIAuthorizationDenied
from api.core.rust_bridge import load_rust_bridge


logger = logging.getLogger(__name__)


class _RustProviderChainPolicy(Protocol):
    def provider_chain_select(self, availability: list[bool]) -> list[int]: ...

    def provider_chain_outcome(
        self,
        available_provider_names: list[str],
        successful_position: int | None,
    ) -> tuple[str, bool]: ...


def _load_rust_provider_chain_policy() -> _RustProviderChainPolicy | None:
    module = load_rust_bridge("RUST_PROVIDER_CHAIN_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustProviderChainPolicy, module)


def _rust_provider_chain_failed(operation: str) -> None:
    logger.exception(
        "Rust provider chain policy failed; using Python fallback: operation=%s",
        operation,
    )


def _select_available_indices(availability: list[bool]) -> list[int]:
    rust = _load_rust_provider_chain_policy()
    if rust is not None:
        try:
            indices = rust.provider_chain_select(availability)
            if (
                any(type(index) is not int for index in indices)
                or indices != sorted(set(indices))
                or any(index < 0 or index >= len(availability) for index in indices)
                or any(not availability[index] for index in indices)
            ):
                raise ValueError("Rust returned invalid provider indices")
            return indices
        except Exception:
            _rust_provider_chain_failed("selection")
    return [index for index, available in enumerate(availability) if available]


def _provider_chain_outcome(
    available_provider_names: list[str],
    successful_position: int | None,
) -> tuple[str, bool]:
    rust = _load_rust_provider_chain_policy()
    if rust is not None:
        try:
            provider_name, fallback_used = rust.provider_chain_outcome(
                available_provider_names,
                successful_position,
            )
            return str(provider_name), bool(fallback_used)
        except Exception:
            _rust_provider_chain_failed("outcome")
    if successful_position is not None:
        return available_provider_names[successful_position], successful_position > 0
    return (
        available_provider_names[-1] if available_provider_names else "none",
        False,
    )


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
        availability = [provider.is_available() for provider in self._providers]
        return [self._providers[index] for index in _select_available_indices(availability)]

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
            provider_name, fallback_used = _provider_chain_outcome([], None)
            return ProviderResult(
                result=None,
                provider_name=provider_name,
                fallback_used=fallback_used,
            )

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
                    provider_name, fallback_used = _provider_chain_outcome(
                        [item.name for item in available],
                        idx,
                    )
                    return ProviderResult(
                        result=result,
                        provider_name=provider_name,
                        fallback_used=fallback_used,
                    )
            except AIAuthorizationDenied:
                raise
            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue

        provider_name, fallback_used = _provider_chain_outcome(
            [item.name for item in available],
            None,
        )
        return ProviderResult(
            result=None,
            provider_name=provider_name,
            fallback_used=fallback_used,
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
