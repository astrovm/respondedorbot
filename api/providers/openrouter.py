"""OpenRouter provider implementation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

from api.ai_pricing import AIUsageResult, CHAT_OUTPUT_TOKEN_LIMIT
from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
from api.tool_runtime import ToolRuntime
from api.providers.base import StreamingAIProvider


class OpenRouterProvider(StreamingAIProvider):
    """OpenRouter provider using the OpenAI SDK."""

    def __init__(
        self,
        *,
        get_client: Callable[[], Any],
        admin_report: Callable[..., Any],
        increment_request_count: Callable[[], Any],
        build_web_search_tool: Callable[[], Dict[str, Any]],
        build_usage_result: Callable[..., AIUsageResult],
        extract_usage_map: Callable[[Any], Dict[str, Any]],
        primary_model: str,
        max_tool_rounds: int = 5,
        tool_runtime: Optional[ToolRuntime] = None,
    ) -> None:
        self._get_client = get_client
        self._admin_report = admin_report
        self._increment_request_count = increment_request_count
        self._build_web_search_tool = build_web_search_tool
        self._build_usage_result = build_usage_result
        self._extract_usage_map = extract_usage_map
        self._primary_model = primary_model
        self._max_tool_rounds = max_tool_rounds
        self._tool_runtime = tool_runtime or ToolRuntime()
        self._runtime = ProviderRuntime(
            ProviderRuntimeDeps(
                get_client=get_client,
                admin_report=admin_report,
                increment_request_count=increment_request_count,
                build_web_search_tool=build_web_search_tool,
                build_usage_result=build_usage_result,
                extract_usage_map=extract_usage_map,
                primary_model=primary_model,
                max_tool_rounds=max_tool_rounds,
            ),
            self._tool_runtime,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    def is_available(self) -> bool:
        client = self._get_client()
        return client is not None

    def complete(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AIUsageResult]:
        return self._runtime.complete(
            system_message,
            messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )

    def stream(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        if client is None:
            return

        has_tools = bool(extra_tools) or enable_web_search

        try:
            if not has_tools:
                self._increment_request_count()
                request_kwargs: Dict[str, Any] = {
                    "model": self._primary_model,
                    "messages": [system_message] + list(messages),
                    "max_tokens": max_tokens if max_tokens is not None else CHAT_OUTPUT_TOKEN_LIMIT,
                    "stream": True,
                }

                for chunk in client.chat.completions.create(**request_kwargs):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return

            if enable_web_search and not extra_tools:
                self._increment_request_count()
                request_kwargs = {
                    "model": self._primary_model,
                    "messages": [system_message] + list(messages),
                    "max_tokens": max_tokens if max_tokens is not None else CHAT_OUTPUT_TOKEN_LIMIT,
                    "stream": True,
                    "tools": [self._build_web_search_tool()],
                }

                has_tool_calls = False
                for chunk in client.chat.completions.create(**request_kwargs):
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if getattr(choice, "finish_reason", None) == "tool_calls":
                        has_tool_calls = True
                        break
                    if delta and delta.content:
                        yield delta.content
                if has_tool_calls:
                    final_messages = self._runtime._execute_tool_rounds(
                        current_messages=list(messages),
                        system_message=system_message,
                        enable_web_search=enable_web_search,
                        extra_tools=extra_tools,
                        tool_context=tool_context,
                    )
                    if final_messages is None:
                        return
                    self._increment_request_count()
                    request_kwargs = {
                        "model": self._primary_model,
                        "messages": [system_message] + final_messages,
                        "max_tokens": max_tokens if max_tokens is not None else CHAT_OUTPUT_TOKEN_LIMIT,
                        "stream": True,
                    }
                    for chunk in client.chat.completions.create(**request_kwargs):
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                return

            final_messages = self._runtime._execute_tool_rounds(
                current_messages=list(messages),
                system_message=system_message,
                enable_web_search=enable_web_search,
                extra_tools=extra_tools,
                tool_context=tool_context,
            )
            if final_messages is None:
                return

            self._increment_request_count()
            request_kwargs = {
                "model": self._primary_model,
                "messages": [system_message] + final_messages,
                "max_tokens": max_tokens if max_tokens is not None else CHAT_OUTPUT_TOKEN_LIMIT,
                "stream": True,
            }

            for chunk in client.chat.completions.create(**request_kwargs):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as error:
            self._admin_report(
                f"OpenRouter stream error model={self._primary_model}",
                error,
                {"model": self._primary_model},
            )
            raise
