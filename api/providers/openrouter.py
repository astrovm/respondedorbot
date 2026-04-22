"""OpenRouter provider implementation."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from api.ai_pricing import AIUsageResult, CHAT_OUTPUT_TOKEN_LIMIT
from api.provider_runtime import ProviderRuntime, ProviderRuntimeDeps
from api.tool_runtime import ToolRuntime
from api.providers.base import AIProvider, StreamingAIProvider


class OpenRouterProvider(StreamingAIProvider):
    """OpenRouter provider using the OpenAI SDK."""

    def __init__(
        self,
        *,
        get_client: callable,
        admin_report: callable,
        increment_request_count: callable,
        build_web_search_tool: callable,
        build_usage_result: callable,
        extract_usage_map: callable,
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
    ) -> Iterator[str]:
        client = self._get_client()
        if client is None:
            return

        self._increment_request_count()
        request_kwargs: Dict[str, Any] = {
            "model": self._primary_model,
            "messages": [system_message] + list(messages),
            "max_tokens": CHAT_OUTPUT_TOKEN_LIMIT,
            "stream": True,
        }
        if enable_web_search:
            request_kwargs["extra_body"] = {
                "reasoning": {"effort": "low"}
            }

        try:
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
