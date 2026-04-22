"""Groq provider implementation for chat completions."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional

from api.ai_pricing import AIUsageResult, CHAT_OUTPUT_TOKEN_LIMIT
from api.provider_backoff import (
    is_provider_cooled_down,
    mark_provider_cooldown,
)
from api.providers.base import AIProvider


class GroqChatProvider(AIProvider):
    """Groq provider for chat completions using the OpenAI-compatible SDK."""

    def __init__(
        self,
        *,
        get_client: Callable[[], Any],
        admin_report: Callable[..., Any],
        increment_request_count: Callable[[], Any],
        build_usage_result: Callable[..., AIUsageResult],
        extract_usage_map: Callable[[Any], Dict[str, Any]],
        primary_model: str = "llama-3.1-70b-versatile",
        max_tool_rounds: int = 5,
        backoff_key: str = "groq:chat",
        backoff_seconds: int = 60,
    ) -> None:
        self._get_client = get_client
        self._admin_report = admin_report
        self._increment_request_count = increment_request_count
        self._build_usage_result = build_usage_result
        self._extract_usage_map = extract_usage_map
        self._primary_model = primary_model
        self._max_tool_rounds = max_tool_rounds
        self._backoff_key = backoff_key
        self._backoff_seconds = backoff_seconds

    @property
    def name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        if is_provider_cooled_down(self._backoff_key):
            return False
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
        client = self._get_client()
        if client is None:
            return None

        self._increment_request_count()

        all_messages = [system_message] + list(messages)
        request_kwargs: Dict[str, Any] = {
            "model": self._primary_model,
            "messages": all_messages,
            "max_tokens": CHAT_OUTPUT_TOKEN_LIMIT,
        }
        if extra_tools:
            request_kwargs["tools"] = extra_tools

        try:
            response = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as error:
                    is_json_error = isinstance(error, json.JSONDecodeError) or (
                        "JSONDecodeError" in type(error).__name__
                    )
                    if is_json_error and attempt < 2:
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    raise
        except Exception as error:
            self._admin_report(
                f"Groq chat error model={self._primary_model}",
                error,
                {"model": self._primary_model},
            )
            mark_provider_cooldown(self._backoff_key, self._backoff_seconds)
            return None

        if not response or not hasattr(response, "choices") or not response.choices:
            return None

        choice = response.choices[0]
        if choice.finish_reason == "stop":
            return self._build_usage_result(
                kind="chat",
                text=str(choice.message.content or ""),
                model=self._primary_model,
                response=response,
                metadata={"provider": "groq"},
            )

        return None
