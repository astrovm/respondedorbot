"""Run provider completions, retries, and tool-call rounds."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from api.ai.pricing import AIUsageResult, chat_output_token_limit, ensure_mapping
from api.core.logging import format_log_context, get_logger
from api.providers.types import (
    EmptyAssistantMessage,
    ToolCall,
    ToolCallLike,
    ToolFunctionCall,
)
from api.tools.runtime import ToolRuntime


logger = get_logger(__name__)
_MAX_RETRIES = 2
_UNGROUNDED_SEARCH_MESSAGE = (
    "No pude verificar eso. La búsqueda web falló o no devolvió resultados útiles; "
    "probá de nuevo en un momento."
)
_MAX_LOGGED_SEARCH_ANSWER_CHARS = 4_000
_PSEUDO_TOOL_CALL_PATTERN = re.compile(
    r'^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<arguments>.*)\)\s*$',
    re.DOTALL,
)
_DSML_TOOL_CALL_PATTERN = re.compile(
    r'<｜｜DSML｜｜invoke\s+name="(?P<name>[A-Za-z_][A-Za-z0-9_]*)"\s*>\s*'
    r'<｜｜DSML｜｜parameter\s+name="url"\s+string="true"\s*>'
    r'(?P<url>https?://[^<\s]+)'
    r'</｜｜DSML｜｜parameter>\s*'
    r'</｜｜DSML｜｜invoke>',
    re.DOTALL,
)


def _is_retryable_provider_error(error: Exception) -> bool:
    if _is_json_decode_error(error):
        return True
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def _is_json_decode_error(error: Exception) -> bool:
    return isinstance(error, json.JSONDecodeError) or (
        "JSONDecodeError" in type(error).__name__
    )


def _format_provider_error_body(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError) and error.doc:
        doc = str(error.doc)
        return _format_body_preview(doc)
    response = getattr(error, "response", None)
    text = getattr(response, "text", "")
    if text:
        return _format_body_preview(str(text))
    return ""


def _format_body_preview(body: str) -> str:
    result = f" body_len={len(body)}"
    if len(body) > 200:
        return f"{result} body_preview={body[:100]!r}...{body[-100:]!r}"
    return f"{result} body={body!r}"


def _extra_tool_names(extra_tools: Optional[List[Dict[str, Any]]]) -> set[str]:
    names: set[str] = set()
    for tool in extra_tools or []:
        function = tool.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
        else:
            name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


@dataclass(frozen=True)
class ProviderRuntimeDeps:
    get_client: Callable[[], Any]
    admin_report: Callable[..., None]
    increment_request_count: Callable[[], None]
    build_web_search_tool: Callable[[], Dict[str, Any]]
    build_usage_result: Callable[..., AIUsageResult]
    extract_usage_map: Callable[[Any], Optional[Dict[str, Any]]]
    primary_model: str
    max_tool_rounds: int = 5


@dataclass(frozen=True)
class ToolRoundDecision:
    messages: List[Dict[str, Any]]
    result: Optional[AIUsageResult] = None
    continue_rounds: bool = False


@dataclass(frozen=True)
class StreamRoundDecision:
    messages: Optional[List[Dict[str, Any]]]
    continue_rounds: bool = False


class ProviderRuntime:
    def __init__(self, deps: ProviderRuntimeDeps, tool_runtime: ToolRuntime) -> None:
        self._deps = deps
        self._tool_runtime = tool_runtime

    def complete(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        enable_web_search: bool = True,
        extra_tools: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[AIUsageResult]:
        context = dict(tool_context or {})
        context["model"] = self._deps.primary_model
        logger.info(
            "openrouter: calling chat enable_web_search=%s extra_tools=%d%s",
            enable_web_search,
            len(extra_tools or []),
            format_log_context(context),
        )
        return self._run_tool_rounds(
            current_messages=list(messages),
            system_message=system_message,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
        )

    def _run_chat_completion(
        self,
        *,
        client: Any,
        system_message: Dict[str, Any],
        current_messages: List[Dict[str, Any]],
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
        web_search_max_uses: Optional[int] = None,
    ) -> Optional[Any]:
        """Build request, retry on transient errors, and return the response."""
        request_kwargs: Dict[str, Any] = {
            "model": self._deps.primary_model,
            "messages": [system_message] + current_messages,
            "max_tokens": chat_output_token_limit(self._deps.primary_model),
        }

        tools_list = self._build_request_tools(
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            web_search_max_uses=web_search_max_uses,
            request_kwargs=request_kwargs,
        )
        if tools_list:
            request_kwargs["tools"] = tools_list

        try:
            for attempt in range(_MAX_RETRIES):
                try:
                    if attempt:
                        self._deps.increment_request_count()
                    response = client.chat.completions.create(**request_kwargs)
                except Exception as error:
                    if _is_retryable_provider_error(error) and attempt < _MAX_RETRIES - 1:
                        wait = 2**attempt
                        raw_body = _format_provider_error_body(error)
                        retry_context = dict(tool_context or {})
                        retry_context.update(
                            {
                                "model": self._deps.primary_model,
                                "tool_round": round_idx + 1,
                            }
                        )
                        logger.warning(
                            "openrouter: transient chat error retrying in %ss attempt=%d/%d error_type=%s%s%s",
                            wait,
                            attempt + 1,
                            _MAX_RETRIES,
                            type(error).__name__,
                            format_log_context(retry_context),
                            raw_body,
                        )
                        time.sleep(wait)
                        continue
                    raise
                choices = getattr(response, "choices", None) or []
                finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
                if (
                    choices
                    and self._is_retryable_finish_response(
                        response,
                        choices[0],
                        finish_reason,
                    )
                    and attempt < _MAX_RETRIES - 1
                ):
                    wait = 2**attempt
                    retry_context = dict(tool_context or {})
                    retry_context.update(
                        {
                            "model": self._deps.primary_model,
                            "tool_round": round_idx + 1,
                            "finish_reason": finish_reason,
                            **self._response_diagnostics(response, choices[0]),
                        }
                    )
                    logger.warning(
                        "openrouter: retryable finish_reason=%r retrying in %ss attempt=%d/%d%s",
                        finish_reason,
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                        format_log_context(retry_context),
                    )
                    time.sleep(wait)
                    continue
                return response
        except Exception as error:
            error_context = dict(tool_context or {})
            error_context.update(
                {"model": self._deps.primary_model, "tool_round": round_idx + 1}
            )
            provider_error_body = _format_provider_error_body(error)
            logger.error(
                "openrouter: chat error %s error=%s%s",
                format_log_context(error_context),
                error,
                provider_error_body,
            )
            self._deps.admin_report(
                f"OpenRouter chat error model={self._deps.primary_model}",
                error,
                {
                    "finish_reason": "error",
                    "enable_web_search": enable_web_search,
                    "tool_round": round_idx + 1,
                    "provider_error_body": provider_error_body,
                },
            )
            return None
        return None

    def _is_retryable_finish_response(
        self,
        response: Any,
        choice: Any,
        finish_reason: Any,
    ) -> bool:
        diagnostics = self._response_diagnostics(response, choice)
        if (
            diagnostics["has_content"]
            or diagnostics["tool_call_count"]
            or self._response_has_usage(response)
        ):
            return False
        if finish_reason is None:
            return True
        if finish_reason != "error":
            return False

        error = self._response_error(response, choice)
        code = error.get("code")
        try:
            status_code = int(code) if code is not None else 0
        except (TypeError, ValueError):
            status_code = 0
        if status_code in {408, 409, 429} or status_code >= 500:
            return True
        error_type = str((ensure_mapping(error.get("metadata")) or {}).get("error_type") or "")
        return error_type in {
            "rate_limit_exceeded",
            "provider_overloaded",
            "provider_unavailable",
            "server",
            "timeout",
        }

    def _response_has_usage(self, response: Any) -> bool:
        usage = self._deps.extract_usage_map(response) or {}
        for key in (
            "cost",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ):
            try:
                if float(usage.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        server_tool_use = ensure_mapping(usage.get("server_tool_use")) or {}
        try:
            return int(server_tool_use.get("web_search_requests") or 0) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _response_error(response: Any, choice: Any) -> Dict[str, Any]:
        return (
            ensure_mapping(getattr(choice, "error", None))
            or ensure_mapping(getattr(response, "error", None))
            or {}
        )

    @staticmethod
    def _response_diagnostics(response: Any, choice: Any) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        fields = {
            "response_id": getattr(response, "id", None),
            "request_id": getattr(response, "_request_id", None),
            "response_model": getattr(response, "model", None),
            "provider": getattr(response, "provider", None),
            "native_finish_reason": getattr(choice, "native_finish_reason", None),
        }
        diagnostics.update({key: value for key, value in fields.items() if value is not None})

        error = ProviderRuntime._response_error(response, choice)
        if error:
            if error.get("code") is not None:
                diagnostics["provider_error_code"] = error["code"]
            metadata = ensure_mapping(error.get("metadata")) or {}
            if metadata.get("error_type") is not None:
                diagnostics["provider_error_type"] = metadata["error_type"]

        message = getattr(choice, "message", None)
        diagnostics["has_content"] = bool(str(getattr(message, "content", "") or "").strip())
        diagnostics["tool_call_count"] = len(getattr(message, "tool_calls", None) or [])
        return diagnostics

    def _filter_known_calls(
        self,
        tool_calls: List[ToolCallLike],
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> List[ToolCallLike]:
        known_calls: List[ToolCallLike] = []
        for tool_call in tool_calls:
            fn = getattr(tool_call, "function", None)
            if fn is None:
                continue
            tool_name = getattr(fn, "name", "")
            if not self._tool_runtime.has_tool(tool_name):
                skipped_context = dict(tool_context or {})
                skipped_context.update(
                    {
                        "model": self._deps.primary_model,
                        "tool_round": round_idx + 1,
                    }
                )
                logger.warning(
                    "tool call skipped: not registered tool_name=%s%s",
                    tool_name,
                    format_log_context(skipped_context),
                )
                continue
            known_calls.append(tool_call)
        return known_calls

    def _run_round_choice(
        self,
        *,
        client: Any,
        system_message: Dict[str, Any],
        current_messages: List[Dict[str, Any]],
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
        web_search_max_uses: Optional[int] = None,
    ) -> Optional[tuple[Any, Any]]:
        response = self._run_chat_completion(
            client=client,
            system_message=system_message,
            current_messages=current_messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
            round_idx=round_idx,
            web_search_max_uses=web_search_max_uses,
        )
        choices = getattr(response, "choices", None) if response is not None else None
        return (response, choices[0]) if choices else None

    def _parse_pseudo_tool_call(
        self,
        text: str,
        round_idx: int,
        extra_tools: Optional[List[Dict[str, Any]]],
    ) -> ToolCall | None:
        dsml_match = _DSML_TOOL_CALL_PATTERN.search(str(text or ""))
        if dsml_match:
            return self._build_pseudo_tool_call(
                tool_name=dsml_match.group("name"),
                url=dsml_match.group("url"),
                round_idx=round_idx,
                extra_tools=extra_tools,
            )

        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        candidate = lines[-1] if lines else ""
        match = _PSEUDO_TOOL_CALL_PATTERN.match(candidate)
        if not match:
            return None

        tool_name = match.group("name")
        url = self._parse_pseudo_tool_url(match.group("arguments"))
        if url is None:
            return None
        return self._build_pseudo_tool_call(
            tool_name=tool_name,
            url=url,
            round_idx=round_idx,
            extra_tools=extra_tools,
        )

    @staticmethod
    def _parse_pseudo_tool_url(raw_arguments: str) -> Optional[str]:
        raw_arguments = raw_arguments.strip()
        if raw_arguments.startswith(("'", '"')):
            try:
                url = ast.literal_eval(raw_arguments)
            except (SyntaxError, ValueError):
                return None
        else:
            try:
                params = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                return None
            if not isinstance(params, dict):
                return None
            url = params.get("url")

        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
        return None

    def _build_pseudo_tool_call(
        self,
        *,
        tool_name: str,
        url: str,
        round_idx: int,
        extra_tools: Optional[List[Dict[str, Any]]],
    ) -> ToolCall | None:
        if tool_name != "web_fetch":
            return None
        if tool_name not in _extra_tool_names(extra_tools):
            return None
        if not self._tool_runtime.has_tool(tool_name):
            return None
        return ToolCall(
            id=f"pseudo_call_{round_idx + 1}",
            function=ToolFunctionCall(
                name=tool_name,
                arguments=json.dumps({"url": url}),
            ),
        )

    def _handle_stream_structured_calls(
        self,
        *,
        message: Any,
        current_messages: List[Dict[str, Any]],
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> StreamRoundDecision:
        tool_calls = getattr(message, "tool_calls", None) or []
        if not extra_tools or not tool_calls:
            return StreamRoundDecision(
                current_messages if str(message.content or "").strip() else None
            )

        known_calls = self._filter_known_calls(tool_calls, tool_context, round_idx)
        if not known_calls:
            return StreamRoundDecision(
                current_messages if str(message.content or "").strip() else None
            )

        messages = self._tool_runtime.apply_tool_calls(
            message,
            known_calls,
            current_messages,
            tool_context or {},
        )
        self._deps.increment_request_count()
        return StreamRoundDecision(messages, continue_rounds=True)

    def _handle_stream_stop(
        self,
        *,
        message: Any,
        current_messages: List[Dict[str, Any]],
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> StreamRoundDecision:
        pseudo_call = self._parse_pseudo_tool_call(
            str(getattr(message, "content", "") or ""),
            round_idx,
            extra_tools,
        )
        if pseudo_call is None:
            return StreamRoundDecision(current_messages)

        messages = self._tool_runtime.apply_tool_calls(
            EmptyAssistantMessage(),
            [pseudo_call],
            current_messages,
            tool_context or {},
        )
        self._deps.increment_request_count()
        return StreamRoundDecision(messages, continue_rounds=True)

    def _execute_tool_rounds(
        self,
        *,
        current_messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Run tool rounds and return final messages for streaming.

        Returns None on failure or when no content to stream.
        """
        client = self._deps.get_client()
        if client is None:
            return None

        self._deps.increment_request_count()
        remaining_web_search_uses = self._configured_web_search_max_uses(
            enable_web_search
        )
        for round_idx in range(self._deps.max_tool_rounds):
            round_response = self._run_round_choice(
                client=client,
                system_message=system_message,
                current_messages=current_messages,
                enable_web_search=enable_web_search,
                extra_tools=extra_tools,
                tool_context=tool_context,
                round_idx=round_idx,
                web_search_max_uses=remaining_web_search_uses,
            )
            if round_response is None:
                return None
            response, choice = round_response
            message = choice.message
            remaining_web_search_uses = self._remaining_web_search_uses(
                remaining_web_search_uses,
                response,
                message,
            )
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                decision = self._handle_stream_structured_calls(
                    message=message,
                    current_messages=current_messages,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
                    round_idx=round_idx,
                )
            elif finish_reason == "stop":
                decision = self._handle_stream_stop(
                    message=message,
                    current_messages=current_messages,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
                    round_idx=round_idx,
                )
            else:
                return current_messages

            if not decision.continue_rounds:
                return decision.messages
            current_messages = decision.messages or current_messages

        return current_messages

    def _build_round_result(
        self,
        response: Any,
        message: Any,
        round_idx: int,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        text_override: Optional[str] = None,
    ) -> AIUsageResult:
        result_metadata = {
            "provider": "openrouter",
            "tool_rounds": round_idx + 1,
            **(metadata or {}),
        }
        return self._deps.build_usage_result(
            kind="chat",
            text=(
                text_override
                if text_override is not None
                else str(getattr(message, "content", "") or "")
            ),
            model=self._deps.primary_model,
            response=response,
            metadata=result_metadata,
        )

    def _handle_structured_tool_calls(
        self,
        *,
        response: Any,
        message: Any,
        round_idx: int,
        current_messages: List[Dict[str, Any]],
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        total_web_search_requests: int = 0,
    ) -> ToolRoundDecision:
        tool_calls = getattr(message, "tool_calls", None) or []
        known_calls = (
            self._filter_known_calls(tool_calls, tool_context, round_idx)
            if extra_tools and tool_calls
            else []
        )
        if not known_calls:
            text = str(getattr(message, "content", "") or "").strip()
            result = (
                self._build_round_result(
                    response,
                    message,
                    round_idx,
                    metadata={"web_search_requests": total_web_search_requests},
                )
                if text
                else None
            )
            return ToolRoundDecision(current_messages, result=result)

        updated_messages = self._tool_runtime.apply_tool_calls(
            message,
            known_calls,
            current_messages,
            tool_context or {},
        )
        self._deps.increment_request_count()
        return ToolRoundDecision(updated_messages, continue_rounds=True)

    def _web_search_metadata(self, response: Any, message: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        usage_map = self._deps.extract_usage_map(response) or {}
        server_tool_use = ensure_mapping(usage_map.get("server_tool_use")) or {}
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is not None:
            try:
                metadata["web_search_requests"] = int(web_search_requests)
            except (TypeError, ValueError):
                pass
        direct_search_requests = self._direct_web_search_request_count(message)
        if direct_search_requests:
            metadata["web_search_requests"] = direct_search_requests
        annotations = getattr(message, "annotations", None) or []
        citation_count = sum(
            1
            for annotation in annotations
            if (
            getattr(annotation, "type", None) == "url_citation"
            or (
                isinstance(annotation, Mapping)
                and str(annotation.get("type") or "") == "url_citation"
            )
            )
        )
        metadata["web_search_citation_count"] = citation_count
        if "web_search_requests" not in metadata and citation_count:
            metadata["web_search_requests"] = 1
        if int(metadata.get("web_search_requests") or 0) > 0:
            metadata["web_search_grounded"] = citation_count > 0
        return metadata

    @staticmethod
    def _web_search_max_uses(tool: Mapping[str, Any]) -> int:
        parameters = ensure_mapping(tool.get("parameters")) or {}
        try:
            return max(0, int(parameters.get("max_uses") or 0))
        except (TypeError, ValueError):
            return 0

    def _configured_web_search_max_uses(self, enabled: bool) -> Optional[int]:
        if not enabled:
            return 0
        configured = self._web_search_max_uses(self._deps.build_web_search_tool())
        return configured or None

    def _build_request_tools(
        self,
        *,
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        web_search_max_uses: Optional[int],
        request_kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Web search is an application-managed Firecrawl function in
        # extra_tools. Do not add OpenRouter's opaque server-managed tool.
        tools = list(extra_tools or [])
        if enable_web_search and web_search_max_uses != 0:
            return tools
        return [
            tool
            for tool in tools
            if str((ensure_mapping(tool.get("function")) or {}).get("name") or "")
            != "web_search"
        ]

    @staticmethod
    def _limit_web_search_tool(tool: Dict[str, Any], remaining: int) -> None:
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            return
        limited = max(0, int(remaining))
        parameters["max_uses"] = limited
        try:
            max_results = max(0, int(parameters.get("max_results") or 0))
            current_total = max(0, int(parameters.get("max_total_results") or 0))
        except (TypeError, ValueError):
            return
        if max_results and current_total:
            parameters["max_total_results"] = min(
                current_total,
                max_results * limited,
            )

    def _web_search_request_count(self, response: Any, message: Any = None) -> int:
        usage_map = self._deps.extract_usage_map(response) or {}
        server_tool_use = ensure_mapping(usage_map.get("server_tool_use")) or {}
        try:
            request_count = max(
                0,
                int(server_tool_use.get("web_search_requests") or 0),
            )
        except (TypeError, ValueError):
            request_count = 0
        if request_count:
            return request_count
        direct_search_requests = self._direct_web_search_request_count(message)
        if direct_search_requests:
            return direct_search_requests
        annotations = getattr(message, "annotations", None) or []
        if any(
            getattr(annotation, "type", None) == "url_citation"
            or (
                isinstance(annotation, Mapping)
                and str(annotation.get("type") or "") == "url_citation"
            )
            for annotation in annotations
        ):
            return 1
        return 0

    @staticmethod
    def _direct_web_search_request_count(message: Any) -> int:
        count = 0
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if str(getattr(function, "name", "") or "") == "web_search":
                count += 1
        return count

    @staticmethod
    def _web_search_call_ids(
        current_messages: List[Dict[str, Any]],
    ) -> set[str]:
        web_search_call_ids: set[str] = set()
        for item in current_messages:
            if str(item.get("role") or "") != "assistant":
                continue
            for tool_call in item.get("tool_calls") or []:
                call = ensure_mapping(tool_call) or {}
                function = ensure_mapping(call.get("function")) or {}
                if str(function.get("name") or "") != "web_search":
                    continue
                call_id = str(call.get("id") or "")
                if call_id:
                    web_search_call_ids.add(call_id)
        return web_search_call_ids

    @classmethod
    def _web_search_source_urls(
        cls,
        current_messages: List[Dict[str, Any]],
    ) -> list[str]:
        web_search_call_ids = cls._web_search_call_ids(current_messages)
        source_urls: list[str] = []
        for item in current_messages:
            if (
                str(item.get("role") or "") != "tool"
                or str(item.get("tool_call_id") or "") not in web_search_call_ids
            ):
                continue
            try:
                payload = json.loads(str(item.get("content") or ""))
            except (json.JSONDecodeError, TypeError):
                continue
            result_items = (ensure_mapping(payload) or {}).get("results")
            if not isinstance(result_items, list):
                continue
            for result_item in result_items:
                result = ensure_mapping(result_item) or {}
                source_url = str(result.get("url") or "").rstrip(".,);]")
                if source_url and source_url not in source_urls:
                    source_urls.append(source_url)
        return source_urls

    def _finalize_web_search_answer(
        self,
        text: str,
        current_messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        *,
        tool_context: Optional[Dict[str, Any]],
        round_idx: int,
    ) -> Optional[str]:
        source_urls = self._web_search_source_urls(current_messages)
        metadata["web_search_source_count"] = len(source_urls)
        clean_text = text.strip()
        if source_urls and clean_text:
            metadata["web_search_grounded"] = True
            return None

        if metadata.get("web_search_citation_count") and clean_text:
            metadata["web_search_grounded"] = True
            return None

        metadata["web_search_grounded"] = False
        warning_context = dict(tool_context or {})
        warning_context.update(
            {
                "model": self._deps.primary_model,
                "tool_round": round_idx + 1,
                "raw_answer_length": len(text),
                **metadata,
            }
        )
        logger.warning(
            "openrouter: rejecting web-search answer raw_answer=%r%s",
            text[:_MAX_LOGGED_SEARCH_ANSWER_CHARS],
            format_log_context(warning_context),
        )
        return _UNGROUNDED_SEARCH_MESSAGE

    def _remaining_web_search_uses(
        self,
        remaining: Optional[int],
        response: Any,
        message: Any = None,
    ) -> Optional[int]:
        if remaining is None:
            return None
        return max(
            0,
            remaining - self._web_search_request_count(response, message),
        )

    @classmethod
    def _apply_web_search_limits(
        cls,
        request_kwargs: Dict[str, Any],
        tool: Mapping[str, Any],
    ) -> None:
        max_uses = cls._web_search_max_uses(tool)
        if max_uses:
            # OpenRouter applies this cap to all server-tool calls in the
            # request. It is a second guard in addition to max_uses.
            request_kwargs["extra_body"] = {"max_tool_calls": max_uses}

    def _handle_stop_response(
        self,
        *,
        response: Any,
        message: Any,
        round_idx: int,
        current_messages: List[Dict[str, Any]],
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
        total_web_search_requests: int = 0,
    ) -> ToolRoundDecision:
        pseudo_call = self._parse_pseudo_tool_call(
            str(getattr(message, "content", "") or ""),
            round_idx,
            extra_tools,
        )
        if pseudo_call is not None:
            updated_messages = self._tool_runtime.apply_tool_calls(
                EmptyAssistantMessage(),
                [pseudo_call],
                current_messages,
                tool_context or {},
            )
            self._deps.increment_request_count()
            return ToolRoundDecision(updated_messages, continue_rounds=True)

        web_metadata = self._web_search_metadata(response, message)
        text_override = None
        if total_web_search_requests > 0:
            web_metadata["web_search_requests"] = total_web_search_requests
            text_override = self._finalize_web_search_answer(
                str(getattr(message, "content", "") or ""),
                current_messages,
                web_metadata,
                tool_context=tool_context,
                round_idx=round_idx,
            )

        return ToolRoundDecision(
            current_messages,
            result=self._build_round_result(
                response,
                message,
                round_idx,
                metadata=web_metadata,
                text_override=text_override,
            ),
        )

    def _report_unexpected_finish(
        self,
        response: Any,
        choice: Any,
        finish_reason: Any,
        round_idx: int,
        *,
        enable_web_search: bool,
        tool_context: Optional[Dict[str, Any]],
    ) -> None:
        unexpected_context = dict(tool_context or {})
        unexpected_context.update(
            {
                "model": self._deps.primary_model,
                "tool_round": round_idx + 1,
                **self._response_diagnostics(response, choice),
            }
        )
        logger.warning(
            "provider_runtime: unexpected finish_reason=%r%s",
            finish_reason,
            format_log_context(unexpected_context),
        )
        self._deps.admin_report(
            f"OpenRouter unexpected finish_reason={finish_reason!r}",
            extra_context={
                "model": self._deps.primary_model,
                "enable_web_search": enable_web_search,
                "tool_round": round_idx + 1,
                **self._response_diagnostics(response, choice),
            },
        )

    def _run_tool_rounds(
        self,
        *,
        current_messages: List[Dict[str, Any]],
        system_message: Dict[str, Any],
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
    ) -> Optional[AIUsageResult]:
        client = self._deps.get_client()
        if client is None:
            return None

        self._deps.increment_request_count()
        remaining_web_search_uses = self._configured_web_search_max_uses(
            enable_web_search
        )
        total_web_search_requests = 0
        for round_idx in range(self._deps.max_tool_rounds):
            round_response = self._run_round_choice(
                client=client,
                system_message=system_message,
                current_messages=current_messages,
                enable_web_search=enable_web_search,
                extra_tools=extra_tools,
                tool_context=tool_context,
                round_idx=round_idx,
                web_search_max_uses=remaining_web_search_uses,
            )
            if round_response is None:
                return None
            response, choice = round_response
            message = choice.message
            round_web_search_requests = self._web_search_request_count(
                response,
                message,
            )
            total_web_search_requests += round_web_search_requests
            remaining_web_search_uses = self._remaining_web_search_uses(
                remaining_web_search_uses,
                response,
                message,
            )
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                decision = self._handle_structured_tool_calls(
                    response=response,
                    message=message,
                    round_idx=round_idx,
                    current_messages=current_messages,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
                    total_web_search_requests=total_web_search_requests,
                )
                if decision.result is not None:
                    return decision.result
                if decision.continue_rounds:
                    current_messages = decision.messages
                    continue
                break

            if finish_reason == "stop":
                decision = self._handle_stop_response(
                    response=response,
                    message=message,
                    round_idx=round_idx,
                    current_messages=current_messages,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
                    total_web_search_requests=total_web_search_requests,
                )
                if decision.continue_rounds:
                    current_messages = decision.messages
                    continue
                return decision.result

            if finish_reason == "length":
                return self._build_round_result(
                    response,
                    message,
                    round_idx,
                    metadata={
                        "truncated": True,
                        "web_search_requests": total_web_search_requests,
                    },
                )

            self._report_unexpected_finish(
                response,
                choice,
                finish_reason,
                round_idx,
                enable_web_search=enable_web_search,
                tool_context=tool_context,
            )
            break

        return None
