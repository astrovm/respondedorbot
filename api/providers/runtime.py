from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from api.ai.pricing import AIUsageResult, CHAT_OUTPUT_TOKEN_LIMIT, ensure_mapping
from api.core.logging import format_log_context, get_logger
from api.providers.types import (
    EmptyAssistantMessage,
    ToolCall,
    ToolCallLike,
    ToolFunctionCall,
)
from api.tools.runtime import ToolRuntime


logger = get_logger(__name__)
_MAX_RETRIES = 5
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
    ) -> Optional[Any]:
        """Build request, retry on transient errors, and return the response."""
        request_kwargs: Dict[str, Any] = {
            "model": self._deps.primary_model,
            "messages": [system_message] + current_messages,
            "max_tokens": CHAT_OUTPUT_TOKEN_LIMIT,
        }

        tools_list: List[Dict[str, Any]] = []
        if enable_web_search:
            tools_list.append(self._deps.build_web_search_tool())
        if extra_tools:
            tools_list.extend(extra_tools)
        if tools_list:
            request_kwargs["tools"] = tools_list

        try:
            for attempt in range(_MAX_RETRIES):
                try:
                    return client.chat.completions.create(**request_kwargs)
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
    ) -> Optional[tuple[Any, Any]]:
        response = self._run_chat_completion(
            client=client,
            system_message=system_message,
            current_messages=current_messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
            round_idx=round_idx,
        )
        choices = getattr(response, "choices", None) if response is not None else None
        return (response, choices[0]) if choices else None

    def _parse_pseudo_tool_call(
        self,
        text: str,
        round_idx: int,
        extra_tools: Optional[List[Dict[str, Any]]],
    ) -> ToolCall | None:
        # Some models print tool syntax as text instead of structured tool calls.
        dsml_match = _DSML_TOOL_CALL_PATTERN.search(str(text or ""))
        if dsml_match:
            tool_name = dsml_match.group("name")
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
                    arguments=json.dumps({"url": dsml_match.group("url")}),
                ),
            )

        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        candidate = lines[-1] if lines else ""
        match = _PSEUDO_TOOL_CALL_PATTERN.match(candidate)
        if not match:
            return None

        tool_name = match.group("name")
        if tool_name not in _extra_tool_names(extra_tools):
            return None
        if not self._tool_runtime.has_tool(tool_name):
            return None

        raw_arguments = match.group("arguments").strip()
        if tool_name != "web_fetch":
            return None

        params: Dict[str, Any]
        if raw_arguments.startswith(("'", '"')):
            try:
                url = ast.literal_eval(raw_arguments)
            except (SyntaxError, ValueError):
                return None
            params = {"url": url}
        else:
            try:
                parsed = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                return None
            if not isinstance(parsed, dict):
                return None
            params = parsed

        url = params.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None

        return ToolCall(
            id=f"pseudo_call_{round_idx + 1}",
            function=ToolFunctionCall(
                name=tool_name,
                arguments=json.dumps({"url": url}),
            ),
        )

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
        for round_idx in range(self._deps.max_tool_rounds):
            round_response = self._run_round_choice(
                client=client,
                system_message=system_message,
                current_messages=current_messages,
                enable_web_search=enable_web_search,
                extra_tools=extra_tools,
                tool_context=tool_context,
                round_idx=round_idx,
            )
            if round_response is None:
                return None
            _response, choice = round_response
            finish_reason = choice.finish_reason

            message = choice.message

            if finish_reason == "tool_calls":
                tool_calls = getattr(message, "tool_calls", None) or []
                if not extra_tools or not tool_calls:
                    text = str(message.content or "").strip()
                    if text:
                        return current_messages
                    return None

                known_calls = self._filter_known_calls(
                    tool_calls, tool_context, round_idx
                )
                if not known_calls:
                    # Unknown tools cannot run; retry once as a plain completion.
                    text = str(message.content or "").strip()
                    if text:
                        return current_messages
                    return None

                current_messages = self._tool_runtime.apply_tool_calls(
                    message,
                    known_calls,
                    current_messages,
                    tool_context or {},
                )

                self._deps.increment_request_count()
                continue

            if finish_reason == "stop":
                pseudo_call = self._parse_pseudo_tool_call(
                    str(getattr(message, "content", "") or ""), round_idx, extra_tools
                )
                if pseudo_call is not None:
                    current_messages = self._tool_runtime.apply_tool_calls(
                        EmptyAssistantMessage(),
                        [pseudo_call],
                        current_messages,
                        tool_context or {},
                    )
                    self._deps.increment_request_count()
                    continue

            return current_messages

        return current_messages

    def _build_round_result(
        self,
        response: Any,
        message: Any,
        round_idx: int,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AIUsageResult:
        result_metadata = {
            "provider": "openrouter",
            "tool_rounds": round_idx + 1,
            **(metadata or {}),
        }
        return self._deps.build_usage_result(
            kind="chat",
            text=str(getattr(message, "content", "") or ""),
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
                self._build_round_result(response, message, round_idx)
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
        if "web_search_requests" in metadata:
            return metadata

        annotations = getattr(message, "annotations", None) or []
        has_url_citation = any(
            getattr(annotation, "type", None) == "url_citation"
            or (
                isinstance(annotation, Mapping)
                and str(annotation.get("type") or "") == "url_citation"
            )
            for annotation in annotations
        )
        if has_url_citation:
            metadata["web_search_requests"] = 1
        return metadata

    def _handle_stop_response(
        self,
        *,
        response: Any,
        message: Any,
        round_idx: int,
        current_messages: List[Dict[str, Any]],
        extra_tools: Optional[List[Dict[str, Any]]],
        tool_context: Optional[Dict[str, Any]],
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

        return ToolRoundDecision(
            current_messages,
            result=self._build_round_result(
                response,
                message,
                round_idx,
                metadata=self._web_search_metadata(response, message),
            ),
        )

    def _report_unexpected_finish(
        self,
        finish_reason: Any,
        round_idx: int,
        *,
        enable_web_search: bool,
        tool_context: Optional[Dict[str, Any]],
    ) -> None:
        unexpected_context = dict(tool_context or {})
        unexpected_context.update(
            {"model": self._deps.primary_model, "tool_round": round_idx + 1}
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
        for round_idx in range(self._deps.max_tool_rounds):
            round_response = self._run_round_choice(
                client=client,
                system_message=system_message,
                current_messages=current_messages,
                enable_web_search=enable_web_search,
                extra_tools=extra_tools,
                tool_context=tool_context,
                round_idx=round_idx,
            )
            if round_response is None:
                return None
            response, choice = round_response
            finish_reason = choice.finish_reason
            message = choice.message

            if finish_reason == "tool_calls":
                decision = self._handle_structured_tool_calls(
                    response=response,
                    message=message,
                    round_idx=round_idx,
                    current_messages=current_messages,
                    extra_tools=extra_tools,
                    tool_context=tool_context,
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
                    metadata={"truncated": True},
                )

            self._report_unexpected_finish(
                finish_reason,
                round_idx,
                enable_web_search=enable_web_search,
                tool_context=tool_context,
            )
            break

        return None
