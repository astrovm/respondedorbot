"""OpenRouter provider implementation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Protocol, cast

from api.ai.pricing import AIUsageResult, chat_output_token_limit
from api.billing.authorization import AIAuthorizationDenied
from api.core.logging import get_logger
from api.core.rust_bridge import load_rust_bridge
from api.providers.runtime import ProviderRuntime, ProviderRuntimeDeps
from api.tools.runtime import ToolRuntime
from api.providers.base import StreamingAIProvider


logger = get_logger(__name__)


class _RustProviderStreamPolicy(Protocol):
    def provider_stream_text_decision(
        self,
        held_text: str,
        content: str,
        hold_all_text: bool,
        text_released: bool,
        possible_pseudo_tools: list[str],
    ) -> tuple[str, str, bool]: ...

    def provider_stream_could_be_pseudo_tool_call(
        self,
        text: str,
        possible_pseudo_tools: list[str],
    ) -> bool: ...

    def provider_stream_accumulate_tool_calls(
        self,
        current_calls_json: str,
        fragments_json: str,
    ) -> str: ...


def _load_rust_provider_stream_policy() -> _RustProviderStreamPolicy | None:
    module = load_rust_bridge("RUST_PROVIDER_STREAM_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustProviderStreamPolicy, module)


def _rust_stream_policy_failed(operation: str) -> None:
    logger.exception(
        "Rust provider stream policy failed; using Python fallback: operation=%s",
        operation,
    )


@dataclass
class _StreamToolCall:
    index: int
    id: str = ""
    type: str = "function"
    name: str = ""
    arguments: str = ""

    def as_message_value(self) -> Any:
        return SimpleNamespace(
            id=self.id,
            type=self.type,
            function=SimpleNamespace(
                name=self.name,
                arguments=self.arguments,
            ),
        )


@dataclass
class _StreamRound:
    text_parts: list[str] = field(default_factory=list)
    annotations: list[Any] = field(default_factory=list)
    tool_calls: dict[int, _StreamToolCall] = field(default_factory=dict)
    finish_reason: Any = None
    usage_response: Any = None
    last_response: Any = None
    resolved_model: str = ""
    upstream_provider: str = ""
    service_tier: str = ""

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def message(self) -> Any:
        calls = [item.as_message_value() for _, item in sorted(self.tool_calls.items())]
        return SimpleNamespace(
            content=self.text,
            annotations=self.annotations,
            tool_calls=calls,
        )


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
        extract_usage_map: Callable[[Any], Optional[Dict[str, Any]]],
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

    def _authorize_stream_round(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tool_context: Dict[str, Any],
        round_idx: int,
        invocation_id: str,
    ) -> None:
        self._runtime.authorize_model_request(
            system_message=system_message,
            messages=messages,
            tool_context=tool_context,
            round_idx=round_idx,
            attempt=0,
            invocation_id=invocation_id,
        )

    def _stream_web_metadata(
        self,
        streamed_round: _StreamRound,
        usage_response: Any,
        message: Any,
    ) -> Dict[str, Any]:
        metadata = self._runtime._web_search_metadata(usage_response, message)
        optional_values = {
            "resolved_model": streamed_round.resolved_model,
            "upstream_provider": streamed_round.upstream_provider,
            "service_tier": streamed_round.service_tier,
        }
        metadata.update({key: value for key, value in optional_values.items() if value})
        return metadata

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
        return self._runtime.complete(
            system_message,
            messages,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
            on_usage_result=on_usage_result,
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
        on_usage_result: Optional[Callable[[AIUsageResult], None]] = None,
    ) -> Iterator[str]:
        client = self._get_client()
        if client is None:
            return

        output_token_limit = chat_output_token_limit(self._primary_model)
        current_messages = list(messages)
        runtime_tool_context, invocation_id = self._runtime.new_invocation(tool_context)
        remaining_web_search_uses = self._runtime._configured_web_search_max_uses(enable_web_search)
        total_web_search_requests = 0
        possible_pseudo_tools = self._extra_tool_names(extra_tools)

        try:
            for round_idx in range(self._max_tool_rounds):
                self._increment_request_count()
                self._authorize_stream_round(
                    system_message,
                    current_messages,
                    runtime_tool_context,
                    round_idx,
                    invocation_id,
                )
                request_kwargs = self._build_stream_request(
                    system_message,
                    current_messages,
                    output_token_limit=output_token_limit,
                    max_tokens=max_tokens,
                    enable_web_search=enable_web_search,
                    extra_tools=extra_tools,
                    web_search_max_uses=remaining_web_search_uses,
                )
                streamed_round = _StreamRound()
                try:
                    streamed_round, held_text, text_released = yield from (
                        self._consume_stream_round(
                            client,
                            request_kwargs,
                            possible_pseudo_tools,
                            hold_all_text=total_web_search_requests > 0,
                            streamed_round=streamed_round,
                        )
                    )
                except Exception:
                    self._report_interrupted_stream_usage(
                        on_usage_result,
                        streamed_round,
                        round_idx,
                    )
                    raise

                message = streamed_round.message()
                usage_response = (
                    streamed_round.usage_response
                    or streamed_round.last_response
                    or SimpleNamespace(usage={})
                )
                web_metadata = self._stream_web_metadata(
                    streamed_round,
                    usage_response,
                    message,
                )
                round_web_search_requests = self._runtime._web_search_request_count(
                    usage_response,
                    message,
                )
                total_web_search_requests += round_web_search_requests
                remaining_web_search_uses = self._runtime._remaining_web_search_uses(
                    remaining_web_search_uses,
                    usage_response,
                    message,
                )

                tool_calls = getattr(message, "tool_calls", None) or []
                known_calls = self._runtime._filter_known_calls(
                    tool_calls,
                    runtime_tool_context,
                    round_idx,
                )
                if known_calls:
                    try:
                        self._runtime._persist_round_before_tools(
                            usage_response,
                            message,
                            round_idx,
                            runtime_tool_context,
                        )
                        current_messages = self._tool_runtime.apply_tool_calls(
                            message,
                            known_calls,
                            current_messages,
                            runtime_tool_context,
                        )
                    except AIAuthorizationDenied:
                        self._report_stream_usage(
                            on_usage_result,
                            usage_response,
                            message,
                            round_idx,
                            web_metadata,
                        )
                        raise
                    self._runtime._add_firecrawl_credits(
                        web_metadata,
                        runtime_tool_context,
                    )
                    self._report_stream_usage(
                        on_usage_result,
                        usage_response,
                        message,
                        round_idx,
                        web_metadata,
                    )
                    continue

                pseudo_call = self._runtime._parse_pseudo_tool_call(
                    streamed_round.text,
                    round_idx,
                    extra_tools,
                )
                if pseudo_call is not None and not text_released:
                    self._report_stream_usage(
                        on_usage_result,
                        usage_response,
                        message,
                        round_idx,
                        web_metadata,
                    )
                    current_messages = self._tool_runtime.apply_tool_calls(
                        SimpleNamespace(content=""),
                        [pseudo_call],
                        current_messages,
                        runtime_tool_context,
                    )
                    continue

                if total_web_search_requests > 0:
                    self._runtime._record_web_search_outcome(
                        streamed_round.text,
                        current_messages,
                        web_metadata,
                        tool_context=runtime_tool_context,
                        round_idx=round_idx,
                    )
                self._report_stream_usage(
                    on_usage_result,
                    usage_response,
                    message,
                    round_idx,
                    web_metadata,
                )

                if held_text:
                    yield held_text

                if streamed_round.finish_reason in {"stop", "length", None}:
                    return

                return
            return
        except Exception as error:
            self._admin_report(
                f"OpenRouter stream error model={self._primary_model}",
                error,
                {"model": self._primary_model},
            )
            raise

    def _report_stream_usage(
        self,
        callback: Optional[Callable[[AIUsageResult], None]],
        response: Any,
        message: Any,
        round_idx: int,
        metadata: Dict[str, Any],
        *,
        text_override: Optional[str] = None,
    ) -> None:
        if callback is None:
            return
        callback(
            self._runtime._build_round_result(
                response,
                message,
                round_idx,
                metadata=metadata,
                text_override=text_override,
            )
        )

    def _report_interrupted_stream_usage(
        self,
        callback: Optional[Callable[[AIUsageResult], None]],
        streamed_round: _StreamRound,
        round_idx: int,
    ) -> None:
        if not (
            streamed_round.text_parts
            or streamed_round.tool_calls
            or streamed_round.usage_response is not None
            or streamed_round.last_response is not None
        ):
            return
        response = (
            streamed_round.usage_response
            or streamed_round.last_response
            or SimpleNamespace(usage={})
        )
        metadata: Dict[str, Any] = {"stream_interrupted": True}
        if streamed_round.resolved_model:
            metadata["resolved_model"] = streamed_round.resolved_model
        if streamed_round.upstream_provider:
            metadata["upstream_provider"] = streamed_round.upstream_provider
        if streamed_round.service_tier:
            metadata["service_tier"] = streamed_round.service_tier
        self._report_stream_usage(
            callback,
            response,
            streamed_round.message(),
            round_idx,
            metadata,
        )

    def _build_stream_request(
        self,
        system_message: Dict[str, Any],
        messages: List[Dict[str, Any]],
        *,
        output_token_limit: int,
        max_tokens: Optional[int],
        enable_web_search: bool,
        extra_tools: Optional[List[Dict[str, Any]]],
        web_search_max_uses: Optional[int],
    ) -> Dict[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self._primary_model,
            "messages": [system_message] + messages,
            "max_tokens": (max_tokens if max_tokens is not None else output_token_limit),
            "stream": True,
        }
        request_tools = self._runtime._build_request_tools(
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            web_search_max_uses=web_search_max_uses,
            request_kwargs=request_kwargs,
        )
        if request_tools:
            request_kwargs["tools"] = request_tools
        return request_kwargs

    def _consume_stream_round(
        self,
        client: Any,
        request_kwargs: Dict[str, Any],
        possible_pseudo_tools: set[str],
        *,
        hold_all_text: bool = False,
        streamed_round: _StreamRound | None = None,
    ) -> Generator[str, None, tuple[_StreamRound, str, bool]]:
        streamed_round = streamed_round or _StreamRound()
        held_text = ""
        text_released = False
        for chunk in client.chat.completions.create(**request_kwargs):
            stream_error = self._field(chunk, "error")
            if stream_error:
                raise RuntimeError(f"OpenRouter stream failed: {stream_error}")
            streamed_round.last_response = chunk
            response_model = self._field(chunk, "model")
            if response_model:
                streamed_round.resolved_model = str(response_model)
            response_provider = self._field(chunk, "provider")
            if response_provider:
                streamed_round.upstream_provider = str(response_provider)
            response_service_tier = self._field(chunk, "service_tier")
            if response_service_tier:
                streamed_round.service_tier = str(response_service_tier)
            if getattr(chunk, "usage", None) is not None:
                streamed_round.usage_response = chunk
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "error":
                choice_error = self._field(choice, "error")
                raise RuntimeError(
                    f"OpenRouter stream failed: {choice_error or 'unknown provider error'}"
                )
            if finish_reason is not None:
                streamed_round.finish_reason = finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            self._accumulate_stream_delta(streamed_round, delta)
            content = str(self._field(delta, "content") or "")
            if not content:
                continue
            held_text, emitted_text, text_released = self._stream_text_decision(
                held_text,
                content,
                hold_all_text,
                text_released,
                possible_pseudo_tools,
            )
            if emitted_text:
                yield emitted_text
        return streamed_round, held_text, text_released

    @staticmethod
    def _stream_text_decision(
        held_text: str,
        content: str,
        hold_all_text: bool,
        text_released: bool,
        possible_pseudo_tools: set[str],
    ) -> tuple[str, str, bool]:
        rust = _load_rust_provider_stream_policy()
        if rust is not None:
            try:
                return rust.provider_stream_text_decision(
                    held_text,
                    content,
                    hold_all_text,
                    text_released,
                    sorted(possible_pseudo_tools),
                )
            except Exception:
                _rust_stream_policy_failed("text_decision")
        if hold_all_text:
            return held_text + content, "", text_released
        if text_released:
            return held_text, content, True
        candidate = held_text + content
        if OpenRouterProvider._python_could_be_pseudo_tool_call(
            candidate,
            possible_pseudo_tools,
        ):
            return candidate, "", False
        return "", candidate, True

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _accumulate_stream_delta(
        cls,
        streamed_round: _StreamRound,
        delta: Any,
    ) -> None:
        content = cls._field(delta, "content")
        if content:
            streamed_round.text_parts.append(str(content))

        annotations = cls._field(delta, "annotations") or []
        streamed_round.annotations.extend(annotations)

        fragments = cls._field(delta, "tool_calls") or []
        if fragments and cls._accumulate_stream_tool_calls_with_rust(
            streamed_round,
            fragments,
        ):
            return
        for position, fragment in enumerate(fragments):
            try:
                index = int(cls._field(fragment, "index", position))
            except TypeError, ValueError:
                index = position
            accumulated = streamed_round.tool_calls.setdefault(
                index,
                _StreamToolCall(index=index),
            )
            call_id = cls._field(fragment, "id")
            call_type = cls._field(fragment, "type")
            function = cls._field(fragment, "function")
            if call_id:
                accumulated.id += str(call_id)
            if call_type:
                accumulated.type = str(call_type)
            if function is not None:
                name = cls._field(function, "name")
                arguments = cls._field(function, "arguments")
                if name:
                    accumulated.name += str(name)
                if arguments:
                    accumulated.arguments += str(arguments)

    @classmethod
    def _accumulate_stream_tool_calls_with_rust(
        cls,
        streamed_round: _StreamRound,
        fragments: Any,
    ) -> bool:
        rust = _load_rust_provider_stream_policy()
        if rust is None:
            return False
        current_calls = [
            {
                "index": call.index,
                "id": call.id,
                "type": call.type,
                "name": call.name,
                "arguments": call.arguments,
            }
            for _, call in sorted(streamed_round.tool_calls.items())
        ]
        normalized_fragments = []
        for position, fragment in enumerate(fragments):
            function = cls._field(fragment, "function")
            normalized_fragments.append(
                {
                    "position": position,
                    "index": cls._field(fragment, "index", position),
                    "id": cls._truthy_string(cls._field(fragment, "id")),
                    "type": cls._truthy_string(cls._field(fragment, "type")),
                    "name": cls._truthy_string(cls._field(function, "name")),
                    "arguments": cls._truthy_string(
                        cls._field(function, "arguments")
                    ),
                }
            )
        try:
            encoded = rust.provider_stream_accumulate_tool_calls(
                json.dumps(current_calls, ensure_ascii=False),
                json.dumps(normalized_fragments, ensure_ascii=False, default=str),
            )
            accumulated_calls = json.loads(encoded)
            streamed_round.tool_calls = {
                int(call["index"]): _StreamToolCall(
                    index=int(call["index"]),
                    id=str(call["id"]),
                    type=str(call["type"]),
                    name=str(call["name"]),
                    arguments=str(call["arguments"]),
                )
                for call in accumulated_calls
            }
        except Exception:
            _rust_stream_policy_failed("tool_call_accumulation")
            return False
        return True

    @staticmethod
    def _truthy_string(value: Any) -> str | None:
        return str(value) if value else None

    @staticmethod
    def _extra_tool_names(
        extra_tools: Optional[List[Dict[str, Any]]],
    ) -> set[str]:
        names: set[str] = set()
        for tool in extra_tools or []:
            function = tool.get("function")
            if isinstance(function, Mapping):
                name = str(function.get("name") or "")
                if name:
                    names.add(name)
        return names

    @staticmethod
    def _could_be_pseudo_tool_call(text: str, tool_names: set[str]) -> bool:
        rust = _load_rust_provider_stream_policy()
        if rust is not None:
            try:
                return bool(
                    rust.provider_stream_could_be_pseudo_tool_call(
                        text,
                        sorted(tool_names),
                    )
                )
            except Exception:
                _rust_stream_policy_failed("pseudo_tool_candidate")
        return OpenRouterProvider._python_could_be_pseudo_tool_call(text, tool_names)

    @staticmethod
    def _python_could_be_pseudo_tool_call(text: str, tool_names: set[str]) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return bool(tool_names)
        return any(
            name.startswith(stripped) or stripped.startswith(f"{name}(") for name in tool_names
        )
