from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from api.ai_pricing import AIUsageResult, CHAT_OUTPUT_TOKEN_LIMIT, ensure_mapping
from api.tool_runtime import ToolRuntime


@dataclass(frozen=True)
class ProviderRuntimeDeps:
    get_client: Callable[[], Any]
    admin_report: Callable[..., None]
    increment_request_count: Callable[[], None]
    build_web_search_tool: Callable[[], Dict[str, Any]]
    build_usage_result: Callable[..., AIUsageResult]
    extract_usage_map: Callable[[Any], Dict[str, Any]]
    primary_model: str
    max_tool_rounds: int = 5


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
        print(f"Trying OpenRouter chat with model={self._deps.primary_model}...")
        return self._run_tool_rounds(
            current_messages=list(messages),
            system_message=system_message,
            enable_web_search=enable_web_search,
            extra_tools=extra_tools,
            tool_context=tool_context,
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
            try:
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

                response = None
                for attempt in range(5):
                    try:
                        response = client.chat.completions.create(**request_kwargs)
                        break
                    except Exception as error:
                        is_json_error = isinstance(error, json.JSONDecodeError) or (
                            "JSONDecodeError" in type(error).__name__
                        )
                        if is_json_error and attempt < 4:
                            wait = 2**attempt
                            raw_body = ""
                            if isinstance(error, json.JSONDecodeError) and error.doc:
                                doc = str(error.doc)
                                raw_body = f" body_len={len(doc)}"
                                if len(doc) > 200:
                                    raw_body += (
                                        f" body_preview={doc[:100]!r}...{doc[-100:]!r}"
                                    )
                                else:
                                    raw_body += f" body={doc!r}"
                            print(
                                f"OpenRouter JSONDecodeError, retrying in {wait}s "
                                f"(attempt {attempt + 1}/5) model={self._deps.primary_model}"
                                f"{raw_body}"
                            )
                            time.sleep(wait)
                            continue
                        raise
            except Exception as error:
                print(
                    f"OpenRouter chat error model={self._deps.primary_model}: {error}"
                )
                self._deps.admin_report(
                    f"OpenRouter chat error model={self._deps.primary_model}",
                    error,
                    {
                        "finish_reason": "error",
                        "enable_web_search": enable_web_search,
                        "tool_round": round_idx,
                    },
                )
                return None

            if not response or not hasattr(response, "choices") or not response.choices:
                return None

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None) or []
                if not extra_tools or not tool_calls:
                    text = str(message.content or "").strip()
                    if text:
                        tool_metadata: Dict[str, Any] = {
                            "provider": "openrouter",
                            "tool_rounds": round_idx + 1,
                        }
                        return self._deps.build_usage_result(
                            kind="chat",
                            text=text,
                            model=self._deps.primary_model,
                            response=response,
                            metadata=tool_metadata,
                        )
                    break

                known_calls = []
                for tool_call in tool_calls:
                    fn = getattr(tool_call, "function", None)
                    if fn is None:
                        continue
                    tool_name = getattr(fn, "name", "")
                    if not self._tool_runtime.has_tool(tool_name):
                        print(f"Tool call skipped (not registered): {tool_name}")
                        continue
                    known_calls.append(tool_call)

                if not known_calls:
                    text = str(message.content or "").strip()
                    if text:
                        metadata = {
                            "provider": "openrouter",
                            "tool_rounds": round_idx + 1,
                        }
                        return self._deps.build_usage_result(
                            kind="chat",
                            text=text,
                            model=self._deps.primary_model,
                            response=response,
                            metadata=metadata,
                        )
                    break

                current_messages = self._tool_runtime.apply_tool_calls(
                    message,
                    known_calls,
                    current_messages,
                    tool_context or {},
                )

                self._deps.increment_request_count()
                continue

            if finish_reason == "stop":
                metadata: Dict[str, Any] = {"provider": "openrouter"}
                message = choice.message
                usage_map = self._deps.extract_usage_map(response) or {}
                server_tool_use = ensure_mapping(usage_map.get("server_tool_use")) or {}
                web_search_requests = server_tool_use.get("web_search_requests")
                if web_search_requests is not None:
                    try:
                        metadata["web_search_requests"] = int(web_search_requests)
                    except (TypeError, ValueError):
                        pass
                if "web_search_requests" not in metadata:
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
                metadata["tool_rounds"] = round_idx + 1
                return self._deps.build_usage_result(
                    kind="chat",
                    text=str(message.content or ""),
                    model=self._deps.primary_model,
                    response=response,
                    metadata=metadata,
                )

            if finish_reason == "length":
                message = choice.message
                return self._deps.build_usage_result(
                    kind="chat",
                    text=str(message.content or ""),
                    model=self._deps.primary_model,
                    response=response,
                    metadata={
                        "provider": "openrouter",
                        "truncated": True,
                        "tool_rounds": round_idx + 1,
                    },
                )

            print(
                f"provider_runtime: unexpected finish_reason={finish_reason!r} model={self._deps.primary_model}"
            )
            self._deps.admin_report(
                f"OpenRouter unexpected finish_reason={finish_reason!r}",
                extra_context={
                    "model": self._deps.primary_model,
                    "enable_web_search": enable_web_search,
                },
            )
            break

        return None
