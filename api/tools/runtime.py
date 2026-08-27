"""Apply validated AI tool calls to a conversation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from api.core.logging import format_log_context, get_logger
from api.providers.types import AssistantMessageLike, ToolCallLike
from api.tools.registry import TOOL_REGISTRY, execute_tool, parse_tool_call_arguments


logger = get_logger(__name__)

_FIRECRAWL_CREDITS_CONTEXT_KEY = "_firecrawl_credits_used"


def _default_log(message: str) -> None:
    logger.info(message)


class ToolRuntime:
    def __init__(
        self,
        execute_tool_fn: Callable[
            [str, Dict[str, Any], Dict[str, Any]], Any
        ] = execute_tool,
        parse_tool_call_arguments_fn: Callable[
            [Any], Dict[str, Any]
        ] = parse_tool_call_arguments,
        tool_registry: Mapping[str, Any] = TOOL_REGISTRY,
        print_fn: Callable[[str], None] = _default_log,
    ) -> None:
        self._execute_tool_fn = execute_tool_fn
        self._parse_tool_call_arguments_fn = parse_tool_call_arguments_fn
        self._tool_registry = tool_registry
        self._print_fn = print_fn

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_registry

    @staticmethod
    def take_firecrawl_credits(tool_context: Mapping[str, Any] | None) -> int:
        if not isinstance(tool_context, dict):
            return 0
        try:
            return max(0, int(tool_context.pop(_FIRECRAWL_CREDITS_CONTEXT_KEY, 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _record_firecrawl_credits(
        tool_name: str,
        result: Any,
        tool_context: Mapping[str, Any],
    ) -> None:
        if tool_name != "web_search" or not isinstance(tool_context, dict):
            return
        metadata = getattr(result, "metadata", None)
        if not isinstance(metadata, Mapping):
            return
        try:
            credits_used = max(0, int(metadata.get("credits_used") or 0))
        except (TypeError, ValueError):
            return
        if credits_used:
            current = ToolRuntime.take_firecrawl_credits(tool_context)
            tool_context[_FIRECRAWL_CREDITS_CONTEXT_KEY] = current + credits_used

    def apply_tool_calls(
        self,
        message: AssistantMessageLike,
        tool_calls: Sequence[ToolCallLike],
        current_messages: List[Dict[str, Any]],
        tool_context: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if getattr(message, "content", None):
            assistant_msg["content"] = str(message.content)
        assistant_msg["tool_calls"] = []
        current_messages.append(assistant_msg)

        for tool_call in tool_calls:
            fn = getattr(tool_call, "function", None)
            if fn is None:
                continue
            tool_name = str(getattr(fn, "name", "") or "")
            if not self.has_tool(tool_name):
                self._print_fn(
                    "tool call skipped: not registered "
                    f"tool_name={tool_name}{format_log_context(tool_context)}"
                )
                continue

            tc_id = str(getattr(tool_call, "id", "") or "")
            raw_args = getattr(fn, "arguments", "{}")
            tool_params = self._parse_tool_call_arguments_fn(raw_args)

            self._print_fn(
                f"tool call: {tool_name} params={tool_params}{format_log_context(tool_context)}"
            )
            execution_context = dict(tool_context)
            execution_context.pop(_FIRECRAWL_CREDITS_CONTEXT_KEY, None)
            result = self._execute_tool_fn(tool_name, tool_params, execution_context)
            self._record_firecrawl_credits(tool_name, result, tool_context)
            self._print_fn(
                f"tool result: {tool_name} output={result.output[:200]!r}"
                f"{format_log_context(tool_context)}"
            )

            assistant_msg["tool_calls"].append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": raw_args,
                    },
                }
            )
            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result.output,
                }
            )

        return current_messages
