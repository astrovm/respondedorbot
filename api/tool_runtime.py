from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence

from api.tools.registry import TOOL_REGISTRY, execute_tool, parse_tool_call_arguments


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
        print_fn: Callable[[str], None] = print,
    ) -> None:
        self._execute_tool_fn = execute_tool_fn
        self._parse_tool_call_arguments_fn = parse_tool_call_arguments_fn
        self._tool_registry = tool_registry
        self._print_fn = print_fn

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_registry

    def apply_tool_calls(
        self,
        message: Any,
        tool_calls: Sequence[Any],
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
                self._print_fn(f"Tool call skipped (not registered): {tool_name}")
                continue

            tc_id = str(getattr(tool_call, "id", "") or "")
            raw_args = getattr(fn, "arguments", "{}")
            tool_params = self._parse_tool_call_arguments_fn(raw_args)

            self._print_fn(f"Tool call: {tool_name}({tool_params})")
            result = self._execute_tool_fn(tool_name, tool_params, dict(tool_context))
            self._print_fn(f"Tool result: {result.output[:200]}")

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
