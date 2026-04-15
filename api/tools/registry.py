"""Tool registry for agentic function calling.

Tools are registered with a JSON schema (OpenAI function calling format) and
an executor callable. The LLM decides which tools to call; this registry
validates and dispatches those calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ToolResult:
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolExecutor(Protocol):
    def __call__(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolResult: ...


_TOOL_REGISTRY: Dict[str, Tuple[ToolSchema, Callable[..., ToolResult]]] = {}


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    executor: Callable[..., ToolResult],
) -> None:
    schema = ToolSchema(name=name, description=description, parameters=parameters)
    _TOOL_REGISTRY[name] = (schema, executor)


def get_all_tool_schemas() -> List[Dict[str, Any]]:
    result = []
    for schema, _ in _TOOL_REGISTRY.values():
        result.append(
            {
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters,
                },
            }
        )
    return result


def execute_tool(
    name: str,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    entry = _TOOL_REGISTRY.get(name)
    if entry is None:
        return ToolResult(output=f"Unknown tool: {name}")
    schema, executor = entry
    try:
        return executor(params, context or {})
    except Exception as e:
        return ToolResult(output=f"Tool '{name}' error: {e}")


def parse_tool_call_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


TOOL_REGISTRY = _TOOL_REGISTRY
