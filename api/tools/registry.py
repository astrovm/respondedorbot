"""Tool registry for agentic function calling.

Tools are registered with a JSON schema (OpenAI function calling format) and
an executor callable. The LLM decides which tools to call; this registry
validates and dispatches those calls.
"""

from __future__ import annotations

import os
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
    requires_env: List[str] = field(default_factory=list)
    requires_context: List[str] = field(default_factory=list)


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
_tool_schemas_cache: Optional[List[Dict[str, Any]]] = None


def _invalidate_schema_cache() -> None:
    global _tool_schemas_cache
    _tool_schemas_cache = None


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    executor: Callable[..., ToolResult],
    requires_env: Optional[List[str]] = None,
    requires_context: Optional[List[str]] = None,
) -> None:
    schema = ToolSchema(
        name=name,
        description=description,
        parameters=parameters,
        requires_env=list(requires_env) if requires_env else [],
        requires_context=list(requires_context) if requires_context else [],
    )
    _TOOL_REGISTRY[name] = (schema, executor)
    _invalidate_schema_cache()


def _tool_schema_to_dict(schema: ToolSchema) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        },
    }


def _tool_is_available(
    schema: ToolSchema,
    context: Optional[Dict[str, Any]],
) -> bool:
    for env_key in schema.requires_env:
        if os.environ.get(env_key) is None:
            return False
    if context is not None:
        for ctx_key in schema.requires_context:
            if ctx_key not in context or context[ctx_key] is None:
                return False
    return True


def get_all_tool_schemas(
    context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    global _tool_schemas_cache

    if _tool_schemas_cache is None:
        _tool_schemas_cache = [
            _tool_schema_to_dict(s) for s, _ in _TOOL_REGISTRY.values()
        ]

    if context is None:
        return list(_tool_schemas_cache)

    return [
        entry
        for schema, entry in zip(
            (s for s, _ in _TOOL_REGISTRY.values()), _tool_schemas_cache
        )
        if _tool_is_available(schema, context)
    ]


def reset_tool_schemas_cache() -> None:
    _invalidate_schema_cache()


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
