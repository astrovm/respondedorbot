"""Tool registry for agentic function calling.

Tools are registered with a JSON schema (OpenAI function calling format) and
an executor callable. The LLM decides which tools to call; this registry
validates and dispatches those calls.
"""

from __future__ import annotations

import json
import logging
import os
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
    cast,
)

from api.core.rust_bridge import load_rust_bridge


logger = logging.getLogger(__name__)


class _RustToolRegistryPolicy(Protocol):
    def tool_parse_arguments(self, raw: str) -> str | None: ...

    def tool_select_available(
        self,
        tools_json: str,
        context_provided: bool,
        task_mode: bool,
    ) -> list[int]: ...


def _load_rust_tool_registry_policy() -> _RustToolRegistryPolicy | None:
    module = load_rust_bridge("RUST_TOOL_REGISTRY_POLICY_ENABLED")
    if module is None:
        return None
    return cast(_RustToolRegistryPolicy, module)


def _rust_tool_registry_failed(operation: str) -> None:
    logger.exception(
        "Rust tool registry policy failed; using Python fallback: operation=%s",
        operation,
    )


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    requires_env: List[str] = field(default_factory=list)
    requires_context: List[str] = field(default_factory=list)
    task_allowed: bool = True


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
_schema_cache: Optional[List[Tuple[ToolSchema, Dict[str, Any]]]] = None


def _invalidate_schema_cache() -> None:
    global _schema_cache
    _schema_cache = None


def register_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    executor: Callable[..., ToolResult],
    requires_env: Optional[List[str]] = None,
    requires_context: Optional[List[str]] = None,
    task_allowed: bool = True,
) -> None:
    schema = ToolSchema(
        name=name,
        description=description,
        parameters=parameters,
        requires_env=list(requires_env) if requires_env else [],
        requires_context=list(requires_context) if requires_context else [],
        task_allowed=task_allowed,
    )
    _TOOL_REGISTRY[name] = (schema, executor)
    _invalidate_schema_cache()


def _get_schema_cache() -> List[Tuple[ToolSchema, Dict[str, Any]]]:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = [
            (s, {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            })
            for s, _ in _TOOL_REGISTRY.values()
        ]
    return _schema_cache


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
    *,
    task_mode: bool = False,
) -> List[Dict[str, Any]]:
    cache = _get_schema_cache()

    rust = _load_rust_tool_registry_policy()
    if rust is not None:
        try:
            facts = [
                {
                    "environment_requirements_met": all(
                        os.environ.get(env_key) is not None
                        for env_key in schema.requires_env
                    ),
                    "context_requirements_met": (
                        context is None
                        or all(
                            ctx_key in context and context[ctx_key] is not None
                            for ctx_key in schema.requires_context
                        )
                    ),
                    "task_allowed": schema.task_allowed,
                }
                for schema, _entry in cache
            ]
            indices = rust.tool_select_available(
                json.dumps(facts, separators=(",", ":")),
                context is not None,
                task_mode,
            )
            if any(index < 0 or index >= len(cache) for index in indices):
                raise ValueError("Rust selected an invalid tool schema index")
            return [cache[index][1] for index in indices]
        except Exception:
            _rust_tool_registry_failed("availability")

    if context is None and not task_mode:
        return [entry for _, entry in cache]

    return [
        entry
        for schema, entry in cache
        if _tool_is_available(schema, context)
        and (not task_mode or schema.task_allowed)
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
    _, executor = entry
    try:
        return executor(params, context or {})
    except Exception as e:
        return ToolResult(output=f"Tool '{name}' error: {e}")


def parse_tool_call_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        rust = _load_rust_tool_registry_policy()
        if rust is not None:
            try:
                normalized = rust.tool_parse_arguments(arguments)
                if normalized is not None:
                    parsed = json.loads(normalized)
                    if not isinstance(parsed, dict):
                        raise ValueError("Rust returned non-object tool arguments")
                    return cast(Dict[str, Any], parsed)
            except Exception:
                _rust_tool_registry_failed("arguments")
        try:
            parsed = json.loads(arguments)
            return cast(Dict[str, Any], parsed) if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


TOOL_REGISTRY = _TOOL_REGISTRY
