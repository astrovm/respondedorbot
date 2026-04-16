"""calculate tool — safe math evaluator for agentic use."""

from __future__ import annotations

import ast
from typing import Any, Dict, Union

from api.tools.registry import ToolResult, register_tool

_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def _safe_eval(expr: str) -> Union[float, int, str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return f"expresion invalida: {expr}"
    if sum(1 for _ in ast.walk(tree)) > 200:
        return "expresion demasiado larga"
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return f"expresion no permitida: {expr}"
    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "no se puede dividir por cero"
    except Exception as e:
        return f"error: {e}"
    if isinstance(result, float):
        if result == int(result):
            return int(result)
        return round(result, 8)
    return result


def _execute_calculate(
    params: Dict[str, Any],
    context: Dict[str, Any],
) -> ToolResult:
    expression = params.get("expression", "")
    if not expression:
        return ToolResult(output="no se proporciono una expresion")
    result = _safe_eval(str(expression))
    return ToolResult(output=str(result))


register_tool(
    name="calculate",
    description="Evaluate a mathematical expression safely. Supports +, -, *, /, %, **. Example: '2 ** 10' or '100 / 35000 * 100'.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g. '150000 / 35000 * 100'",
            },
        },
        "required": ["expression"],
    },
    executor=_execute_calculate,
)
