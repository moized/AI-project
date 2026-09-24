from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone
from typing import Any, Callable


_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def calculator(expression: str) -> str:
    """Safely evaluate a bounded arithmetic expression without eval()."""
    try:
        if len(expression) > 200:
            return "Error: expression is too long."

        tree = ast.parse(expression, mode="eval")

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)

            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value

            if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 20:
                    raise ValueError("exponent too large")
                result = _OPERATORS[type(node.op)](left, right)
                if abs(result) > 1e100:
                    raise ValueError("result too large")
                return result

            raise ValueError("unsupported expression")

        result = evaluate(tree.body)
        return str(int(result) if result.is_integer() else result)

    except Exception as exc:
        return f"Error: {type(exc).__name__}: invalid arithmetic expression."


def get_current_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "calculator": calculator,
    "get_current_date": get_current_date,
}

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Calculate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression such as 2 + 2.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "get_current_date",
        "description": "Return today's UTC date in YYYY-MM-DD format.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
