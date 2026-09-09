"""A tiny tool registry.

Register a Python function with @tool and it becomes callable by the model:
the JSON schema handed to the API is generated from the function signature,
so there is exactly one place to edit when you add a capability.
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
from datetime import datetime, timezone
from typing import Any, Callable, get_type_hints

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}

REGISTRY: dict[str, "Tool"] = {}


class Tool:
    def __init__(self, fn: Callable[..., Any], description: str) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.description = description
        self.schema = _schema_from_signature(fn, description)

    def __call__(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def _schema_from_signature(fn: Callable[..., Any], description: str) -> dict[str, Any]:
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        json_type = _PY_TO_JSON.get(hints.get(name, str), "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool(description: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[fn.__name__] = Tool(fn, description)
        return fn

    return decorator


def schemas() -> list[dict[str, Any]]:
    return [t.schema for t in REGISTRY.values()]


def call(name: str, raw_arguments: str) -> str:
    """Execute a tool call from the model and always return a string."""
    tool_obj = REGISTRY.get(name)
    if tool_obj is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        kwargs = json.loads(raw_arguments or "{}")
        result = tool_obj(**kwargs)
    except Exception as exc:  # surfaced back to the model so it can recover
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------
# Built-in tools. Delete what you do not need, add what your case does need.
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("only + - * / // % ** and numbers are allowed")


@tool("Evaluate an arithmetic expression. Use it instead of doing mental math.")
def calculator(expression: str) -> str:
    value = _eval_node(ast.parse(expression, mode="eval").body)
    return f"{expression} = {value:g}"


@tool("Current UTC date and time in ISO-8601 format.")
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@tool(
    "Semantic search over the documents the user uploaded. "
    "Returns the most relevant chunks with their source file names."
)
def search_docs(query: str, top_k: int = 4) -> str:
    from .rag import store  # imported lazily to keep this module import-cheap

    hits = store.search(query, top_k=top_k)
    if not hits:
        return "No documents have been uploaded yet."
    return "\n\n".join(f"[{h.source}] {h.text}" for h in hits)
