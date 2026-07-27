"""Validated tools exposed to the MathRAG agent."""

from __future__ import annotations

import ast
import multiprocessing as mp
from queue import Empty
from typing import Literal

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolInputError(ValueError):
    """Raised when an agent tool receives unsafe or invalid input."""


class SearchToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("检索问题不能为空")
        return normalized


class CalculatorToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal[
        "simplify",
        "differentiate",
        "integrate",
        "limit",
        "solve",
        "evaluate",
    ]
    expression: str = Field(min_length=1, max_length=240)
    variable: Literal["x", "y", "z", "t", "n"] = "x"
    lower_bound: str | None = Field(default=None, max_length=80)
    upper_bound: str | None = Field(default=None, max_length=80)
    point: str | None = Field(default=None, max_length=80)
    direction: Literal["+", "-", "+-"] = "+-"

    @field_validator("expression", "lower_bound", "upper_bound", "point")
    @classmethod
    def normalize_expression(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("数学表达式不能为空")
        return normalized

    @model_validator(mode="after")
    def validate_operation_arguments(self):
        has_lower = self.lower_bound is not None
        has_upper = self.upper_bound is not None
        if self.operation == "integrate" and has_lower != has_upper:
            raise ValueError("定积分必须同时提供上下限")
        if self.operation == "limit" and self.point is None:
            raise ValueError("求极限必须提供 point")
        return self


_SYMBOLS = {name: sp.Symbol(name) for name in ("x", "y", "z", "t", "n")}
_CONSTANTS = {
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "oo": sp.oo,
    "inf": sp.oo,
}
_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}
_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}


class _SafeSympyParser(ast.NodeVisitor):
    def __init__(self, *, max_nodes: int = 80):
        self.max_nodes = max_nodes
        self.node_count = 0

    def visit(self, node):
        self.node_count += 1
        if self.node_count > self.max_nodes:
            raise ToolInputError("数学表达式过于复杂")
        return super().visit(node)

    def generic_visit(self, node):
        raise ToolInputError(f"不支持的表达式结构: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolInputError("只允许数值常量")
        if abs(float(node.value)) > 1e12:
            raise ToolInputError("数值常量过大")
        return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)

    def visit_Name(self, node: ast.Name):
        if node.id in _SYMBOLS:
            return _SYMBOLS[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolInputError(f"不允许的名称: {node.id}")

    def visit_BinOp(self, node: ast.BinOp):
        operator = _BINARY_OPERATORS.get(type(node.op))
        if operator is None:
            raise ToolInputError(f"不支持的运算符: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Pow) and right.is_number:
            try:
                if abs(float(right)) > 100:
                    raise ToolInputError("指数绝对值不能超过 100")
            except (TypeError, ValueError):
                raise ToolInputError("指数必须是可计算的数值") from None
        return operator(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ToolInputError(f"不支持的一元运算符: {type(node.op).__name__}")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ToolInputError("只允许白名单数学函数")
        if node.keywords or not 1 <= len(node.args) <= 2:
            raise ToolInputError("数学函数参数数量不正确")
        arguments = [self.visit(argument) for argument in node.args]
        return _FUNCTIONS[node.func.id](*arguments)


def parse_safe_expression(expression: str):
    normalized = expression.strip().replace("^", "**")
    if "__" in normalized:
        raise ToolInputError("表达式包含禁止内容")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ToolInputError("数学表达式语法不正确") from exc
    return _SafeSympyParser().visit(tree)


def calculate_math(arguments: CalculatorToolArguments) -> dict:
    expression = parse_safe_expression(arguments.expression)
    variable = _SYMBOLS[arguments.variable]

    try:
        if arguments.operation == "simplify":
            result = sp.simplify(expression)
        elif arguments.operation == "differentiate":
            result = sp.diff(expression, variable)
        elif arguments.operation == "integrate":
            if arguments.lower_bound is None:
                result = sp.integrate(expression, variable)
            else:
                lower = parse_safe_expression(arguments.lower_bound)
                upper = parse_safe_expression(arguments.upper_bound or "")
                result = sp.integrate(expression, (variable, lower, upper))
        elif arguments.operation == "limit":
            point = parse_safe_expression(arguments.point or "")
            direction = arguments.direction if arguments.direction in {"+", "-"} else "+-"
            result = sp.limit(expression, variable, point, dir=direction)
        elif arguments.operation == "solve":
            result = sp.solve(expression, variable)
        else:
            result = sp.N(expression, 12)
    except (ArithmeticError, TypeError, ValueError, NotImplementedError) as exc:
        raise ToolInputError(f"数学计算失败: {exc}") from exc

    rendered_result = str(result)
    rendered_latex = sp.latex(result)
    if len(rendered_result) > 2000 or len(rendered_latex) > 4000:
        raise ToolInputError("数学计算结果过长")

    return {
        "operation": arguments.operation,
        "expression": arguments.expression,
        "variable": arguments.variable,
        "lower_bound": arguments.lower_bound,
        "upper_bound": arguments.upper_bound,
        "point": arguments.point,
        "direction": arguments.direction,
        "result": rendered_result,
        "latex": rendered_latex,
    }


def _calculation_worker(payload: dict, result_queue) -> None:
    try:
        arguments = CalculatorToolArguments.model_validate(payload)
        result_queue.put({"ok": True, "result": calculate_math(arguments)})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)[:500]})


def calculate_math_isolated(
    arguments: CalculatorToolArguments,
    *,
    timeout_seconds: float = 4.0,
) -> dict:
    """Run symbolic work in a child process that can be terminated on timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_calculation_worker,
        args=(arguments.model_dump(), result_queue),
        daemon=True,
    )
    try:
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            raise ToolInputError("数学计算超时")
        try:
            payload = result_queue.get(timeout=0.5)
        except Empty as exc:
            raise ToolInputError("数学计算进程异常退出") from exc
        if not payload.get("ok"):
            raise ToolInputError(payload.get("error") or "数学计算失败")
        return payload["result"]
    finally:
        result_queue.close()
        result_queue.join_thread()
        if process.pid is not None and not process.is_alive():
            process.close()
