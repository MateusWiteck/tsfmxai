"""Binary Local Commutative Canonicalization (CCBL) experiment snapshot."""

from __future__ import annotations

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    Multiply,
    Parameter,
    Tanh,
    Time,
    to_skeleton_formula,
)


METHOD_NAME = "CCBL: binary local commutativity + preorder slot relabeling"


def _structural_key(expression: ExpressionNode) -> str:
    if isinstance(expression, Parameter):
        return "PARAMETER"
    if isinstance(expression, Time):
        return "TIME"
    if isinstance(expression, Lag):
        return f"LAG[{expression.variable}]"
    if isinstance(expression, Tanh):
        return f"TANH({_structural_key(expression.operand)})"
    if isinstance(expression, Add):
        return f"ADD({_structural_key(expression.left)},{_structural_key(expression.right)})"
    if isinstance(expression, Multiply):
        return f"MULTIPLY({_structural_key(expression.left)},{_structural_key(expression.right)})"
    raise TypeError(type(expression))


def _canonicalize_commutative(expression: ExpressionNode) -> ExpressionNode:
    if isinstance(expression, (Parameter, Time, Lag)):
        return expression
    if isinstance(expression, Tanh):
        return Tanh(_canonicalize_commutative(expression.operand))
    left = _canonicalize_commutative(expression.left)
    right = _canonicalize_commutative(expression.right)
    if _structural_key(right) < _structural_key(left):
        left, right = right, left
    return Add(left, right) if isinstance(expression, Add) else Multiply(left, right)


def _relabel_slots(expression: ExpressionNode) -> ExpressionNode:
    parameter_index = 0
    lag_index = 0

    def visit(node: ExpressionNode) -> ExpressionNode:
        nonlocal parameter_index, lag_index
        if isinstance(node, Parameter):
            result = Parameter(f"P{parameter_index}")
            parameter_index += 1
            return result
        if isinstance(node, Lag):
            result = Lag(node.variable, f"L{lag_index}")
            lag_index += 1
            return result
        if isinstance(node, Time):
            return node
        if isinstance(node, Tanh):
            return Tanh(visit(node.operand))
        left = visit(node.left)
        right = visit(node.right)
        return Add(left, right) if isinstance(node, Add) else Multiply(left, right)

    return visit(expression)


def canonicalize(expression: ExpressionNode) -> ExpressionNode:
    """Return the AST produced by CCBL."""

    return _relabel_slots(_canonicalize_commutative(expression))


def canonical_key(expression: ExpressionNode) -> str:
    """Return the serialized CCBL tree."""

    return to_skeleton_formula(canonicalize(expression))
