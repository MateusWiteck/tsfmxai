"""Canonical expression nodes for discrete-time symbolic mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, TypeAlias


class Expression:
    """Base class for nodes in a symbolic mechanism."""


@dataclass(frozen=True)
class Parameter(Expression):
    """Continuous parameter slot whose value is stored outside the AST."""

    slot: str

    def __post_init__(self) -> None:
        if not self.slot:
            raise ValueError("A parameter slot must have a non-empty name.")


@dataclass(frozen=True)
class Time(Expression):
    """Current discrete-time index."""


@dataclass(frozen=True)
class Lag(Expression):
    """Reference to a variable at a lag supplied through a discrete slot."""

    variable: int
    slot: str

    def __post_init__(self) -> None:
        if self.variable < 0:
            raise ValueError("A lag variable index must be non-negative.")
        if not self.slot:
            raise ValueError("A lag slot must have a non-empty name.")


@dataclass(frozen=True)
class Add(Expression):
    """Binary addition."""

    left: Expression
    right: Expression


@dataclass(frozen=True)
class Multiply(Expression):
    """Binary multiplication."""

    left: Expression
    right: Expression


ExpressionNode: TypeAlias = Parameter | Time | Lag | Add | Multiply


@dataclass(frozen=True)
class MechanismBinding:
    """Numeric values bound to the continuous and discrete AST slots."""

    parameter_values: Mapping[str, float]
    lag_values: Mapping[str, int]


def children(expression: ExpressionNode) -> tuple[ExpressionNode, ...]:
    """Return direct children in display and evaluation order."""

    if isinstance(expression, (Add, Multiply)):
        return expression.left, expression.right
    return ()


def walk(expression: ExpressionNode) -> Iterator[ExpressionNode]:
    """Traverse an expression in pre-order."""

    yield expression
    for child in children(expression):
        yield from walk(child)


def node_count(expression: ExpressionNode) -> int:
    """Count every operator and terminal node."""

    return sum(1 for _ in walk(expression))


def operator_count(expression: ExpressionNode) -> int:
    """Count binary operator nodes."""

    return sum(isinstance(node, (Add, Multiply)) for node in walk(expression))


def leaf_count(expression: ExpressionNode) -> int:
    """Count parameter, time, and lag terminals."""

    return sum(isinstance(node, (Parameter, Time, Lag)) for node in walk(expression))


def tree_depth(expression: ExpressionNode) -> int:
    """Return tree depth in nodes, assigning depth one to a terminal."""

    direct_children = children(expression)
    if not direct_children:
        return 1
    return 1 + max(tree_depth(child) for child in direct_children)


def parameter_slots(expression: ExpressionNode) -> tuple[str, ...]:
    """Return parameter slots in pre-order without duplicates."""

    slots = (node.slot for node in walk(expression) if isinstance(node, Parameter))
    return tuple(dict.fromkeys(slots))


def lag_slots(expression: ExpressionNode) -> tuple[str, ...]:
    """Return lag slots in pre-order without duplicates."""

    return tuple(dict.fromkeys(node.slot for node in walk(expression) if isinstance(node, Lag)))


def _structural_key(expression: ExpressionNode) -> str:
    """Build a slot-name-independent key used to order commutative children."""

    if isinstance(expression, Parameter):
        return "PARAMETER"
    if isinstance(expression, Time):
        return "TIME"
    if isinstance(expression, Lag):
        return f"LAG[{expression.variable}]"
    if isinstance(expression, Add):
        return f"ADD({_structural_key(expression.left)},{_structural_key(expression.right)})"
    if isinstance(expression, Multiply):
        return f"MULTIPLY({_structural_key(expression.left)},{_structural_key(expression.right)})"
    raise TypeError(type(expression))


def canonicalize_commutative(expression: ExpressionNode) -> ExpressionNode:
    """Recursively order the children of addition and multiplication."""

    if isinstance(expression, (Parameter, Time, Lag)):
        return expression

    left = canonicalize_commutative(expression.left)
    right = canonicalize_commutative(expression.right)
    if _structural_key(right) < _structural_key(left):
        left, right = right, left
    if isinstance(expression, Add):
        return Add(left, right)
    return Multiply(left, right)


def relabel_slots(expression: ExpressionNode) -> ExpressionNode:
    """Assign deterministic P0... and L0... names in pre-order."""

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
        left = visit(node.left)
        right = visit(node.right)
        if isinstance(node, Add):
            return Add(left, right)
        return Multiply(left, right)

    return visit(expression)


def canonicalize_and_relabel(expression: ExpressionNode) -> ExpressionNode:
    """Canonicalize commutative nodes and then assign deterministic slot names."""

    return relabel_slots(canonicalize_commutative(expression))


def node_label(expression: ExpressionNode) -> str:
    """Return the concise label used in tree diagrams."""

    if isinstance(expression, Parameter):
        return f"PARAMETER({expression.slot})"
    if isinstance(expression, Time):
        return "TIME"
    if isinstance(expression, Lag):
        return f"LAG(x{expression.variable}, {expression.slot})"
    if isinstance(expression, Add):
        return "ADD"
    if isinstance(expression, Multiply):
        return "MULTIPLY"
    raise TypeError(type(expression))


def format_tree(expression: ExpressionNode) -> str:
    """Format the AST as a readable Unicode tree."""

    def visit(node: ExpressionNode, prefix: str, is_last: bool) -> list[str]:
        connector = "└── " if is_last else "├── "
        lines = [prefix + connector + node_label(node)]
        direct_children = children(node)
        child_prefix = prefix + ("    " if is_last else "│   ")
        for index, child in enumerate(direct_children):
            lines.extend(visit(child, child_prefix, index == len(direct_children) - 1))
        return lines

    lines = visit(expression, "", True)
    return "\n".join([lines[0].removeprefix("└── "), *lines[1:]])


def to_skeleton_formula(expression: ExpressionNode) -> str:
    """Serialize the structure while retaining parameter and lag placeholders."""

    if isinstance(expression, Parameter):
        return expression.slot
    if isinstance(expression, Time):
        return "t"
    if isinstance(expression, Lag):
        return f"LAG({expression.variable}, {expression.slot})"
    if isinstance(expression, Add):
        return f"({to_skeleton_formula(expression.left)} + {to_skeleton_formula(expression.right)})"
    if isinstance(expression, Multiply):
        return (
            f"({to_skeleton_formula(expression.left)} * "
            f"{to_skeleton_formula(expression.right)})"
        )
    raise TypeError(type(expression))


def validate_binding(expression: ExpressionNode, binding: MechanismBinding) -> None:
    """Verify that a binding covers exactly the slots present in an expression."""

    expected_parameters = set(parameter_slots(expression))
    expected_lags = set(lag_slots(expression))
    supplied_parameters = set(binding.parameter_values)
    supplied_lags = set(binding.lag_values)
    if supplied_parameters != expected_parameters:
        raise ValueError(
            "Parameter binding mismatch: "
            f"expected {sorted(expected_parameters)}, received {sorted(supplied_parameters)}."
        )
    if supplied_lags != expected_lags:
        raise ValueError(
            f"Lag binding mismatch: expected {sorted(expected_lags)}, "
            f"received {sorted(supplied_lags)}."
        )
    if any(type(lag) is not int or lag < 1 for lag in binding.lag_values.values()):
        raise ValueError("Every bound lag must be an integer greater than or equal to one.")


def to_bound_formula(expression: ExpressionNode, binding: MechanismBinding) -> str:
    """Render a mathematical formula after binding coefficients and lags."""

    validate_binding(expression, binding)

    def visit(node: ExpressionNode) -> str:
        if isinstance(node, Parameter):
            return f"{float(binding.parameter_values[node.slot]):.6g}"
        if isinstance(node, Time):
            return "t"
        if isinstance(node, Lag):
            lag = binding.lag_values[node.slot]
            return f"x_{node.variable}(t-{lag})"
        if isinstance(node, Add):
            return f"({visit(node.left)} + {visit(node.right)})"
        if isinstance(node, Multiply):
            return f"({visit(node.left)} * {visit(node.right)})"
        raise TypeError(type(node))

    return visit(expression)
