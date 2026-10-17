"""Bounded exhaustive audits of expression canonicalizers.

Enumeration preserves ordered unary/binary trees: no normalization is applied
before the method under test. Equal oracle keys must be sufficient evidence of
identity, while different oracle keys may simply mean an unrecognized identity.
SymPy is an optional dependency of the exact-expression adapter, not enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Hashable, Iterator, Mapping, Sequence

from tsfmxai.representations.expression import (
    Add, ExpressionNode, Lag, Multiply, Parameter, Tanh, Time, operator_count,
)


def enumerate_ordered_trees(
    leaves: Sequence[ExpressionNode],
    *,
    max_operators: int,
    binary: Sequence[type] = (Add, Multiply),
    unary: Sequence[type] = (Tanh,),
) -> Iterator[ExpressionNode]:
    """Enumerate each ordered tree once, in increasing operator count.

    The finite vocabulary and size bound define the exhaustive search space.
    In a binary root, every split of the remaining operator budget is visited.
    Leaves are actual terminals, never prebuilt composite expressions.
    """
    leaves, binary, unary = tuple(leaves), tuple(binary), tuple(unary)
    if max_operators < 0:
        raise ValueError("max_operators must be nonnegative")
    if not leaves or len(set(leaves)) != len(leaves):
        raise ValueError("Provide distinct, nonempty leaves")
    if any(not isinstance(leaf, (Lag, Parameter, Time)) for leaf in leaves):
        raise ValueError("Leaves must be terminal AST nodes")
    if len(set(binary)) != len(binary) or any(op not in (Add, Multiply) for op in binary):
        raise ValueError("Binary operators must be distinct Add/Multiply constructors")
    if len(set(unary)) != len(unary) or any(op is not Tanh for op in unary):
        raise ValueError("Unary operators must be distinct Tanh constructors")

    @lru_cache(maxsize=None)
    def layer(size: int) -> tuple[ExpressionNode, ...]:
        if size == 0:
            return leaves
        result = [op(child) for op in unary for child in layer(size - 1)]
        for op in binary:
            for left_size in range(size):
                for left in layer(left_size):
                    for right in layer(size - 1 - left_size):
                        result.append(op(left, right))
        return tuple(result)

    for size in range(max_operators + 1):
        yield from layer(size)


def exact_expression(
    expression: ExpressionNode,
    *,
    parameter_values: Mapping[str, int | str | Fraction] | None = None,
    lag_values: Mapping[str, int] | None = None,
    analytic_tanh: bool = False,
):
    """Translate to independent, exact SymPy algebra over real inputs.

    Without bindings each named slot is a formal atom. With bindings, equal
    variable/lag pairs denote the same input and supplied constants are exact
    rationals. Float bindings are rejected to make arithmetic semantics explicit.
    Opaque tanh tests ring identities only. Analytic tanh additionally permits
    identities SymPy recognizes (e.g. oddness); this is a sufficient oracle,
    not a complete decision procedure for all transcendental identities.
    """
    import sympy as sp

    parameters = parameter_values or {}
    lags = lag_values or {}

    def visit(node):
        if isinstance(node, Parameter):
            if node.slot not in parameters:
                return sp.Symbol(f"parameter:{node.slot}", real=True)
            value = parameters[node.slot]
            if not isinstance(value, (int, str, Fraction)) or isinstance(value, bool):
                raise TypeError("Use int, rational string or Fraction for exact constants")
            return sp.Rational(str(value))
        if isinstance(node, Lag):
            if node.slot in lags:
                lag = lags[node.slot]
                if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
                    raise ValueError("Bound lags must be positive integers")
                suffix = f"delay:{lag}"
            else:
                suffix = f"slot:{node.slot}"
            return sp.Symbol(f"lag:{node.variable}:{suffix}", real=True)
        if isinstance(node, Time):
            return sp.Symbol("time", real=True)
        if isinstance(node, Tanh):
            argument = sp.expand(visit(node.operand))
            return sp.tanh(argument) if analytic_tanh else sp.Function("Tanh_opaque")(argument)
        if isinstance(node, Add):
            return visit(node.left) + visit(node.right)
        if isinstance(node, Multiply):
            return visit(node.left) * visit(node.right)
        raise TypeError(type(node))

    return sp.expand(visit(expression))


@dataclass(frozen=True)
class Counterexample:
    """Two distinct ordered ASTs with one exact identity key and two method keys."""

    left: ExpressionNode
    right: ExpressionNode
    oracle_key: Hashable
    left_key: Hashable
    right_key: Hashable
    visited_at_discovery: int
    tested_at_discovery: int
    operator_layer: int


@dataclass
class AuditResult:
    counterexamples: dict[str, Counterexample]
    tested_by_method: dict[str, int]
    errors_by_method: dict[str, int]
    first_errors: dict[str, str]
    visited: int
    eligible: int
    visited_by_layer: dict[int, int]
    stop_reason: str


def search_first_counterexamples(
    trees: Iterator[ExpressionNode],
    *,
    methods: Mapping[str, Callable[[ExpressionNode], Hashable]],
    oracle: Callable[[ExpressionNode], Hashable],
    eligible: Callable[[ExpressionNode], bool] = lambda tree: True,
) -> AuditResult:
    """Stop each method at its first witnessed failure of uniqueness.

    A dictionary indexes actual oracle keys (not digests or sampled values),
    avoiding quadratic all-pairs comparison. Continue unresolved methods until
    the finite iterator is exhausted. Exceptions are reported separately and
    never interpreted as counterexamples or successful canonicalizations.
    """
    if not methods:
        raise ValueError("At least one method is required")
    result = AuditResult({}, dict.fromkeys(methods, 0), dict.fromkeys(methods, 0),
                         {}, 0, 0, {}, "size_bound_exhausted")
    seen: dict[str, dict[Hashable, tuple[ExpressionNode, Hashable]]] = {
        name: {} for name in methods
    }
    for tree in trees:
        result.visited += 1
        size = operator_count(tree)
        result.visited_by_layer[size] = result.visited_by_layer.get(size, 0) + 1
        if not eligible(tree):
            continue
        result.eligible += 1
        identity = oracle(tree)
        for name, method in methods.items():
            if name in result.counterexamples:
                continue
            result.tested_by_method[name] += 1
            try:
                key = method(tree)
                hash(key)
            except Exception as error:
                result.errors_by_method[name] += 1
                result.first_errors.setdefault(name, f"{type(error).__name__}: {error}")
                continue
            previous = seen[name].get(identity)
            if previous is None:
                seen[name][identity] = (tree, key)
            elif previous[1] != key:
                result.counterexamples[name] = Counterexample(
                    previous[0], tree, identity, previous[1], key,
                    result.visited, result.eligible, size,
                )
                seen[name].clear()
        if len(result.counterexamples) == len(methods):
            result.stop_reason = "all_methods_found"
            break
    return result
