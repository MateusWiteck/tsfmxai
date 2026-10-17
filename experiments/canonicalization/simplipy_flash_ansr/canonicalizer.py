"""Restricted SimpliPy-style normalizer for the project's current AST vocabulary.

This is an experiment-local adaptation, not the official SimpliPy package.  It
implements the part most relevant to this vocabulary: deterministic
associative-commutative bags plus a semiring polynomial normal form.  ``Tanh``
is an opaque function whose argument is recursively normalized.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    Multiply,
    Parameter,
    Tanh,
    Time,
)


METHOD_NAME = "SimpliPy/Flash-ANSR-inspired restricted normal form"


class ExpansionLimitError(RuntimeError):
    """Raised before distributive expansion exceeds the configured term cap."""


@dataclass(frozen=True, order=True)
class Atom:
    """An indivisible factor in the restricted polynomial normal form."""

    kind: str
    payload: tuple[object, ...] = ()


Monomial = tuple[Atom, ...]
Polynomial = Counter[Monomial]


def _estimated_product_terms(left: Polynomial, right: Polynomial) -> int:
    return sum(left.values()) * sum(right.values())


def _polynomial(expression: ExpressionNode, *, max_terms: int) -> Polynomial:
    if isinstance(expression, Parameter):
        return Counter({(Atom("PARAMETER", (expression.slot,)),): 1})
    if isinstance(expression, Time):
        return Counter({(Atom("TIME"),): 1})
    if isinstance(expression, Lag):
        return Counter({(Atom("LAG", (expression.variable, expression.slot)),): 1})
    if isinstance(expression, Tanh):
        inner = canonical_key(expression.operand, max_terms=max_terms)
        return Counter({(Atom("TANH", (inner,)),): 1})
    if isinstance(expression, Add):
        result = _polynomial(expression.left, max_terms=max_terms)
        result.update(_polynomial(expression.right, max_terms=max_terms))
        if sum(result.values()) > max_terms:
            raise ExpansionLimitError(f"normal form would exceed {max_terms} additive terms")
        return result
    if isinstance(expression, Multiply):
        left = _polynomial(expression.left, max_terms=max_terms)
        right = _polynomial(expression.right, max_terms=max_terms)
        if _estimated_product_terms(left, right) > max_terms:
            raise ExpansionLimitError(f"normal form would exceed {max_terms} additive terms")
        result: Polynomial = Counter()
        for left_monomial, left_count in left.items():
            for right_monomial, right_count in right.items():
                monomial = tuple(sorted(left_monomial + right_monomial))
                result[monomial] += left_count * right_count
        return result
    raise TypeError(type(expression))


def _atom_key(atom: Atom) -> str:
    if atom.kind == "TIME":
        return "TIME"
    if atom.kind == "PARAMETER":
        return f"PARAMETER[{atom.payload[0]}]"
    if atom.kind == "LAG":
        return f"LAG[{atom.payload[0]},{atom.payload[1]}]"
    if atom.kind == "TANH":
        return f"TANH[{atom.payload[0]}]"
    raise ValueError(atom.kind)


def _render(polynomial: Polynomial) -> str:
    terms: list[str] = []
    for monomial in sorted(polynomial):
        rendered = "MUL(" + ",".join(_atom_key(atom) for atom in monomial) + ")"
        terms.extend([rendered] * polynomial[monomial])
    return "ADD(" + ",".join(terms) + ")"


def canonical_key(expression: ExpressionNode, *, max_terms: int = 4096) -> str:
    """Return a unique key for the supported semiring fragment.

    Slot names are semantic atoms in this method.  The key is invariant to
    associativity, commutativity, and distributivity of ``Add``/``Multiply``.
    It is not a complete identity system for ``tanh`` or floating bindings.
    """

    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    return _render(_polynomial(expression, max_terms=max_terms))


def canonical_structure(
    expression: ExpressionNode,
    *,
    max_terms: int = 4096,
) -> tuple[str, tuple[object, ...]]:
    """Return the canonical n-ary tree used by notebook visualizations."""

    polynomial = _polynomial(expression, max_terms=max_terms)

    def atom_node(atom: Atom) -> tuple[str, tuple[object, ...]]:
        if atom.kind == "TIME":
            return "TIME", ()
        if atom.kind == "PARAMETER":
            return f"PARAMETER\n{atom.payload[0]}", ()
        if atom.kind == "LAG":
            return f"LAG x{atom.payload[0]}\n{atom.payload[1]}", ()
        if atom.kind == "TANH":
            return "TANH", ((str(atom.payload[0]), ()),)
        raise ValueError(atom.kind)

    terms: list[tuple[str, tuple[object, ...]]] = []
    for monomial in sorted(polynomial):
        term = ("MUL", tuple(atom_node(atom) for atom in monomial))
        terms.extend([term] * polynomial[monomial])
    return "ADD", tuple(terms)


def canonicalize(expression: ExpressionNode, *, max_terms: int = 4096) -> str:
    """Alias emphasizing that this method's canonical object is a key."""

    return canonical_key(expression, max_terms=max_terms)
