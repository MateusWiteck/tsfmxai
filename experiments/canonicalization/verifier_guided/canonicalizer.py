"""Verifier-guided candidate canonicalization for scalar discrete recurrences.

The referenced verifier-guided paper states that symbolic canonicalization
removes duplicates before candidates are pooled, but does not prescribe that
canonicalizer.  This experiment therefore uses an AC structural key, then adds
the paper's central idea: deduplicate/rank using executable behavior over
multiple initial conditions.  Behavioral signatures are empirical, not proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from typing import Iterable, Mapping, Sequence

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    Multiply,
    Parameter,
    Tanh,
    Time,
)


METHOD_NAME = "Verifier-Guided-inspired structural + rollout deduplication"


def _flatten(expression: ExpressionNode, node_type: type) -> list[ExpressionNode]:
    if isinstance(expression, node_type):
        return _flatten(expression.left, node_type) + _flatten(expression.right, node_type)
    return [expression]


def canonical_key(expression: ExpressionNode) -> str:
    """Return an associative-commutative structural key.

    Distributivity and analytic identities are deliberately not applied here.
    """

    if isinstance(expression, Parameter):
        return f"PARAMETER[{expression.slot}]"
    if isinstance(expression, Time):
        return "TIME"
    if isinstance(expression, Lag):
        return f"LAG[{expression.variable},{expression.slot}]"
    if isinstance(expression, Tanh):
        return f"TANH({canonical_key(expression.operand)})"
    if isinstance(expression, Add):
        items = sorted(canonical_key(item) for item in _flatten(expression, Add))
        return "ADD(" + ",".join(items) + ")"
    if isinstance(expression, Multiply):
        items = sorted(canonical_key(item) for item in _flatten(expression, Multiply))
        return "MUL(" + ",".join(items) + ")"
    raise TypeError(type(expression))


def canonical_structure(expression: ExpressionNode) -> tuple[str, tuple[object, ...]]:
    """Return the n-ary AC tree corresponding to :func:`canonical_key`."""

    if isinstance(expression, Parameter):
        return f"PARAMETER\n{expression.slot}", ()
    if isinstance(expression, Time):
        return "TIME", ()
    if isinstance(expression, Lag):
        return f"LAG x{expression.variable}\n{expression.slot}", ()
    if isinstance(expression, Tanh):
        return "TANH", (canonical_structure(expression.operand),)
    if isinstance(expression, Add):
        items = sorted(_flatten(expression, Add), key=canonical_key)
        return "ADD", tuple(canonical_structure(item) for item in items)
    if isinstance(expression, Multiply):
        items = sorted(_flatten(expression, Multiply), key=canonical_key)
        return "MUL", tuple(canonical_structure(item) for item in items)
    raise TypeError(type(expression))


def _evaluate(
    expression: ExpressionNode,
    *,
    step: int,
    history: Sequence[float],
    parameter_values: Mapping[str, float],
    lag_values: Mapping[str, int],
) -> float:
    if isinstance(expression, Parameter):
        return float(parameter_values[expression.slot])
    if isinstance(expression, Time):
        return float(step)
    if isinstance(expression, Lag):
        if expression.variable != 0:
            raise ValueError("Rollout verification currently supports only Lag(variable=0, ...).")
        lag = int(lag_values[expression.slot])
        if lag < 1 or lag > len(history):
            raise ValueError(f"Lag {lag} is unavailable for history length {len(history)}.")
        return float(history[-lag])
    if isinstance(expression, Tanh):
        return tanh(
            _evaluate(
                expression.operand,
                step=step,
                history=history,
                parameter_values=parameter_values,
                lag_values=lag_values,
            )
        )
    left = _evaluate(
        expression.left,
        step=step,
        history=history,
        parameter_values=parameter_values,
        lag_values=lag_values,
    )
    right = _evaluate(
        expression.right,
        step=step,
        history=history,
        parameter_values=parameter_values,
        lag_values=lag_values,
    )
    return left + right if isinstance(expression, Add) else left * right


def rollout_signature(
    expression: ExpressionNode,
    *,
    parameter_values: Mapping[str, float],
    lag_values: Mapping[str, int],
    initial_conditions: Iterable[Sequence[float]],
    steps: int = 12,
    decimals: int = 10,
    absolute_bound: float = 1e12,
) -> tuple[object, ...]:
    """Simulate several rollouts and return a rounded empirical signature."""

    if steps < 1:
        raise ValueError("steps must be positive")
    signature: list[object] = []
    for initial in initial_conditions:
        history = [float(value) for value in initial]
        if not history:
            raise ValueError("Each initial condition must contain at least one value.")
        for offset in range(steps):
            try:
                value = _evaluate(
                    expression,
                    step=len(initial) + offset,
                    history=history,
                    parameter_values=parameter_values,
                    lag_values=lag_values,
                )
            except (ArithmeticError, OverflowError, ValueError):
                signature.append("INVALID")
                break
            if not isfinite(value) or abs(value) > absolute_bound:
                signature.append("NONFINITE")
                break
            history.append(value)
            signature.append(round(value, decimals))
        signature.append("|")
    return tuple(signature)


@dataclass(frozen=True)
class Candidate:
    """One executable candidate and the binding required by its recurrence."""

    label: str
    expression: ExpressionNode
    parameter_values: Mapping[str, float]
    lag_values: Mapping[str, int]


@dataclass(frozen=True)
class DeduplicationResult:
    """Two-stage symbolic and empirical deduplication result."""

    symbolic_representatives: tuple[Candidate, ...]
    behavioral_representatives: tuple[Candidate, ...]
    symbolic_groups: Mapping[str, tuple[str, ...]]
    behavioral_groups: Mapping[tuple[object, ...], tuple[str, ...]]


def deduplicate_candidates(
    candidates: Sequence[Candidate],
    *,
    initial_conditions: Iterable[Sequence[float]],
    steps: int = 12,
    decimals: int = 10,
) -> DeduplicationResult:
    """Pool by symbolic key, then pool remaining candidates by rollout signature."""

    initials = tuple(tuple(values) for values in initial_conditions)
    symbolic: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        symbolic.setdefault(canonical_key(candidate.expression), []).append(candidate)
    symbolic_representatives = tuple(group[0] for group in symbolic.values())

    behavioral: dict[tuple[object, ...], list[Candidate]] = {}
    for candidate in symbolic_representatives:
        signature = rollout_signature(
            candidate.expression,
            parameter_values=candidate.parameter_values,
            lag_values=candidate.lag_values,
            initial_conditions=initials,
            steps=steps,
            decimals=decimals,
        )
        if "INVALID" in signature or "NONFINITE" in signature:
            # A failed rollout provides no evidence of behavioral equivalence.
            # Keep such candidates separate using their already canonical AC key.
            signature = ("UNVERIFIED", canonical_key(candidate.expression), *signature)
        behavioral.setdefault(signature, []).append(candidate)

    return DeduplicationResult(
        symbolic_representatives=symbolic_representatives,
        behavioral_representatives=tuple(group[0] for group in behavioral.values()),
        symbolic_groups={key: tuple(item.label for item in group) for key, group in symbolic.items()},
        behavioral_groups={key: tuple(item.label for item in group) for key, group in behavioral.items()},
    )
