"""Exact sparse normal form over RAW and TANH lag factors.

Lag atoms use resolved (series, lag, transformation), not AST slot names.
Parameter products are retained symbolically and evaluated exactly from the
binary floating-point binding using ``Fraction.from_float``.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import isfinite, tanh

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Tanh,
    validate_binding,
)

LagAtom = tuple[int, int] | tuple[int, int, str]
Monomial = tuple[LagAtom, ...]
ParameterMonomial = tuple[str, ...]
SymbolicCoefficient = tuple[tuple[ParameterMonomial, int], ...]


def monomial_order(key: Monomial) -> tuple[int, Monomial]:
    """Order by degree, then series, lag, and RAW before TANH."""

    return len(key), key


@dataclass(frozen=True)
class CanonicalTerm:
    monomial: Monomial
    coefficient_exact: Fraction
    symbolic_coefficient: SymbolicCoefficient

    @property
    def coefficient(self) -> float:
        """Convert for numerical targets, rejecting overflow or loss to zero."""

        try:
            value = float(self.coefficient_exact)
        except OverflowError as error:
            raise ValueError("Coefficient overflows float; use coefficient_exact.") from error
        if not isfinite(value) or (value == 0 and self.coefficient_exact != 0):
            raise ValueError("Coefficient underflows/overflows float; use coefficient_exact.")
        return value


@dataclass(frozen=True)
class CanonicalPolynomial:
    """Sparse, ordered realized terms plus support before numeric binding."""

    terms: tuple[CanonicalTerm, ...]
    potential_skeleton: tuple[Monomial, ...]
    max_degree: int

    @property
    def skeleton(self) -> tuple[Monomial, ...]:
        """Realized support after substituting bound parameter values."""

        return tuple(term.monomial for term in self.terms)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return tuple(term.coefficient for term in self.terms)

    @property
    def canonical_key(self) -> tuple[tuple[Monomial, Fraction], ...]:
        """Function identity, independent of parameter names and provenance.

        Dataclass equality includes symbolic provenance and configuration.
        Compare this key when checking equality of realized polynomials.
        """

        return tuple((term.monomial, term.coefficient_exact) for term in self.terms)

    def evaluate(
        self, histories: Mapping[int, Sequence[float]], *, time: int | None = None
    ) -> float:
        """Evaluate at a shared time index (default: equal-length history ends).

        The indexing agrees with ``evaluate_bound_expression``. Histories
        contain observations up to time-1; explicit time permits longer arrays.
        """

        sources = {atom[0] for key in self.skeleton for atom in key}
        if time is None:
            try:
                lengths = {len(histories[series]) for series in sources}
            except KeyError as error:
                raise ValueError(f"Missing history for series {error.args[0]}.") from error
            if len(lengths) > 1:
                raise ValueError("Histories must share a length, or supply an explicit time.")
            time = next(iter(lengths), 0)
        if type(time) is not int or time < 0:
            raise ValueError("time must be a non-negative integer.")

        total = 0.0
        for term in self.terms:
            value = term.coefficient
            for atom in term.monomial:
                series, lag = atom[:2]
                if series not in histories or time < lag or time > len(histories[series]):
                    raise ValueError(f"History for s_{series}[t-{lag}] is unavailable at t={time}.")
                factor = histories[series][time - lag]
                value *= tanh(factor) if len(atom) == 3 else factor
            total += value
        return total


def _add_coefficients(
    left: dict[ParameterMonomial, int], right: dict[ParameterMonomial, int]
) -> dict[ParameterMonomial, int]:
    result = left.copy()
    for key, value in right.items():
        result[key] = result.get(key, 0) + value
        if result[key] == 0:
            del result[key]
    return result


def _add_polynomials(
    left: dict[Monomial, dict[ParameterMonomial, int]],
    right: dict[Monomial, dict[ParameterMonomial, int]],
    max_terms: int,
) -> dict[Monomial, dict[ParameterMonomial, int]]:
    result = {key: coefficient.copy() for key, coefficient in left.items()}
    for key, coefficient in right.items():
        combined = _add_coefficients(result.get(key, {}), coefficient)
        if combined:
            result[key] = combined
        else:
            result.pop(key, None)
    if len(result) > max_terms:
        raise ValueError(f"Expanded polynomial exceeds max_terms={max_terms}.")
    return result


def canonicalize_polynomial(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    max_degree: int = 4,
    max_terms: int = 10_000,
    max_coefficient_terms: int = 10_000,
) -> CanonicalPolynomial:
    """Distribute products, combine equal lag monomials, and sort sparse terms.

    ``Tanh(Lag)`` is a distinct bounded factor; other tanh expressions are
    outside this subgrammar. A product
    exceeding ``max_degree`` raises before numeric cancellations, including
    zero-bound coefficients. The degree limit defines the supported input
    expansion, not just the degree of the final numerically simplified result.
    ``max_coefficient_terms`` also bounds parameter monomials per coefficient.
    Coefficients use exact arithmetic for the supplied binary float values;
    no numerical zero tolerance is applied.
    """

    if type(max_degree) is not int or max_degree < 0:
        raise ValueError("max_degree must be a non-negative integer.")
    for name, value in (("max_terms", max_terms), ("max_coefficient_terms", max_coefficient_terms)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    validate_binding(expression, binding)
    if not all(isfinite(float(value)) for value in binding.parameter_values.values()):
        raise ValueError("Every parameter value must be finite.")

    def expand(node: ExpressionNode) -> dict[Monomial, dict[ParameterMonomial, int]]:
        if isinstance(node, Parameter):
            return {(): {(node.slot,): 1}}
        if isinstance(node, Lag):
            if type(node.variable) is not int or node.variable < 0:
                raise ValueError("Series IDs must be non-negative integers.")
            if max_degree < 1:
                raise ValueError("A lag term exceeds max_degree=0.")
            return {((node.variable, binding.lag_values[node.slot]),): {(): 1}}
        if isinstance(node, Add):
            result = _add_polynomials(expand(node.left), expand(node.right), max_terms)
            if any(len(coefficient) > max_coefficient_terms for coefficient in result.values()):
                raise ValueError(f"Symbolic coefficient exceeds max_coefficient_terms={max_coefficient_terms}.")
            return result
        if isinstance(node, Multiply):
            left, right = expand(node.left), expand(node.right)
            result: dict[Monomial, dict[ParameterMonomial, int]] = {}
            for left_key, left_coefficient in left.items():
                for right_key, right_coefficient in right.items():
                    key = tuple(sorted((*left_key, *right_key)))
                    if len(key) > max_degree:
                        raise ValueError(
                            f"Expanded monomial has degree {len(key)}; "
                            f"max_degree={max_degree}."
                        )
                    products: dict[ParameterMonomial, int] = {}
                    for left_slots, left_weight in left_coefficient.items():
                        for right_slots, right_weight in right_coefficient.items():
                            slots = tuple(sorted((*left_slots, *right_slots)))
                            products[slots] = (
                                products.get(slots, 0) + left_weight * right_weight
                            )
                            if len(products) > max_coefficient_terms:
                                raise ValueError(f"Symbolic coefficient exceeds max_coefficient_terms={max_coefficient_terms}.")
                    combined = _add_coefficients(result.get(key, {}), products)
                    if len(combined) > max_coefficient_terms:
                        raise ValueError(f"Symbolic coefficient exceeds max_coefficient_terms={max_coefficient_terms}.")
                    if combined:
                        result[key] = combined
                    else:
                        result.pop(key, None)
                    if len(result) > max_terms:
                        raise ValueError(f"Expanded polynomial exceeds max_terms={max_terms}.")
            return result
        if isinstance(node, Tanh):
            if not isinstance(node.operand, Lag):
                raise TypeError("Only Tanh(Lag) is supported as a canonical factor.")
            lag = node.operand
            if type(lag.variable) is not int or lag.variable < 0:
                raise ValueError("Series IDs must be non-negative integers.")
            if max_degree < 1:
                raise ValueError("A lag term exceeds max_degree=0.")
            return {((lag.variable, binding.lag_values[lag.slot], "tanh"),): {(): 1}}
        raise TypeError(f"Unsupported expression node: {type(node).__name__}.")

    symbolic = expand(expression)
    terms: list[CanonicalTerm] = []
    for monomial in sorted(symbolic, key=monomial_order):
        coefficient = Fraction(0)
        for slots, weight in symbolic[monomial].items():
            product = Fraction(weight)
            for slot in slots:
                product *= Fraction.from_float(float(binding.parameter_values[slot]))
            coefficient += product
        if coefficient:
            terms.append(
                CanonicalTerm(
                    monomial,
                    coefficient,
                    tuple(sorted(symbolic[monomial].items())),
                )
            )
    return CanonicalPolynomial(
        tuple(terms), tuple(sorted(symbolic, key=monomial_order)), max_degree
    )


def format_monomial(monomial: Monomial) -> str:
    """Render one sorted lag product with multiplicity as repeated factors."""

    return " * ".join(
        f"tanh(s_{atom[0]}[t-{atom[1]}])" if len(atom) == 3
        else f"s_{atom[0]}[t-{atom[1]}]" for atom in monomial
    ) or "1"


def format_canonical_polynomial(polynomial: CanonicalPolynomial, *, exact: bool = False) -> str:
    """Render ordered terms; approximate decimals by default, exact fractions optionally.

    Coefficients outside float range always fall back to exact fractions.
    This human-readable display is not the identity/serialization of a function;
    use ``canonical_key`` for exact comparisons.
    """

    if not polynomial.terms:
        return "0"
    pieces = []
    for term in polynomial.terms:
        try:
            coefficient = str(term.coefficient_exact) if exact else f"{term.coefficient:.6g}"
        except ValueError:
            coefficient = str(term.coefficient_exact)
        pieces.append(
            coefficient if not term.monomial else f"({coefficient})*{format_monomial(term.monomial)}"
        )
    return " + ".join(pieces)
