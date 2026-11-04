"""Polynomial canonicalization is exact within its declared subgrammar."""

from fractions import Fraction
from itertools import product

import pytest

from tsfmxai.generators.trajectory import evaluate_bound_expression
from tsfmxai.generators.shao_wu import saturate_product_lags
from tsfmxai.representations import (
    Add,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Tanh,
    canonicalize_polynomial,
    format_canonical_polynomial,
)


def test_distributive_expansion_order_and_symbolic_coefficients() -> None:
    u, v = Lag(0, "L0"), Lag(2, "L1")
    expression = Add(
        Multiply(
            Add(Multiply(Parameter("a"), u), Parameter("b")),
            Add(Multiply(Parameter("c"), v), Parameter("d")),
        ),
        Multiply(Parameter("e"), u),
    )
    binding = MechanismBinding(
        {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0, "e": 7.0},
        {"L0": 1, "L1": 2},
    )
    result = canonicalize_polynomial(expression, binding)
    assert result.skeleton == ((), ((0, 1),), ((2, 2),), ((0, 1), (2, 2)))
    assert result.coefficients == (15.0, 17.0, 12.0, 8.0)
    assert result.terms[1].symbolic_coefficient == ((('a', 'd'), 1), (('e',), 1))


def test_reordered_equivalent_trees_and_repeated_lags() -> None:
    x1 = Lag(0, "L1")
    x2 = Lag(0, "L2")
    y = Lag(1, "Y")
    first = Multiply(Add(x1, y), x2)
    second = Add(Multiply(y, x2), Multiply(x2, x1))
    binding = MechanismBinding({}, {"L1": 1, "L2": 1, "Y": 2})
    result1 = canonicalize_polynomial(first, binding)
    result2 = canonicalize_polynomial(second, binding)
    assert result1 == result2
    assert result1.skeleton == (((0, 1), (0, 1)), ((0, 1), (1, 2)))


def test_realized_support_drops_zero_coefficient_but_potential_retains_it() -> None:
    expression = Add(Multiply(Parameter("zero"), Lag(0, "L")), Parameter("b"))
    result = canonicalize_polynomial(
        expression, MechanismBinding({"zero": 0.0, "b": 2.0}, {"L": 1})
    )
    assert result.potential_skeleton == ((), ((0, 1),))
    assert result.skeleton == ((),)
    assert result.coefficients == (2.0,)


def test_exact_float_cancellation_and_zero_polynomial() -> None:
    expression = Add(Multiply(Parameter("a"), Lag(0, "L")), Multiply(Parameter("b"), Lag(0, "L")))
    result = canonicalize_polynomial(
        expression, MechanismBinding({"a": 0.125, "b": -0.125}, {"L": 3})
    )
    assert result.potential_skeleton == (((0, 3),),)
    assert result.skeleton == ()
    assert result.evaluate({0: [1.0, 2.0, 3.0]}) == 0.0


def test_four_factors_allowed_and_five_rejected() -> None:
    x = Lag(0, "L")
    binding = MechanismBinding({}, {"L": 1})
    four = Multiply(Multiply(x, x), Multiply(x, x))
    assert canonicalize_polynomial(four, binding).skeleton == (((0, 1),) * 4,)
    with pytest.raises(ValueError, match="degree 5"):
        canonicalize_polynomial(Multiply(four, x), binding)


@pytest.mark.parametrize("node", [Tanh(Multiply(Lag(0, "L"), Lag(0, "L")))])
def test_nonpolynomial_nodes_rejected(node) -> None:
    binding = MechanismBinding({}, {"L": 1} if isinstance(node, Tanh) else {})
    with pytest.raises(TypeError, match=r"Only Tanh\(Lag\)"):
        canonicalize_polynomial(node, binding)


def test_raw_and_tanh_factors_remain_distinct() -> None:
    x = Lag(0, "L")
    expression = Add(x, Tanh(x))
    result = canonicalize_polynomial(expression, MechanismBinding({}, {"L": 1}))
    assert result.skeleton == (((0, 1),), ((0, 1, "tanh"),))
    assert result.evaluate({0: [2.0]}) == pytest.approx(2.0 + __import__("math").tanh(2.0))


def test_saturation_precedes_distribution_and_preserves_ast_local_meaning() -> None:
    x, y, two = Lag(0, "x"), Lag(1, "y"), Parameter("two")
    binding = MechanismBinding({"two": 2.0}, {"x": 1, "y": 2})
    left = Multiply(Add(x, two), y)
    right = Add(Multiply(x, y), Multiply(two, y))
    a = canonicalize_polynomial(saturate_product_lags(left), binding)
    b = canonicalize_polynomial(saturate_product_lags(right), binding)
    assert a.skeleton == (((1, 2, "tanh"),), ((0, 1, "tanh"), (1, 2, "tanh")))
    assert b.skeleton == (((1, 2),), ((0, 1, "tanh"), (1, 2, "tanh")))
    assert a.canonical_key != b.canonical_key


def test_term_budget_and_invalid_binding() -> None:
    x = Lag(0, "L0")
    y = Lag(1, "L1")
    expression = Add(x, y)
    binding = MechanismBinding({}, {"L0": 1, "L1": 1})
    with pytest.raises(ValueError, match="max_terms"):
        canonicalize_polynomial(expression, binding, max_terms=1)
    with pytest.raises(ValueError, match="non-negative"):
        canonicalize_polynomial(expression, binding, max_degree=-1)


def test_exact_coefficient_keeps_binary_float_value() -> None:
    result = canonicalize_polynomial(Parameter("p"), MechanismBinding({"p": 0.1}, {}))
    assert result.terms[0].coefficient_exact == Fraction.from_float(0.1)


def test_functional_equivalence_across_small_histories() -> None:
    x = Lag(0, "L0")
    y = Lag(1, "L1")
    expression = Multiply(Add(Parameter("a"), x), Add(Parameter("b"), y))
    binding = MechanismBinding({"a": 0.25, "b": -0.5}, {"L0": 1, "L1": 2})
    normal = canonicalize_polynomial(expression, binding)
    for x_value, y_value in product((-2.0, 0.0, 3.0), repeat=2):
        histories = {0: [11.0, x_value], 1: [y_value, 11.0]}
        original = evaluate_bound_expression(expression, binding, time=2, histories=histories)
        assert normal.evaluate(histories) == pytest.approx(original)


def test_function_identity_ignores_parameter_names_and_factorization() -> None:
    x = Lag(0, "L")
    variants = [
        (Multiply(Parameter("a"), x), {"a": 2.0}),
        (Multiply(x, Parameter("renamed")), {"renamed": 2.0}),
        (Add(x, x), {}),
        (Add(Multiply(Parameter("b"), x), x), {"b": 1.0}),
    ]
    forms = [canonicalize_polynomial(expr, MechanismBinding(values, {"L": 1}))
             for expr, values in variants]
    assert len({form.canonical_key for form in forms}) == 1
    # Metadata differs, while the actual function identity is shared.
    assert forms[0].terms[0].symbolic_coefficient != forms[1].terms[0].symbolic_coefficient


def test_order_is_numeric_including_series_ten_and_lags_ten() -> None:
    expression = Add(Add(Lag(10, "a"), Lag(2, "b")), Lag(2, "c"))
    normal = canonicalize_polynomial(expression, MechanismBinding({}, {"a": 1, "b": 10, "c": 2}))
    assert normal.skeleton == (((2, 2),), ((2, 10),), ((10, 1),))


@pytest.mark.parametrize("value", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_parameters_rejected(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonicalize_polynomial(Parameter("p"), MechanismBinding({"p": value}, {}))


@pytest.mark.parametrize("options", [
    {"max_degree": True}, {"max_degree": 1.5}, {"max_terms": 0},
    {"max_terms": 1.5}, {"max_coefficient_terms": False},
])
def test_invalid_configuration(options) -> None:
    with pytest.raises(ValueError, match="integer"):
        canonicalize_polynomial(Parameter("p"), MechanismBinding({"p": 1.0}, {}), **options)


def test_coefficient_budget_covers_constant_only_expansion() -> None:
    expression = Multiply(Add(Parameter("a"), Parameter("b")), Add(Parameter("c"), Parameter("d")))
    binding = MechanismBinding({slot: 1.0 for slot in 'abcd'}, {})
    assert canonicalize_polynomial(expression, binding, max_coefficient_terms=4).coefficients == (4.0,)
    with pytest.raises(ValueError, match="max_coefficient_terms"):
        canonicalize_polynomial(expression, binding, max_coefficient_terms=3)


def test_zero_and_cancelled_high_degree_still_follow_input_degree_contract() -> None:
    x = Lag(0, "L")
    five = Multiply(Multiply(Multiply(Multiply(x, x), x), x), x)
    for expression, parameters in [
        (Multiply(Parameter("zero"), five), {"zero": 0.0}),
        (Add(five, Multiply(Parameter("negative"), five)), {"negative": -1.0}),
    ]:
        with pytest.raises(ValueError, match="degree 5"):
            canonicalize_polynomial(expression, MechanismBinding(parameters, {"L": 1}))


@pytest.mark.parametrize("value", [1e-300, 1e300])
def test_float_range_does_not_silently_change_exact_support(value) -> None:
    expression = Multiply(Parameter("a"), Parameter("b"))
    normal = canonicalize_polynomial(expression, MechanismBinding({"a": value, "b": value}, {}))
    assert normal.skeleton == ((),)
    assert normal.terms[0].coefficient_exact == Fraction.from_float(value) ** 2
    with pytest.raises(ValueError, match="float"):
        _ = normal.coefficients
    assert format_canonical_polynomial(normal) == str(normal.terms[0].coefficient_exact)


def test_decimal_residue_retained_and_roundoff_distinguished_from_algebra() -> None:
    normal = canonicalize_polynomial(
        Add(Add(Parameter("a"), Parameter("b")), Parameter("c")),
        MechanismBinding({"a": 0.1, "b": 0.2, "c": -0.3}, {}),
    )
    assert normal.terms[0].coefficient_exact == Fraction(1, 2**55)
    x = Lag(0, "L")
    expression = Add(Add(Parameter("big"), x), Parameter("negative"))
    binding = MechanismBinding({"big": 1e16, "negative": -1e16}, {"L": 1})
    result = canonicalize_polynomial(expression, binding)
    assert result.canonical_key == ((((0, 1),), Fraction(1)),)
    assert result.evaluate({0: [1.0]}) == 1.0
    assert evaluate_bound_expression(expression, binding, time=1, histories={0: [1.0]}) == 0.0


def test_evaluation_uses_shared_time_and_validates_history() -> None:
    expr = Add(Lag(0, "a"), Lag(1, "b"))
    binding = MechanismBinding({}, {"a": 1, "b": 2})
    normal = canonicalize_polynomial(expr, binding)
    histories = {0: [1.0, 2.0, 99.0], 1: [3.0, 4.0]}
    assert normal.evaluate(histories, time=2) == 5.0
    with pytest.raises(ValueError, match="share a length"):
        normal.evaluate(histories)
    with pytest.raises(ValueError, match="unavailable"):
        normal.evaluate({0: [1.0], 1: [2.0]}, time=1)
    with pytest.raises(ValueError, match="Missing history"):
        normal.evaluate({0: [1.0, 2.0]})


def test_exhaustive_small_trees_exact_semantics_and_commutativity() -> None:
    """Enumerate all binary trees up to two operators over four terminals."""
    from tsfmxai.representations.expression import lag_slots, parameter_slots

    levels = [[Lag(0, "x"), Lag(1, "y"), Parameter("p"), Parameter("n")]]
    for size in (1, 2):
        levels.append([op(left, right) for left_size in range(size)
                       for left in levels[left_size] for right in levels[size - 1 - left_size]
                       for op in (Add, Multiply)])

    def exact_ast(node, parameters, atoms):
        if isinstance(node, Parameter):
            return Fraction(parameters[node.slot])
        if isinstance(node, Lag):
            return atoms[node.variable]
        if isinstance(node, Add):
            return exact_ast(node.left, parameters, atoms) + exact_ast(node.right, parameters, atoms)
        return exact_ast(node.left, parameters, atoms) * exact_ast(node.right, parameters, atoms)

    for nodes in levels:
        for expression in nodes:
            parameters = {slot: {'p': 2, 'n': -1}[slot] for slot in parameter_slots(expression)}
            binding = MechanismBinding(parameters, {slot: 1 for slot in lag_slots(expression)})
            normal = canonicalize_polynomial(expression, binding)
            for atoms in ((Fraction(2), Fraction(-3)), (Fraction(1, 3), Fraction(7, 5))):
                total = Fraction(0)
                for key, coefficient in normal.canonical_key:
                    term = coefficient
                    for series, _ in key:
                        term *= atoms[series]
                    total += term
                assert total == exact_ast(expression, parameters, atoms)
            if isinstance(expression, (Add, Multiply)):
                swapped = type(expression)(expression.right, expression.left)
                assert normal.canonical_key == canonicalize_polynomial(swapped, binding).canonical_key
