"""Exact and exhaustive-search contracts; no Monte Carlo identity claims."""

from collections import Counter

import pytest

from tsfmxai.evaluation.canonicalization_audit import (
    enumerate_ordered_trees, exact_expression, search_first_counterexamples,
)
from tsfmxai.representations.expression import (
    Add, Lag, Multiply, Parameter, Tanh, operator_count,
)


X, Y = Lag(0, "l1"), Lag(0, "l2")


def test_enumeration_is_ordered_exhaustive_and_deterministic():
    trees = list(enumerate_ordered_trees((X, Y), max_operators=3))
    assert Counter(map(operator_count, trees)) == {0: 2, 1: 10, 2: 90, 3: 1010}
    assert len(set(trees)) == len(trees)
    assert trees == list(enumerate_ordered_trees((X, Y), max_operators=3))
    assert Add(X, Y) in trees and Add(Y, X) in trees
    assert Tanh(Multiply(X, Y)) in trees
    assert list(enumerate_ordered_trees((X,), max_operators=0)) == [X]


@pytest.mark.parametrize("kwargs", [
    {"leaves": ()}, {"leaves": (X, X)}, {"leaves": (Add(X, Y),)},
    {"leaves": (X,), "max_operators": -1},
    {"leaves": (X,), "binary": (Add, Add)},
    {"leaves": (X,), "unary": (Add,)},
])
def test_reject_invalid_grammars(kwargs):
    options = {"max_operators": 2, **kwargs}
    with pytest.raises(ValueError):
        list(enumerate_ordered_trees(**options))


def test_exact_oracle_proves_distributivity_without_sampling():
    pytest.importorskip("sympy")
    assert exact_expression(Multiply(X, Add(X, Y))) == exact_expression(
        Add(Multiply(X, X), Multiply(X, Y))
    )
    assert exact_expression(X) != exact_expression(Y)


def test_binding_and_unary_scope_are_explicit():
    pytest.importorskip("sympy")
    two, negative = Parameter("two"), Parameter("minus_one")
    assert exact_expression(Add(X, X)) != exact_expression(Multiply(two, X))
    bindings = {"two": 2, "minus_one": -1}
    assert exact_expression(Add(X, X), parameter_values=bindings) == exact_expression(
        Multiply(two, X), parameter_values=bindings
    )
    a, b = Tanh(Multiply(negative, X)), Multiply(negative, Tanh(X))
    assert exact_expression(a, parameter_values=bindings) != exact_expression(
        b, parameter_values=bindings
    )
    assert exact_expression(a, parameter_values=bindings, analytic_tanh=True) == (
        exact_expression(b, parameter_values=bindings, analytic_tanh=True)
    )
    assert exact_expression(X, lag_values={"l1": 1}) == exact_expression(
        Y, lag_values={"l2": 1}
    )
    with pytest.raises(TypeError):
        exact_expression(two, parameter_values={"two": 2.0})
    with pytest.raises(ValueError):
        exact_expression(X, lag_values={"l1": 0})


def test_first_failure_stops_only_the_affected_method():
    result = search_first_counterexamples(
        iter([X, Y, Add(X, Y)]),
        methods={"raw": repr, "constant": lambda tree: "same"},
        oracle=lambda tree: "same",
    )
    assert result.counterexamples["raw"].right == Y
    assert result.tested_by_method == {"raw": 2, "constant": 3}
    assert result.stop_reason == "size_bound_exhausted"
    # The constant method illustrates why no duplicate witness alone is not
    # sufficient evidence of a correct, meaning-preserving canonicalizer.


def test_exception_is_distinct_from_a_counterexample():
    def broken(tree):
        raise RuntimeError("budget")

    result = search_first_counterexamples(
        iter([X, Y]), methods={"broken": broken}, oracle=repr,
    )
    assert result.errors_by_method == {"broken": 2}
    assert not result.counterexamples
    assert result.first_errors["broken"] == "RuntimeError: budget"


def test_search_finds_a_proven_pair_and_stops():
    pytest.importorskip("sympy")
    result = search_first_counterexamples(
        enumerate_ordered_trees((X, Y), max_operators=2, unary=()),
        methods={"unmodified": repr}, oracle=exact_expression,
    )
    witness = result.counterexamples["unmodified"]
    assert witness.left != witness.right
    assert exact_expression(witness.left) == exact_expression(witness.right)
    assert witness.operator_layer == 1
    assert result.stop_reason == "all_methods_found"


def test_ineligible_trees_still_count_in_visited_prefix():
    result = search_first_counterexamples(
        iter([X, Y]), methods={"raw": repr}, oracle=repr,
        eligible=lambda tree: tree == Y,
    )
    assert (result.visited, result.eligible, result.tested_by_method["raw"]) == (2, 1, 1)
