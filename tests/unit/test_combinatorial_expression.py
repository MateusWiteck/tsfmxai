from collections import Counter
from functools import lru_cache
from math import comb

import numpy as np
import pytest

from tsfmxai.generators.combinatorial_expression import (
    CombinatorialTreeConfig, CombinatorialTreeSampler,
)
from tsfmxai.representations import (
    Add, Lag, Multiply, Parameter, canonicalize_polynomial, operator_count, walk,
)


@lru_cache(None)
def brute_trees(n):
    if n == 0:
        return (("p",), ("lag", 0, 2), ("lag", 1, 2))
    return tuple(
        (op, left, right)
        for k in range(n)
        for left in brute_trees(k)
        for right in brute_trees(n - 1 - k)
        for op in ("+", "*")
    )


def degree(tree):
    if tree[0] == "p":
        return 0
    if tree[0] == "lag":
        return 1
    a, b = degree(tree[1]), degree(tree[2])
    return max(a, b) if tree[0] == "+" else a + b


def signature(node, binding):
    if isinstance(node, Parameter):
        return ("p",)
    if isinstance(node, Lag):
        return ("lag", node.variable, binding.lag_values[node.slot])
    return ("+" if isinstance(node, Add) else "*",
            signature(node.left, binding), signature(node.right, binding))


def test_counts_and_unranking_match_independent_exhaustive_enumeration():
    config = CombinatorialTreeConfig(minimum_operators=0, maximum_operators=3,
                                      num_series=2, minimum_lag=2, maximum_lag=2,
                                      max_degree=2)
    sampler = CombinatorialTreeSampler(config)
    expected = {tree for n in range(4) for tree in brute_trees(n)
                if 1 <= degree(tree) <= 2}
    seen = set()
    size_counts = Counter()
    additions = parameters = operators = leaves = 0
    for rank in range(sampler.total_structures):
        mechanism = sampler.unrank(rank)
        key = signature(mechanism.expression, mechanism.binding)
        assert key not in seen
        seen.add(key)
        size_counts[operator_count(mechanism.expression)] += 1
        canonicalize_polynomial(mechanism.expression, mechanism.binding, max_degree=2)
        for node in walk(mechanism.expression):
            additions += isinstance(node, Add)
            parameters += isinstance(node, Parameter)
            operators += isinstance(node, (Add, Multiply))
            leaves += isinstance(node, (Parameter, Lag))
    assert seen == expected
    assert sampler.size_counts() == dict(size_counts)
    assert sampler.marginal_profile() == {
        "mean_operators": operators / len(seen),
        "addition_fraction": additions / operators,
        "parameter_fraction": parameters / leaves,
    }


@pytest.mark.parametrize("require_lag", [False, True])
def test_unrestricted_counts_equal_closed_form(require_lag):
    sampler = CombinatorialTreeSampler(CombinatorialTreeConfig(
        minimum_operators=0, maximum_operators=5, max_degree=6,
        require_lag=require_lag, num_series=2, maximum_lag=3,
    ))
    for n, count in sampler.size_counts().items():
        catalan = comb(2 * n, n) // (n + 1)
        assert count == catalan * 2**n * (7**(n + 1) - int(require_lag))


def test_seed_is_repeatable_order_independent_and_ignores_external_rng():
    sampler = CombinatorialTreeSampler()
    first = sampler.sample(seed=1234, sample_index=7)
    np.random.seed(999)
    np.random.uniform(size=1000)
    sampler.sample(seed=98)
    assert first == sampler.sample(seed=1234, sample_index=7)
    for seed in range(100):
        sample = sampler.sample(seed=seed)
        assert 1 <= operator_count(sample.expression) <= 6
        assert sample.binding.lag_values
        canonicalize_polynomial(sample.expression, sample.binding, max_degree=4)


def test_zero_degree_and_zero_operator_boundaries():
    sampler = CombinatorialTreeSampler(CombinatorialTreeConfig(
        minimum_operators=0, maximum_operators=0, max_degree=0, require_lag=False,
    ))
    assert sampler.total_structures == 1
    assert isinstance(sampler.sample(seed=1).expression, Parameter)
    assert sampler.marginal_profile()["parameter_fraction"] == 1.0
    for rank in (-1, 1, 0.5):
        with pytest.raises(ValueError):
            sampler.unrank(rank)


def test_integer_counts_and_sampling_work_beyond_uint64():
    sampler = CombinatorialTreeSampler(CombinatorialTreeConfig(maximum_operators=12))
    assert sampler.total_structures > 2**64
    for rank in (0, sampler.total_structures - 1):
        sample = sampler.unrank(rank)
        canonicalize_polynomial(sample.expression, sample.binding)


@pytest.mark.parametrize("kwargs", [
    {"minimum_operators": -1}, {"maximum_operators": 0}, {"num_series": 0},
    {"minimum_lag": 0}, {"max_degree": 0}, {"minimum_parameter_magnitude": 0},
    {"maximum_parameter_magnitude": float("inf")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CombinatorialTreeConfig(**kwargs)
