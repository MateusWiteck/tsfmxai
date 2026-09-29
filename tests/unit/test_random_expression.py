from tsfmxai.generators import (
    RandomTreeConfig,
    derive_sample_seed,
    full_binary_tree_shape_count,
    sample_random_mechanism,
    sample_random_mechanisms,
)
from tsfmxai.representations import (
    Add,
    Lag,
    Multiply,
    Parameter,
    canonicalize_and_relabel,
    lag_slots,
    leaf_count,
    node_count,
    operator_count,
    parameter_slots,
)


def test_catalan_shape_counts() -> None:
    assert [full_binary_tree_shape_count(index) for index in range(6)] == [1, 1, 2, 5, 14, 42]


def test_sample_is_reproducible_and_factorized() -> None:
    config = RandomTreeConfig(maximum_operators=5, num_series=2)
    first = sample_random_mechanism(config, seed=1234)
    second = sample_random_mechanism(config, seed=1234)

    assert first == second
    assert set(parameter_slots(first.expression)) == set(first.binding.parameter_values)
    assert set(lag_slots(first.expression)) == set(first.binding.lag_values)
    assert lag_slots(first.expression)
    assert all(
        config.minimum_lag <= lag <= config.maximum_lag
        for lag in first.binding.lag_values.values()
    )


def test_full_binary_tree_invariants_hold() -> None:
    config = RandomTreeConfig(minimum_operators=4, maximum_operators=4)
    sample = sample_random_mechanism(config, seed=9)

    assert operator_count(sample.expression) == 4
    assert leaf_count(sample.expression) == 5
    assert node_count(sample.expression) == 9


def test_zero_operators_produces_one_required_lag_terminal() -> None:
    config = RandomTreeConfig(minimum_operators=0, maximum_operators=0, require_lag=True)
    sample = sample_random_mechanism(config, seed=7)

    assert isinstance(sample.expression, Lag)
    assert operator_count(sample.expression) == 0
    assert leaf_count(sample.expression) == 1


def test_commutative_permutations_share_a_canonical_skeleton() -> None:
    left = Add(Parameter("arbitrary_a"), Lag(0, "arbitrary_lag"))
    right = Add(Lag(0, "other_lag"), Parameter("arbitrary_b"))

    assert canonicalize_and_relabel(left) == canonicalize_and_relabel(right)

    product_left = Multiply(Parameter("x"), Lag(1, "tau"))
    product_right = Multiply(Lag(1, "delay"), Parameter("coefficient"))
    assert canonicalize_and_relabel(product_left) == canonicalize_and_relabel(product_right)


def test_indexed_batch_is_order_independent() -> None:
    config = RandomTreeConfig(maximum_operators=3)
    batch = sample_random_mechanisms(config, root_seed=20260929, count=5)

    reconstructed = [
        sample_random_mechanism(
            config,
            seed=derive_sample_seed(20260929, index),
            sample_index=index,
        )
        for index in reversed(range(5))
    ]

    assert batch == list(reversed(reconstructed))
    assert len({sample.seed for sample in batch}) == len(batch)


def test_required_lag_and_time_are_both_preserved() -> None:
    config = RandomTreeConfig(
        minimum_operators=1,
        maximum_operators=1,
        parameter_probability=1.0,
        time_probability=0.0,
        lag_probability=0.0,
        require_lag=True,
        require_time=True,
    )

    for seed in range(20):
        sample = sample_random_mechanism(config, seed=seed)
        assert lag_slots(sample.expression)
        assert "t" in sample.skeleton_formula
