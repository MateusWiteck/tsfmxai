import random

import numpy as np
import pytest

from tsfmxai.generators import (
    generate_multivariate_trajectory,
    required_multivariate_history_length,
    sample_multivariate_trajectories,
)
from tsfmxai.representations import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
)


def two_series_system() -> tuple[list[ExpressionNode], list[MechanismBinding]]:
    # x0(t) = 0.5 x0(t-1) + x1(t-1)
    first = Add(Multiply(Parameter("P0"), Lag(0, "L0")), Lag(1, "L1"))
    first_binding = MechanismBinding(
        parameter_values={"P0": 0.5},
        lag_values={"L0": 1, "L1": 1},
    )
    # x1(t) = x0(t-1)
    second = Lag(0, "L0")
    second_binding = MechanismBinding(parameter_values={}, lag_values={"L0": 1})
    return [first, second], [first_binding, second_binding]


def test_two_series_recurrence_is_executed_synchronously() -> None:
    expressions, bindings = two_series_system()

    values = generate_multivariate_trajectory(
        expressions,
        bindings,
        initial_values=[[2.0], [4.0]],
        generated_steps=2,
    )

    np.testing.assert_allclose(
        values,
        [
            [2.0, 5.0, 4.5],
            [4.0, 2.0, 5.0],
        ],
    )


def test_multivariate_ensemble_is_reproducible_and_locally_seeded() -> None:
    expressions, bindings = two_series_system()
    arguments = dict(
        trajectory_count=4,
        generated_steps=8,
        initial_value_minimum=-2.0,
        initial_value_maximum=2.0,
        innovation_std=0.1,
        root_seed=77,
    )

    random.seed(1)
    np.random.seed(2)
    first = sample_multivariate_trajectories(expressions, bindings, **arguments)
    random.seed(999)
    np.random.seed(888)
    second = sample_multivariate_trajectories(expressions, bindings, **arguments)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 2, 9)
    assert len(np.unique(first[:, 0, 0])) == 4
    assert len(np.unique(first[:, 1, 0])) == 4


def test_shared_history_uses_largest_lag_across_equations() -> None:
    expressions = [Lag(1, "L0"), Lag(0, "L0")]
    bindings = [
        MechanismBinding(parameter_values={}, lag_values={"L0": 3}),
        MechanismBinding(parameter_values={}, lag_values={"L0": 7}),
    ]

    assert required_multivariate_history_length(expressions, bindings) == 7


def test_multivariate_execution_rejects_unavailable_source_series() -> None:
    expressions = [Lag(2, "L0"), Lag(0, "L0")]
    bindings = [
        MechanismBinding(parameter_values={}, lag_values={"L0": 1}),
        MechanismBinding(parameter_values={}, lag_values={"L0": 1}),
    ]

    with pytest.raises(ValueError, match="unavailable series"):
        required_multivariate_history_length(expressions, bindings)


def test_multivariate_initial_values_must_match_shared_history() -> None:
    expressions, bindings = two_series_system()

    with pytest.raises(ValueError, match="initial_values with shape"):
        generate_multivariate_trajectory(
            expressions,
            bindings,
            initial_values=[[1.0], [2.0, 3.0]],
            generated_steps=1,
        )


def test_multivariate_execution_supports_correlated_innovations() -> None:
    expressions = [Lag(0, "L0"), Lag(1, "L0")]
    bindings = [
        MechanismBinding(parameter_values={}, lag_values={"L0": 1}),
        MechanismBinding(parameter_values={}, lag_values={"L0": 1}),
    ]
    covariance = np.array([[1.0, 0.75], [0.75, 1.0]])
    draws = np.stack(
        [
            generate_multivariate_trajectory(
                expressions,
                bindings,
                initial_values=[[0.0], [0.0]],
                generated_steps=1,
                innovation_covariance=covariance,
                seed=seed,
            )[:, 1]
            for seed in range(2_000)
        ]
    )

    np.testing.assert_allclose(np.cov(draws, rowvar=False, bias=True), covariance, atol=0.08)


def test_multivariate_execution_rejects_two_innovation_specifications() -> None:
    expressions, bindings = two_series_system()

    with pytest.raises(ValueError, match="either innovation_std or innovation_covariance"):
        generate_multivariate_trajectory(
            expressions,
            bindings,
            initial_values=[[0.0], [0.0]],
            generated_steps=1,
            innovation_std=0.1,
            innovation_covariance=np.eye(2),
        )
