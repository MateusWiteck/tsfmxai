import numpy as np
import pytest

from tsfmxai.generators import generate_univariate_trajectory, sample_univariate_trajectories
from tsfmxai.representations import Lag, MechanismBinding, Multiply, Parameter


def autoregressive_mechanism() -> tuple[Multiply, MechanismBinding]:
    expression = Multiply(Parameter("P0"), Lag(0, "L0"))
    binding = MechanismBinding(parameter_values={"P0": 0.5}, lag_values={"L0": 1})
    return expression, binding


def test_bound_recurrence_is_executed_step_by_step() -> None:
    expression, binding = autoregressive_mechanism()

    values = generate_univariate_trajectory(
        expression,
        binding,
        initial_values=[2.0],
        generated_steps=3,
    )

    np.testing.assert_allclose(values, [2.0, 1.0, 0.5, 0.25])


def test_trajectory_ensemble_is_reproducible() -> None:
    expression, binding = autoregressive_mechanism()
    arguments = dict(
        trajectory_count=4,
        generated_steps=8,
        initial_value_minimum=-2.0,
        initial_value_maximum=2.0,
        innovation_std=0.1,
        root_seed=77,
    )

    first = sample_univariate_trajectories(expression, binding, **arguments)
    second = sample_univariate_trajectories(expression, binding, **arguments)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 9)
    assert len(np.unique(first[:, 0])) == 4


def test_univariate_execution_rejects_foreign_series_lags() -> None:
    expression = Lag(1, "L0")
    binding = MechanismBinding(parameter_values={}, lag_values={"L0": 1})

    with pytest.raises(ValueError, match="Univariate execution"):
        sample_univariate_trajectories(
            expression,
            binding,
            trajectory_count=1,
            generated_steps=2,
        )


def test_magnitude_limit_can_be_disabled() -> None:
    expression = Parameter("P0")
    binding = MechanismBinding(parameter_values={"P0": 20_000.0}, lag_values={})

    values = generate_univariate_trajectory(
        expression,
        binding,
        initial_values=[],
        generated_steps=1,
        maximum_absolute_value=None,
    )

    np.testing.assert_array_equal(values, [20_000.0])
