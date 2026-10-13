"""Execution of bound symbolic mechanisms as discrete-time recurrences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Time,
    Tanh,
    validate_binding,
    walk,
)


class DivergentTrajectoryError(ValueError):
    """Raised when a generated trajectory becomes non-finite or exceeds a limit."""


def required_history_length(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    target_series: int = 0,
) -> int:
    """Return the largest bound lag required by one target recurrence."""

    history_length = expression_required_history_length(expression, binding)
    lag_nodes = [node for node in walk(expression) if isinstance(node, Lag)]
    foreign_series = sorted({node.variable for node in lag_nodes if node.variable != target_series})
    if foreign_series:
        raise ValueError(
            "Univariate execution only accepts lags of the target series; "
            f"found series indices {foreign_series}."
        )
    return history_length


def expression_required_history_length(
    expression: ExpressionNode,
    binding: MechanismBinding,
) -> int:
    """Return the largest lag used by one expression, regardless of source series."""

    validate_binding(expression, binding)
    return max(
        (
            binding.lag_values[node.slot]
            for node in walk(expression)
            if isinstance(node, Lag)
        ),
        default=0,
    )


def required_multivariate_history_length(
    expressions: Sequence[ExpressionNode],
    bindings: Sequence[MechanismBinding],
) -> int:
    """Return the shared history required by a synchronous multivariate system."""

    if not expressions:
        raise ValueError("A multivariate system must contain at least one equation.")
    if len(expressions) != len(bindings):
        raise ValueError("Every multivariate equation must have one binding.")

    num_series = len(expressions)
    history_lengths: list[int] = []
    for target_series, (expression, binding) in enumerate(zip(expressions, bindings)):
        validate_binding(expression, binding)
        invalid_sources = sorted(
            {
                node.variable
                for node in walk(expression)
                if isinstance(node, Lag) and node.variable >= num_series
            }
        )
        if invalid_sources:
            raise ValueError(
                f"Equation x_{target_series} references unavailable series "
                f"indices {invalid_sources}; the system contains {num_series} series."
            )
        history_lengths.append(expression_required_history_length(expression, binding))
    return max(history_lengths, default=0)


def evaluate_bound_expression(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    time: int,
    histories: Mapping[int, Sequence[float]],
) -> float:
    """Evaluate a bound AST at one discrete time using the supplied histories."""

    if isinstance(expression, Parameter):
        return float(binding.parameter_values[expression.slot])
    if isinstance(expression, Time):
        return float(time)
    if isinstance(expression, Tanh):
        return float(np.tanh(evaluate_bound_expression(
            expression.operand, binding, time=time, histories=histories,
        )))
    if isinstance(expression, Lag):
        lag = binding.lag_values[expression.slot]
        try:
            return float(histories[expression.variable][time - lag])
        except (KeyError, IndexError) as error:
            raise ValueError(
                f"History for x_{expression.variable}(t-{lag}) is unavailable at t={time}."
            ) from error
    if isinstance(expression, Add):
        return evaluate_bound_expression(
            expression.left, binding, time=time, histories=histories
        ) + evaluate_bound_expression(expression.right, binding, time=time, histories=histories)
    if isinstance(expression, Multiply):
        return evaluate_bound_expression(
            expression.left, binding, time=time, histories=histories
        ) * evaluate_bound_expression(expression.right, binding, time=time, histories=histories)
    raise TypeError(type(expression))


def generate_univariate_trajectory(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    initial_values: Sequence[float],
    generated_steps: int,
    innovation_std: float = 0.0,
    seed: int = 0,
    maximum_absolute_value: float | None = 1_000_000.0,
    target_series: int = 0,
) -> np.ndarray:
    """Execute one recurrence, optionally adding Gaussian innovation per generated step."""

    if generated_steps < 0:
        raise ValueError("generated_steps must be non-negative.")
    if innovation_std < 0:
        raise ValueError("innovation_std must be non-negative.")
    if maximum_absolute_value is not None and maximum_absolute_value <= 0:
        raise ValueError("maximum_absolute_value must be positive.")

    history_length = required_history_length(expression, binding, target_series=target_series)
    if len(initial_values) != history_length:
        raise ValueError(
            f"Expected exactly {history_length} initial values, received {len(initial_values)}."
        )

    values = [float(value) for value in initial_values]
    if any(not np.isfinite(value) for value in values):
        raise ValueError("Initial values must be finite.")

    rng = np.random.default_rng(seed)
    final_time = history_length + generated_steps
    for time in range(history_length, final_time):
        next_value = evaluate_bound_expression(
            expression,
            binding,
            time=time,
            histories={target_series: values},
        )
        if innovation_std:
            next_value += float(rng.normal(0.0, innovation_std))
        if not np.isfinite(next_value):
            raise DivergentTrajectoryError(
                f"Trajectory produced a non-finite value at t={time}: value={next_value!r}."
            )
        if maximum_absolute_value is not None and abs(next_value) > maximum_absolute_value:
            raise DivergentTrajectoryError(
                f"Trajectory exceeded the configured magnitude at t={time}: "
                f"value={next_value!r}, limit={maximum_absolute_value:g}."
            )
        values.append(float(next_value))
    return np.asarray(values, dtype=float)


def sample_univariate_trajectories(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    trajectory_count: int,
    generated_steps: int,
    initial_value_minimum: float = -1.0,
    initial_value_maximum: float = 1.0,
    innovation_std: float = 0.0,
    root_seed: int = 0,
    maximum_absolute_value: float | None = 1_000_000.0,
    target_series: int = 0,
) -> np.ndarray:
    """Sample independent realizations of one fixed recurrence."""

    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive.")
    if initial_value_maximum < initial_value_minimum:
        raise ValueError("The initial-value interval is invalid.")

    history_length = required_history_length(expression, binding, target_series=target_series)
    root_sequence = np.random.SeedSequence(int(root_seed))
    child_sequences = root_sequence.spawn(trajectory_count)
    trajectories: list[np.ndarray] = []
    for child_sequence in child_sequences:
        initialization_sequence, innovation_sequence = child_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        initial_values = initialization_rng.uniform(
            initial_value_minimum,
            initial_value_maximum,
            size=history_length,
        )
        innovation_seed = int(innovation_sequence.generate_state(1, dtype=np.uint64)[0])
        trajectories.append(
            generate_univariate_trajectory(
                expression,
                binding,
                initial_values=initial_values,
                generated_steps=generated_steps,
                innovation_std=innovation_std,
                seed=innovation_seed,
                maximum_absolute_value=maximum_absolute_value,
                target_series=target_series,
            )
        )
    return np.stack(trajectories)


def _multivariate_innovation_transform(
    num_series: int,
    *,
    innovation_std: float,
    innovation_covariance: Sequence[Sequence[float]] | None,
) -> np.ndarray:
    if innovation_std < 0:
        raise ValueError("innovation_std must be non-negative.")
    if innovation_covariance is None:
        return np.eye(num_series, dtype=float) * innovation_std
    if innovation_std != 0.0:
        raise ValueError(
            "Specify either innovation_std or innovation_covariance, not both."
        )

    covariance = np.asarray(innovation_covariance, dtype=float)
    if covariance.shape != (num_series, num_series):
        raise ValueError(
            "innovation_covariance must have shape "
            f"({num_series}, {num_series}), received {covariance.shape}."
        )
    if not np.all(np.isfinite(covariance)):
        raise ValueError("innovation_covariance must be finite.")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
        raise ValueError("innovation_covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-12:
        raise ValueError("innovation_covariance must be positive semidefinite.")
    return eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))


def generate_multivariate_trajectory(
    expressions: Sequence[ExpressionNode],
    bindings: Sequence[MechanismBinding],
    *,
    initial_values: Sequence[Sequence[float]],
    generated_steps: int,
    innovation_std: float = 0.0,
    innovation_covariance: Sequence[Sequence[float]] | None = None,
    seed: int = 0,
    maximum_absolute_value: float | None = 1_000_000.0,
) -> np.ndarray:
    """Execute one synchronous recurrence system and return ``[series, time]``."""

    if generated_steps < 0:
        raise ValueError("generated_steps must be non-negative.")
    if maximum_absolute_value is not None and maximum_absolute_value <= 0:
        raise ValueError("maximum_absolute_value must be positive.")

    history_length = required_multivariate_history_length(expressions, bindings)
    num_series = len(expressions)
    innovation_transform = _multivariate_innovation_transform(
        num_series,
        innovation_std=innovation_std,
        innovation_covariance=innovation_covariance,
    )
    try:
        initial_array = np.asarray(initial_values, dtype=float)
    except ValueError as error:
        raise ValueError(
            "Expected rectangular initial_values with shape "
            f"({num_series}, {history_length})."
        ) from error
    if initial_array.shape != (num_series, history_length):
        raise ValueError(
            "Expected initial_values with shape "
            f"({num_series}, {history_length}), received {initial_array.shape}."
        )
    if not np.all(np.isfinite(initial_array)):
        raise ValueError("Initial values must be finite.")

    histories = [list(initial_array[index]) for index in range(num_series)]
    rng = np.random.default_rng(seed)
    final_time = history_length + generated_steps
    for time in range(history_length, final_time):
        # Evaluate every equation against the same past before appending any result.
        next_values = np.asarray(
            [
                evaluate_bound_expression(
                    expression,
                    binding,
                    time=time,
                    histories=histories,
                )
                for expression, binding in zip(expressions, bindings)
            ],
            dtype=float,
        )
        next_values += innovation_transform @ rng.standard_normal(num_series)
        if not np.all(np.isfinite(next_values)):
            raise DivergentTrajectoryError(
                f"Multivariate trajectory produced a non-finite value at t={time}: "
                f"values={next_values.tolist()}."
            )
        if maximum_absolute_value is not None and np.any(
            np.abs(next_values) > maximum_absolute_value
        ):
            raise DivergentTrajectoryError(
                f"Multivariate trajectory exceeded the configured magnitude at t={time}: "
                f"values={next_values.tolist()}, limit={maximum_absolute_value:g}."
            )
        for series_index, value in enumerate(next_values):
            histories[series_index].append(float(value))
    return np.asarray(histories, dtype=float)


def sample_multivariate_trajectories(
    expressions: Sequence[ExpressionNode],
    bindings: Sequence[MechanismBinding],
    *,
    trajectory_count: int,
    generated_steps: int,
    initial_value_minimum: float = -1.0,
    initial_value_maximum: float = 1.0,
    innovation_std: float = 0.0,
    innovation_covariance: Sequence[Sequence[float]] | None = None,
    root_seed: int = 0,
    maximum_absolute_value: float | None = 1_000_000.0,
) -> np.ndarray:
    """Sample realizations of one system as ``[trajectory, series, time]``."""

    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive.")
    if initial_value_maximum < initial_value_minimum:
        raise ValueError("The initial-value interval is invalid.")

    history_length = required_multivariate_history_length(expressions, bindings)
    num_series = len(expressions)
    root_sequence = np.random.SeedSequence(int(root_seed))
    child_sequences = root_sequence.spawn(trajectory_count)
    trajectories: list[np.ndarray] = []
    for child_sequence in child_sequences:
        initialization_sequence, innovation_sequence = child_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        initial_values = initialization_rng.uniform(
            initial_value_minimum,
            initial_value_maximum,
            size=(num_series, history_length),
        )
        innovation_seed = int(innovation_sequence.generate_state(1, dtype=np.uint64)[0])
        trajectories.append(
            generate_multivariate_trajectory(
                expressions,
                bindings,
                initial_values=initial_values,
                generated_steps=generated_steps,
                innovation_std=innovation_std,
                innovation_covariance=innovation_covariance,
                seed=innovation_seed,
                maximum_absolute_value=maximum_absolute_value,
            )
        )
    return np.stack(trajectories)
