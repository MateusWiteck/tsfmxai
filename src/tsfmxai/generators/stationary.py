"""Stationary-by-construction linear time-series mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tsfmxai.generators.random_expression import derive_sample_seed
from tsfmxai.generators.trajectory import (
    generate_multivariate_trajectory,
    generate_univariate_trajectory,
)
from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    canonicalize_commutative,
    format_tree,
    to_bound_formula,
    to_skeleton_formula,
)


@dataclass(frozen=True)
class StationarityCertificate:
    """A pre-simulation mathematical check for one stationary family."""

    family: str
    condition: str
    metric_name: str
    metric_value: float
    threshold: float
    margin: float

    @property
    def holds(self) -> bool:
        """Return whether the certified inequality has a strictly positive margin."""

        return bool(np.isfinite(self.margin) and self.margin > 0.0)


@dataclass(frozen=True)
class StationaryARConfig:
    """Sampling ranges for Gaussian stationary AR mechanisms."""

    minimum_order: int = 1
    maximum_order: int = 8
    partial_autocorrelation_minimum: float = -0.80
    partial_autocorrelation_maximum: float = 0.80
    mean_minimum: float = -1.0
    mean_maximum: float = 1.0
    innovation_std_minimum: float = 0.05
    innovation_std_maximum: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_order < 1:
            raise ValueError("minimum_order must be positive.")
        if self.maximum_order < self.minimum_order:
            raise ValueError("maximum_order must be at least minimum_order.")
        if not (
            -1.0
            < self.partial_autocorrelation_minimum
            <= self.partial_autocorrelation_maximum
            < 1.0
        ):
            raise ValueError(
                "Partial autocorrelations must define a non-empty interval inside (-1, 1)."
            )
        if self.mean_maximum < self.mean_minimum:
            raise ValueError("The stationary-mean interval is invalid.")
        if not (
            0.0 < self.innovation_std_minimum <= self.innovation_std_maximum
        ):
            raise ValueError("Innovation standard deviations must define a positive interval.")


@dataclass(frozen=True)
class StationaryVAR1Config:
    """Sampling ranges for Gaussian stationary VAR(1) mechanisms."""

    num_series: int = 2
    spectral_radius_minimum: float = 0.10
    spectral_radius_maximum: float = 0.90
    edge_probability: float = 0.65
    mean_minimum: float = -1.0
    mean_maximum: float = 1.0
    innovation_std_minimum: float = 0.05
    innovation_std_maximum: float = 0.50

    def __post_init__(self) -> None:
        if self.num_series < 1:
            raise ValueError("num_series must be positive.")
        if not (
            0.0
            <= self.spectral_radius_minimum
            <= self.spectral_radius_maximum
            < 1.0
        ):
            raise ValueError(
                "Spectral radii must define a non-empty interval inside [0, 1)."
            )
        if not 0.0 < self.edge_probability <= 1.0:
            raise ValueError("edge_probability must lie in (0, 1].")
        if self.mean_maximum < self.mean_minimum:
            raise ValueError("The stationary-mean interval is invalid.")
        if not (
            0.0 < self.innovation_std_minimum <= self.innovation_std_maximum
        ):
            raise ValueError("Innovation standard deviations must define a positive interval.")


@dataclass(frozen=True)
class StationaryARMechanism:
    """One stationary Gaussian AR mechanism and its exact initialization law."""

    expression: ExpressionNode
    binding: MechanismBinding
    partial_autocorrelations: tuple[float, ...]
    coefficients: tuple[float, ...]
    stationary_mean: float
    innovation_std: float
    stationary_state_covariance: tuple[tuple[float, ...], ...]
    certificate: StationarityCertificate
    seed: int
    sample_index: int | None

    @property
    def order(self) -> int:
        return len(self.coefficients)

    @property
    def skeleton_formula(self) -> str:
        return f"x_0(t) = {to_skeleton_formula(self.expression)} + epsilon_0(t)"

    @property
    def formula(self) -> str:
        return f"x_0(t) = {to_bound_formula(self.expression, self.binding)} + epsilon_0(t)"

    @property
    def tree(self) -> str:
        return format_tree(self.expression)

    @property
    def innovation_distribution(self) -> str:
        return f"Normal(0, {self.innovation_std**2:.6g})"


@dataclass(frozen=True)
class StationaryVAR1Mechanism:
    """One stationary Gaussian VAR(1) mechanism and exact initialization law."""

    expressions: tuple[ExpressionNode, ...]
    bindings: tuple[MechanismBinding, ...]
    coefficient_matrix: tuple[tuple[float, ...], ...]
    stationary_mean: tuple[float, ...]
    innovation_covariance: tuple[tuple[float, ...], ...]
    stationary_state_covariance: tuple[tuple[float, ...], ...]
    certificate: StationarityCertificate
    seed: int
    sample_index: int | None

    @property
    def num_series(self) -> int:
        return len(self.expressions)

    @property
    def skeleton_formulas(self) -> tuple[str, ...]:
        return tuple(
            f"x_{target}(t) = {to_skeleton_formula(expression)} + epsilon_{target}(t)"
            for target, expression in enumerate(self.expressions)
        )

    @property
    def formulas(self) -> tuple[str, ...]:
        return tuple(
            f"x_{target}(t) = {to_bound_formula(expression, binding)} + epsilon_{target}(t)"
            for target, (expression, binding) in enumerate(
                zip(self.expressions, self.bindings)
            )
        )

    @property
    def trees(self) -> tuple[str, ...]:
        return tuple(format_tree(expression) for expression in self.expressions)

    @property
    def innovation_distribution(self) -> str:
        return "MultivariateNormal(0, innovation_covariance)"


def partial_autocorrelations_to_ar(
    partial_autocorrelations: Sequence[float],
) -> np.ndarray:
    """Map reflection/PACF coefficients in ``(-1, 1)`` to stationary AR coefficients."""

    pacf = np.asarray(partial_autocorrelations, dtype=float)
    if pacf.ndim != 1 or pacf.size < 1:
        raise ValueError("partial_autocorrelations must be a non-empty vector.")
    if not np.all(np.isfinite(pacf)) or np.any(np.abs(pacf) >= 1.0):
        raise ValueError("Every partial autocorrelation must be finite and lie in (-1, 1).")

    coefficients = np.empty(0, dtype=float)
    for reflection in pacf:
        previous = coefficients
        coefficients = np.empty(previous.size + 1, dtype=float)
        if previous.size:
            coefficients[:-1] = previous - reflection * previous[::-1]
        coefficients[-1] = reflection
    return coefficients


def ar_lag_polynomial_roots(coefficients: Sequence[float]) -> np.ndarray:
    """Return roots of ``1 - phi_1 z - ... - phi_p z^p``."""

    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or phi.size < 1:
        raise ValueError("coefficients must be a non-empty vector.")
    if not np.all(np.isfinite(phi)):
        raise ValueError("coefficients must be finite.")
    return np.polynomial.polynomial.polyroots(np.r_[1.0, -phi])


def ar_companion_matrix(coefficients: Sequence[float]) -> np.ndarray:
    """Construct the state transition for ``[x_t, x_{t-1}, ...]``."""

    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or phi.size < 1 or not np.all(np.isfinite(phi)):
        raise ValueError("coefficients must be a non-empty finite vector.")
    companion = np.zeros((phi.size, phi.size), dtype=float)
    companion[0] = phi
    if phi.size > 1:
        companion[1:, :-1] = np.eye(phi.size - 1)
    return companion


def solve_stationary_covariance(
    transition: Sequence[Sequence[float]],
    innovation_covariance: Sequence[Sequence[float]],
) -> np.ndarray:
    """Solve ``P = F P F.T + Q`` for a stable discrete-time linear system."""

    matrix = np.asarray(transition, dtype=float)
    noise = np.asarray(innovation_covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be a square matrix.")
    if noise.shape != matrix.shape:
        raise ValueError("innovation_covariance must match the transition shape.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(noise)):
        raise ValueError("Transition and covariance entries must be finite.")
    if not np.allclose(noise, noise.T, atol=1e-12, rtol=0.0):
        raise ValueError("innovation_covariance must be symmetric.")
    if np.min(np.linalg.eigvalsh(noise)) < -1e-12:
        raise ValueError("innovation_covariance must be positive semidefinite.")

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if spectral_radius >= 1.0:
        raise ValueError("transition must have spectral radius below one.")

    dimension = matrix.shape[0]
    operator = np.eye(dimension * dimension) - np.kron(matrix, matrix)
    covariance_vector = np.linalg.solve(
        operator,
        noise.reshape(dimension * dimension, order="F"),
    )
    covariance = covariance_vector.reshape((dimension, dimension), order="F")
    covariance = (covariance + covariance.T) / 2.0
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
        raise ArithmeticError("The stationary covariance is not positive semidefinite.")
    return covariance


def _sum_expressions(expressions: Sequence[ExpressionNode]) -> ExpressionNode:
    if not expressions:
        raise ValueError("At least one expression is required.")
    result = expressions[0]
    for expression in expressions[1:]:
        result = Add(result, expression)
    return result


def _canonicalize_bound_expression(
    expression: ExpressionNode,
    parameter_values: dict[str, float],
    lag_values: dict[str, int],
) -> tuple[ExpressionNode, MechanismBinding]:
    canonical = canonicalize_commutative(expression)
    rebound_parameters: dict[str, float] = {}
    rebound_lags: dict[str, int] = {}
    parameter_index = 0
    lag_index = 0

    def visit(node: ExpressionNode) -> ExpressionNode:
        nonlocal parameter_index, lag_index
        if isinstance(node, Parameter):
            slot = f"P{parameter_index}"
            parameter_index += 1
            rebound_parameters[slot] = float(parameter_values[node.slot])
            return Parameter(slot)
        if isinstance(node, Lag):
            slot = f"L{lag_index}"
            lag_index += 1
            rebound_lags[slot] = int(lag_values[node.slot])
            return Lag(node.variable, slot)
        if isinstance(node, Add):
            return Add(visit(node.left), visit(node.right))
        if isinstance(node, Multiply):
            return Multiply(visit(node.left), visit(node.right))
        raise TypeError(type(node))

    rebound_expression = visit(canonical)
    return rebound_expression, MechanismBinding(rebound_parameters, rebound_lags)


def _build_ar_expression(
    coefficients: np.ndarray,
    stationary_mean: float,
) -> tuple[ExpressionNode, MechanismBinding]:
    intercept = float(stationary_mean * (1.0 - coefficients.sum()))
    parameter_values = {"intercept": intercept}
    lag_values: dict[str, int] = {}
    terms: list[ExpressionNode] = [Parameter("intercept")]
    for index, coefficient in enumerate(coefficients, start=1):
        parameter_slot = f"coefficient_{index}"
        lag_slot = f"lag_{index}"
        parameter_values[parameter_slot] = float(coefficient)
        lag_values[lag_slot] = index
        terms.append(
            Multiply(Parameter(parameter_slot), Lag(variable=0, slot=lag_slot))
        )
    return _canonicalize_bound_expression(
        _sum_expressions(terms), parameter_values, lag_values
    )


def _build_var1_expressions(
    coefficient_matrix: np.ndarray,
    stationary_mean: np.ndarray,
) -> tuple[tuple[ExpressionNode, ...], tuple[MechanismBinding, ...]]:
    intercept = (np.eye(coefficient_matrix.shape[0]) - coefficient_matrix) @ stationary_mean
    expressions: list[ExpressionNode] = []
    bindings: list[MechanismBinding] = []
    for target, row in enumerate(coefficient_matrix):
        parameter_values = {"intercept": float(intercept[target])}
        lag_values: dict[str, int] = {}
        terms: list[ExpressionNode] = [Parameter("intercept")]
        for source, coefficient in enumerate(row):
            if coefficient == 0.0:
                continue
            parameter_slot = f"coefficient_{source}"
            lag_slot = f"source_{source}"
            parameter_values[parameter_slot] = float(coefficient)
            lag_values[lag_slot] = 1
            terms.append(
                Multiply(
                    Parameter(parameter_slot),
                    Lag(variable=source, slot=lag_slot),
                )
            )
        expression, binding = _canonicalize_bound_expression(
            _sum_expressions(terms), parameter_values, lag_values
        )
        expressions.append(expression)
        bindings.append(binding)
    return tuple(expressions), tuple(bindings)


def build_stationary_ar_mechanism(
    partial_autocorrelations: Sequence[float],
    *,
    stationary_mean: float = 0.0,
    innovation_std: float = 1.0,
    seed: int = 0,
    sample_index: int | None = None,
) -> StationaryARMechanism:
    """Build and certify an AR mechanism from admissible PACF coefficients."""

    if not np.isfinite(stationary_mean):
        raise ValueError("stationary_mean must be finite.")
    if not np.isfinite(innovation_std) or innovation_std <= 0.0:
        raise ValueError("innovation_std must be finite and positive.")

    pacf = np.asarray(partial_autocorrelations, dtype=float)
    coefficients = partial_autocorrelations_to_ar(pacf)
    roots = ar_lag_polynomial_roots(coefficients)
    root_moduli = np.abs(roots)
    minimum_root_modulus = float(np.min(root_moduli))
    certificate = StationarityCertificate(
        family=f"AR({coefficients.size})",
        condition="all roots of 1 - sum(phi_l z^l) lie outside the unit circle",
        metric_name="minimum_lag_polynomial_root_modulus",
        metric_value=minimum_root_modulus,
        threshold=1.0,
        margin=minimum_root_modulus - 1.0,
    )
    if not certificate.holds:
        raise ArithmeticError("PACF conversion failed to produce a stationary AR mechanism.")

    transition = ar_companion_matrix(coefficients)
    noise = np.zeros_like(transition)
    noise[0, 0] = innovation_std**2
    covariance = solve_stationary_covariance(transition, noise)
    expression, binding = _build_ar_expression(coefficients, float(stationary_mean))
    return StationaryARMechanism(
        expression=expression,
        binding=binding,
        partial_autocorrelations=tuple(float(value) for value in pacf),
        coefficients=tuple(float(value) for value in coefficients),
        stationary_mean=float(stationary_mean),
        innovation_std=float(innovation_std),
        stationary_state_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        certificate=certificate,
        seed=int(seed),
        sample_index=sample_index,
    )


def sample_stationary_ar_mechanism(
    config: StationaryARConfig,
    *,
    seed: int,
    sample_index: int | None = None,
) -> StationaryARMechanism:
    """Sample one Gaussian AR mechanism that is stationary by construction."""

    rng = np.random.default_rng(seed)
    order = int(rng.integers(config.minimum_order, config.maximum_order + 1))
    pacf = rng.uniform(
        config.partial_autocorrelation_minimum,
        config.partial_autocorrelation_maximum,
        size=order,
    )
    stationary_mean = float(rng.uniform(config.mean_minimum, config.mean_maximum))
    innovation_std = float(
        rng.uniform(config.innovation_std_minimum, config.innovation_std_maximum)
    )
    return build_stationary_ar_mechanism(
        pacf,
        stationary_mean=stationary_mean,
        innovation_std=innovation_std,
        seed=seed,
        sample_index=sample_index,
    )


def sample_stationary_ar_mechanisms(
    config: StationaryARConfig,
    *,
    root_seed: int,
    count: int,
) -> list[StationaryARMechanism]:
    """Sample an index-stable batch of stationary AR mechanisms."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    return [
        sample_stationary_ar_mechanism(
            config,
            seed=derive_sample_seed(root_seed, sample_index),
            sample_index=sample_index,
        )
        for sample_index in range(count)
    ]


def build_stationary_var1_mechanism(
    coefficient_matrix: Sequence[Sequence[float]],
    *,
    stationary_mean: Sequence[float] | None = None,
    innovation_covariance: Sequence[Sequence[float]] | None = None,
    seed: int = 0,
    sample_index: int | None = None,
) -> StationaryVAR1Mechanism:
    """Build and certify a Gaussian VAR(1) mechanism with spectral radius below one."""

    matrix = np.asarray(coefficient_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("coefficient_matrix must be a non-empty square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("coefficient_matrix must be finite.")
    dimension = matrix.shape[0]
    mean = (
        np.zeros(dimension, dtype=float)
        if stationary_mean is None
        else np.asarray(stationary_mean, dtype=float)
    )
    noise = (
        np.eye(dimension, dtype=float)
        if innovation_covariance is None
        else np.asarray(innovation_covariance, dtype=float)
    )
    if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
        raise ValueError("stationary_mean must contain one finite value per series.")

    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    certificate = StationarityCertificate(
        family=f"VAR({dimension}, 1)",
        condition="the coefficient-matrix spectral radius is below one",
        metric_name="spectral_radius",
        metric_value=spectral_radius,
        threshold=1.0,
        margin=1.0 - spectral_radius,
    )
    if not certificate.holds:
        raise ValueError("coefficient_matrix must have spectral radius below one.")

    covariance = solve_stationary_covariance(matrix, noise)
    expressions, bindings = _build_var1_expressions(matrix, mean)
    return StationaryVAR1Mechanism(
        expressions=expressions,
        bindings=bindings,
        coefficient_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        stationary_mean=tuple(float(value) for value in mean),
        innovation_covariance=tuple(tuple(float(value) for value in row) for row in noise),
        stationary_state_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        certificate=certificate,
        seed=int(seed),
        sample_index=sample_index,
    )


def sample_stationary_var1_mechanism(
    config: StationaryVAR1Config,
    *,
    seed: int,
    sample_index: int | None = None,
) -> StationaryVAR1Mechanism:
    """Sample one sparse Gaussian VAR(1) mechanism that is stationary by construction."""

    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(config.num_series, config.num_series))
    raw *= rng.random(size=raw.shape) < config.edge_probability
    raw_radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
    if raw_radius <= np.finfo(float).eps:
        raw[0, 0] = 1.0
        raw_radius = float(np.max(np.abs(np.linalg.eigvals(raw))))
    target_radius = float(
        rng.uniform(config.spectral_radius_minimum, config.spectral_radius_maximum)
    )
    coefficient_matrix = raw * (target_radius / raw_radius)
    mean = rng.uniform(config.mean_minimum, config.mean_maximum, size=config.num_series)
    innovation_std = rng.uniform(
        config.innovation_std_minimum,
        config.innovation_std_maximum,
        size=config.num_series,
    )
    innovation_covariance = np.diag(innovation_std**2)
    return build_stationary_var1_mechanism(
        coefficient_matrix,
        stationary_mean=mean,
        innovation_covariance=innovation_covariance,
        seed=seed,
        sample_index=sample_index,
    )


def sample_stationary_var1_mechanisms(
    config: StationaryVAR1Config,
    *,
    root_seed: int,
    count: int,
) -> list[StationaryVAR1Mechanism]:
    """Sample an index-stable batch of stationary VAR(1) mechanisms."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    return [
        sample_stationary_var1_mechanism(
            config,
            seed=derive_sample_seed(root_seed, sample_index),
            sample_index=sample_index,
        )
        for sample_index in range(count)
    ]


def sample_stationary_ar_trajectories(
    mechanism: StationaryARMechanism,
    *,
    trajectory_count: int,
    generated_steps: int,
    root_seed: int,
    maximum_absolute_value: float | None = None,
) -> np.ndarray:
    """Sample exact stationary Gaussian AR realizations as ``[trajectory, time]``."""

    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive.")
    if generated_steps < 0:
        raise ValueError("generated_steps must be non-negative.")

    state_covariance = np.asarray(mechanism.stationary_state_covariance, dtype=float)
    state_mean = np.full(mechanism.order, mechanism.stationary_mean, dtype=float)
    root_sequence = np.random.SeedSequence(int(root_seed))
    trajectories: list[np.ndarray] = []
    for trajectory_sequence in root_sequence.spawn(trajectory_count):
        initialization_sequence, innovation_sequence = trajectory_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        newest_to_oldest = initialization_rng.multivariate_normal(
            state_mean,
            state_covariance,
            check_valid="raise",
        )
        innovation_seed = int(
            innovation_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        trajectories.append(
            generate_univariate_trajectory(
                mechanism.expression,
                mechanism.binding,
                initial_values=newest_to_oldest[::-1],
                generated_steps=generated_steps,
                innovation_std=mechanism.innovation_std,
                seed=innovation_seed,
                maximum_absolute_value=maximum_absolute_value,
            )
        )
    return np.stack(trajectories)


def sample_stationary_var1_trajectories(
    mechanism: StationaryVAR1Mechanism,
    *,
    trajectory_count: int,
    generated_steps: int,
    root_seed: int,
    maximum_absolute_value: float | None = None,
) -> np.ndarray:
    """Sample exact stationary Gaussian VAR(1) realizations as ``[trajectory, series, time]``."""

    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive.")
    if generated_steps < 0:
        raise ValueError("generated_steps must be non-negative.")

    state_covariance = np.asarray(mechanism.stationary_state_covariance, dtype=float)
    state_mean = np.asarray(mechanism.stationary_mean, dtype=float)
    innovation_covariance = np.asarray(mechanism.innovation_covariance, dtype=float)

    root_sequence = np.random.SeedSequence(int(root_seed))
    trajectories: list[np.ndarray] = []
    for trajectory_sequence in root_sequence.spawn(trajectory_count):
        initialization_sequence, innovation_sequence = trajectory_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        initial_state = initialization_rng.multivariate_normal(
            state_mean,
            state_covariance,
            check_valid="raise",
        )
        innovation_seed = int(
            innovation_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        trajectories.append(
            generate_multivariate_trajectory(
                mechanism.expressions,
                mechanism.bindings,
                initial_values=initial_state[:, None],
                generated_steps=generated_steps,
                innovation_covariance=innovation_covariance,
                seed=innovation_seed,
                maximum_absolute_value=maximum_absolute_value,
            )
        )
    return np.stack(trajectories)
