"""Nonlinear recurrent AST generation certified by contraction on a bounded domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from tsfmxai.generators.random_expression import (
    RandomTreeConfig,
    SampledMechanism,
    derive_sample_seed,
    sample_random_mechanism,
)
from tsfmxai.generators.trajectory import evaluate_bound_expression
from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Time,
    format_tree,
    to_bound_formula,
    to_skeleton_formula,
    validate_binding,
    walk,
)


def _default_tree_config() -> RandomTreeConfig:
    return RandomTreeConfig(
        minimum_operators=1,
        maximum_operators=6,
        num_series=1,
        minimum_lag=1,
        maximum_lag=12,
        parameter_probability=0.55,
        time_probability=0.0,
        lag_probability=0.45,
        add_probability=0.80,
        minimum_parameter_magnitude=0.05,
        maximum_parameter_magnitude=0.80,
        require_lag=True,
        require_time=False,
        coefficient_gate_lags=True,
    )


@dataclass(frozen=True)
class ContractionTreeConfig:
    """Configuration for a nonlinear AST certified on a bounded state domain."""

    tree_config: RandomTreeConfig = field(default_factory=_default_tree_config)
    domain_radius: float = 1.0
    innovation_bound: float = 0.05
    lag_weight_growth: float = 1.25
    target_lipschitz: float = 0.90
    scale_safety_factor: float = 0.95
    minimum_scale: float = 1e-10

    def __post_init__(self) -> None:
        if self.tree_config.num_series != 1:
            raise ValueError("ContractionTreeConfig currently supports one series.")
        if self.tree_config.time_probability != 0.0 or self.tree_config.require_time:
            raise ValueError("A time terminal is incompatible with the invariant state box.")
        if not self.tree_config.require_lag:
            raise ValueError("The contraction generator requires at least one lag.")
        if not np.isfinite(self.domain_radius) or self.domain_radius <= 0.0:
            raise ValueError("domain_radius must be finite and positive.")
        if not np.isfinite(self.innovation_bound) or self.innovation_bound < 0.0:
            raise ValueError("innovation_bound must be finite and non-negative.")
        if self.innovation_bound >= self.domain_radius:
            raise ValueError("innovation_bound must be smaller than domain_radius.")
        if not np.isfinite(self.lag_weight_growth) or self.lag_weight_growth <= 1.0:
            raise ValueError("lag_weight_growth must be finite and greater than one.")
        if not 0.0 < self.target_lipschitz < 1.0:
            raise ValueError("target_lipschitz must lie in (0, 1).")
        if 1.0 / self.lag_weight_growth >= self.target_lipschitz:
            raise ValueError(
                "lag_weight_growth must make the shift gain smaller than "
                "target_lipschitz."
            )
        if not 0.0 < self.scale_safety_factor < 1.0:
            raise ValueError("scale_safety_factor must lie in (0, 1).")
        if not np.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive.")


@dataclass(frozen=True)
class ExpressionDomainBound:
    """Interval and componentwise derivative bounds over a state box."""

    lower: float
    upper: float
    derivative_absolute_bounds: tuple[float, ...]

    @property
    def maximum_absolute_value(self) -> float:
        return max(abs(self.lower), abs(self.upper))


@dataclass(frozen=True)
class ContractionCertificate:
    """A one-step contraction and invariant-domain certificate."""

    state_dimension: int
    domain_radius: float
    innovation_bound: float
    state_weights: tuple[float, ...]
    deterministic_output_bound: float
    noisy_output_bound: float
    derivative_absolute_bounds: tuple[float, ...]
    top_row_gain_bound: float
    shift_gain: float
    lipschitz_upper_bound: float
    contraction_margin: float
    domain_margin: float
    singularity_free: bool
    noise_distribution: str

    @property
    def holds(self) -> bool:
        return bool(
            self.singularity_free
            and np.isfinite(self.lipschitz_upper_bound)
            and self.contraction_margin > 0.0
            and self.domain_margin >= -1e-12
        )


@dataclass(frozen=True)
class ContractiveMechanism:
    """One scaled nonlinear recurrence with a bounded-domain certificate."""

    expression: ExpressionNode
    binding: MechanismBinding
    base_expression: ExpressionNode
    base_binding: MechanismBinding
    scale: float
    certificate: ContractionCertificate
    seed: int
    sample_index: int | None

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
        bound = self.certificate.innovation_bound
        return f"Uniform(-{bound:.6g}, {bound:.6g})"


@dataclass(frozen=True)
class ContractionMonteCarloReport:
    """Empirical stress-test results that complement an analytic certificate."""

    trajectory_count: int
    generated_steps: int
    evaluated_state_transitions: int
    domain_violations: int
    certificate_violations: int
    maximum_observed_absolute_value: float
    maximum_observed_local_lipschitz: float
    certified_lipschitz_upper_bound: float

    @property
    def passed(self) -> bool:
        return self.domain_violations == 0 and self.certificate_violations == 0


def _state_dimension(expression: ExpressionNode, binding: MechanismBinding) -> int:
    validate_binding(expression, binding)
    nodes = list(walk(expression))
    if any(isinstance(node, Time) for node in nodes):
        raise ValueError("Time terminals are not supported by the contraction certificate.")
    invalid_sources = sorted(
        {node.variable for node in nodes if isinstance(node, Lag) and node.variable != 0}
    )
    if invalid_sources:
        raise ValueError(
            "The univariate contraction certificate only accepts lags from series zero."
        )
    lags = [binding.lag_values[node.slot] for node in nodes if isinstance(node, Lag)]
    if not lags:
        raise ValueError("The contraction certificate requires at least one lag.")
    return max(lags)


def _interval_product(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def bound_expression_on_box(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    domain_radius: float,
) -> ExpressionDomainBound:
    """Bound an AST value and its gradient on ``[-radius, radius]^L``."""

    if not np.isfinite(domain_radius) or domain_radius <= 0.0:
        raise ValueError("domain_radius must be finite and positive.")
    dimension = _state_dimension(expression, binding)

    def visit(node: ExpressionNode) -> tuple[float, float, np.ndarray]:
        if isinstance(node, Parameter):
            value = float(binding.parameter_values[node.slot])
            if not np.isfinite(value):
                raise ValueError("Parameter values must be finite.")
            return value, value, np.zeros(dimension, dtype=float)
        if isinstance(node, Lag):
            gradient = np.zeros(dimension, dtype=float)
            gradient[binding.lag_values[node.slot] - 1] = 1.0
            return -domain_radius, domain_radius, gradient
        if isinstance(node, Time):
            raise ValueError("Time terminals are not supported by the contraction certificate.")
        left_low, left_high, left_gradient = visit(node.left)
        right_low, right_high, right_gradient = visit(node.right)
        if isinstance(node, Add):
            return (
                left_low + right_low,
                left_high + right_high,
                left_gradient + right_gradient,
            )
        product_low, product_high = _interval_product(
            (left_low, left_high),
            (right_low, right_high),
        )
        left_abs = max(abs(left_low), abs(left_high))
        right_abs = max(abs(right_low), abs(right_high))
        gradient = right_abs * left_gradient + left_abs * right_gradient
        return product_low, product_high, gradient

    lower, upper, derivative_bounds = visit(expression)
    if not (
        np.isfinite(lower)
        and np.isfinite(upper)
        and np.all(np.isfinite(derivative_bounds))
    ):
        raise ArithmeticError("Interval propagation produced a non-finite bound.")
    return ExpressionDomainBound(
        lower=float(lower),
        upper=float(upper),
        derivative_absolute_bounds=tuple(float(value) for value in derivative_bounds),
    )


def certify_contractive_expression(
    expression: ExpressionNode,
    binding: MechanismBinding,
    *,
    domain_radius: float,
    innovation_bound: float,
    lag_weight_growth: float,
) -> ContractionCertificate:
    """Certify an augmented lag-state map in a weighted infinity norm."""

    if not np.isfinite(innovation_bound) or innovation_bound < 0.0:
        raise ValueError("innovation_bound must be finite and non-negative.")
    if not np.isfinite(lag_weight_growth) or lag_weight_growth <= 1.0:
        raise ValueError("lag_weight_growth must be finite and greater than one.")

    bounds = bound_expression_on_box(
        expression,
        binding,
        domain_radius=domain_radius,
    )
    dimension = len(bounds.derivative_absolute_bounds)
    weights = lag_weight_growth ** np.arange(dimension, dtype=float)
    derivative_bounds = np.asarray(bounds.derivative_absolute_bounds)
    top_row_gain = float(np.dot(derivative_bounds, weights))
    shift_gain = 0.0 if dimension == 1 else float(1.0 / lag_weight_growth)
    lipschitz_bound = max(top_row_gain, shift_gain)
    deterministic_bound = bounds.maximum_absolute_value
    noisy_bound = deterministic_bound + innovation_bound
    return ContractionCertificate(
        state_dimension=dimension,
        domain_radius=float(domain_radius),
        innovation_bound=float(innovation_bound),
        state_weights=tuple(float(value) for value in weights),
        deterministic_output_bound=deterministic_bound,
        noisy_output_bound=noisy_bound,
        derivative_absolute_bounds=bounds.derivative_absolute_bounds,
        top_row_gain_bound=top_row_gain,
        shift_gain=shift_gain,
        lipschitz_upper_bound=lipschitz_bound,
        contraction_margin=1.0 - lipschitz_bound,
        domain_margin=float(domain_radius - noisy_bound),
        singularity_free=True,
        noise_distribution=f"Uniform(-{innovation_bound:.6g}, {innovation_bound:.6g})",
    )


def _unique_scale_slot(binding: MechanismBinding) -> str:
    base = "CONTRACTION_SCALE"
    slot = base
    index = 0
    while slot in binding.parameter_values:
        index += 1
        slot = f"{base}_{index}"
    return slot


def build_contractive_mechanism(
    base_expression: ExpressionNode,
    base_binding: MechanismBinding,
    config: ContractionTreeConfig,
    *,
    seed: int = 0,
    sample_index: int | None = None,
) -> ContractiveMechanism:
    """Scale one sampled AST until contraction and domain conditions both hold."""

    base_bound = bound_expression_on_box(
        base_expression,
        base_binding,
        domain_radius=config.domain_radius,
    )
    dimension = len(base_bound.derivative_absolute_bounds)
    weights = config.lag_weight_growth ** np.arange(dimension, dtype=float)
    base_top_gain = float(
        np.dot(np.asarray(base_bound.derivative_absolute_bounds), weights)
    )
    contraction_scale_limit = (
        np.inf
        if base_top_gain == 0.0
        else config.target_lipschitz / base_top_gain
    )
    available_output = config.domain_radius - config.innovation_bound
    domain_scale_limit = (
        np.inf
        if base_bound.maximum_absolute_value == 0.0
        else available_output / base_bound.maximum_absolute_value
    )
    scale = config.scale_safety_factor * min(
        1.0,
        contraction_scale_limit,
        domain_scale_limit,
    )
    if not np.isfinite(scale) or scale < config.minimum_scale:
        raise ArithmeticError("The sampled AST requires a scale below minimum_scale.")

    scale_slot = _unique_scale_slot(base_binding)
    expression = Multiply(Parameter(scale_slot), base_expression)
    binding = MechanismBinding(
        parameter_values={**base_binding.parameter_values, scale_slot: float(scale)},
        lag_values=dict(base_binding.lag_values),
    )
    certificate = certify_contractive_expression(
        expression,
        binding,
        domain_radius=config.domain_radius,
        innovation_bound=config.innovation_bound,
        lag_weight_growth=config.lag_weight_growth,
    )
    if not certificate.holds:
        raise ArithmeticError("Scaling failed to produce a valid contraction certificate.")
    if certificate.lipschitz_upper_bound > config.target_lipschitz + 1e-12:
        raise ArithmeticError("The certificate exceeds target_lipschitz.")
    return ContractiveMechanism(
        expression=expression,
        binding=binding,
        base_expression=base_expression,
        base_binding=base_binding,
        scale=float(scale),
        certificate=certificate,
        seed=int(seed),
        sample_index=sample_index,
    )


def sample_contractive_mechanism(
    config: ContractionTreeConfig,
    *,
    seed: int,
    sample_index: int | None = None,
) -> ContractiveMechanism:
    """Sample and certify one nonlinear contractive AST mechanism."""

    base: SampledMechanism = sample_random_mechanism(
        config.tree_config,
        seed=seed,
        sample_index=sample_index,
    )
    return build_contractive_mechanism(
        base.expression,
        base.binding,
        config,
        seed=seed,
        sample_index=sample_index,
    )


def sample_contractive_mechanisms(
    config: ContractionTreeConfig,
    *,
    root_seed: int,
    count: int,
) -> list[ContractiveMechanism]:
    """Sample an index-stable batch of certified nonlinear mechanisms."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    return [
        sample_contractive_mechanism(
            config,
            seed=derive_sample_seed(root_seed, sample_index),
            sample_index=sample_index,
        )
        for sample_index in range(count)
    ]


def evaluate_expression_with_gradient(
    expression: ExpressionNode,
    binding: MechanismBinding,
    state_newest_to_oldest: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Evaluate an AST and its exact gradient with respect to augmented lag state."""

    dimension = _state_dimension(expression, binding)
    state = np.asarray(state_newest_to_oldest, dtype=float)
    if state.shape != (dimension,) or not np.all(np.isfinite(state)):
        raise ValueError("state_newest_to_oldest must match the finite state dimension.")

    def visit(node: ExpressionNode) -> tuple[float, np.ndarray]:
        if isinstance(node, Parameter):
            return float(binding.parameter_values[node.slot]), np.zeros(dimension)
        if isinstance(node, Lag):
            coordinate = binding.lag_values[node.slot] - 1
            gradient = np.zeros(dimension)
            gradient[coordinate] = 1.0
            return float(state[coordinate]), gradient
        if isinstance(node, Time):
            raise ValueError("Time terminals are not supported by the contraction certificate.")
        left_value, left_gradient = visit(node.left)
        right_value, right_gradient = visit(node.right)
        if isinstance(node, Add):
            return left_value + right_value, left_gradient + right_gradient
        return (
            left_value * right_value,
            right_value * left_gradient + left_value * right_gradient,
        )

    return visit(expression)


def augmented_jacobian_norm(
    mechanism: ContractiveMechanism,
    state_newest_to_oldest: Sequence[float],
) -> float:
    """Evaluate the local weighted-infinity norm of the augmented Jacobian."""

    _, gradient = evaluate_expression_with_gradient(
        mechanism.expression,
        mechanism.binding,
        state_newest_to_oldest,
    )
    weights = np.asarray(mechanism.certificate.state_weights)
    top_row_gain = float(np.dot(np.abs(gradient), weights))
    shift_gain = (
        0.0
        if weights.size == 1
        else float(np.max(weights[:-1] / weights[1:]))
    )
    return max(top_row_gain, shift_gain)


def _trajectory_sequences(root_seed: int, trajectory_count: int) -> list[np.random.SeedSequence]:
    if trajectory_count < 1:
        raise ValueError("trajectory_count must be positive.")
    return np.random.SeedSequence(int(root_seed)).spawn(trajectory_count)


def sample_contractive_trajectories(
    mechanism: ContractiveMechanism,
    *,
    trajectory_count: int,
    generated_steps: int,
    root_seed: int,
) -> np.ndarray:
    """Generate trajectories while enforcing the certified domain and noise law."""

    if generated_steps < 0:
        raise ValueError("generated_steps must be non-negative.")
    dimension = mechanism.certificate.state_dimension
    radius = mechanism.certificate.domain_radius
    noise_bound = mechanism.certificate.innovation_bound
    trajectories: list[np.ndarray] = []
    for trajectory_sequence in _trajectory_sequences(root_seed, trajectory_count):
        initialization_sequence, innovation_sequence = trajectory_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        innovation_rng = np.random.default_rng(innovation_sequence)
        values = list(initialization_rng.uniform(-radius, radius, size=dimension))
        innovations = innovation_rng.uniform(
            -noise_bound,
            noise_bound,
            size=generated_steps,
        )
        for step, innovation in enumerate(innovations):
            next_value = evaluate_bound_expression(
                mechanism.expression,
                mechanism.binding,
                time=dimension + step,
                histories={0: values},
            ) + float(innovation)
            if not np.isfinite(next_value) or abs(next_value) > radius + 1e-12:
                raise ArithmeticError("A certified trajectory left its invariant domain.")
            values.append(float(next_value))
        trajectories.append(np.asarray(values))
    return np.stack(trajectories)


def validate_contraction_monte_carlo(
    mechanism: ContractiveMechanism,
    *,
    trajectory_count: int,
    generated_steps: int,
    root_seed: int,
) -> ContractionMonteCarloReport:
    """Stress-test domain invariance and local Jacobians on sampled trajectories."""

    if generated_steps < 1:
        raise ValueError("generated_steps must be positive.")
    dimension = mechanism.certificate.state_dimension
    radius = mechanism.certificate.domain_radius
    noise_bound = mechanism.certificate.innovation_bound
    maximum_value = 0.0
    maximum_lipschitz = 0.0
    domain_violations = 0
    certificate_violations = 0

    for trajectory_sequence in _trajectory_sequences(root_seed, trajectory_count):
        initialization_sequence, innovation_sequence = trajectory_sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        innovation_rng = np.random.default_rng(innovation_sequence)
        values = list(initialization_rng.uniform(-radius, radius, size=dimension))
        maximum_value = max(maximum_value, float(np.max(np.abs(values))))
        innovations = innovation_rng.uniform(
            -noise_bound,
            noise_bound,
            size=generated_steps,
        )
        for step, innovation in enumerate(innovations):
            state = np.asarray(values[-dimension:][::-1])
            local_lipschitz = augmented_jacobian_norm(mechanism, state)
            maximum_lipschitz = max(maximum_lipschitz, local_lipschitz)
            if local_lipschitz > mechanism.certificate.lipschitz_upper_bound + 1e-10:
                certificate_violations += 1
            next_value = evaluate_bound_expression(
                mechanism.expression,
                mechanism.binding,
                time=dimension + step,
                histories={0: values},
            ) + float(innovation)
            maximum_value = max(maximum_value, abs(float(next_value)))
            if not np.isfinite(next_value) or abs(next_value) > radius + 1e-12:
                domain_violations += 1
            values.append(float(next_value))

    return ContractionMonteCarloReport(
        trajectory_count=trajectory_count,
        generated_steps=generated_steps,
        evaluated_state_transitions=trajectory_count * generated_steps,
        domain_violations=domain_violations,
        certificate_violations=certificate_violations,
        maximum_observed_absolute_value=maximum_value,
        maximum_observed_local_lipschitz=maximum_lipschitz,
        certified_lipschitz_upper_bound=mechanism.certificate.lipschitz_upper_bound,
    )
