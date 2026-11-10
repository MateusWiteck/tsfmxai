"""Product-lag saturation with a Shao--Wu GMC certificate.

Reference: Theorem 5.1, https://arxiv.org/pdf/math/0611029, pp. 11--12.
The product-lag rule specializes Shao--Wu; the random AST grammar and
finite-history initialization are project adaptations.
"""

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from tsfmxai.generators.random_expression import (
    RandomTreeConfig, derive_sample_seed, sample_random_mechanism,
)
from tsfmxai.generators.stochastic import simulate_innovation_path
from tsfmxai.generators.trajectory import DivergentTrajectoryError, evaluate_bound_expression
from tsfmxai.representations.expression import (
    Add, ExpressionNode, Lag, MechanismBinding, Multiply, Parameter, Tanh,
    format_tree, to_bound_formula, validate_binding, walk,
)


def _tree_default() -> RandomTreeConfig:
    return RandomTreeConfig(parameter_probability=.55, lag_probability=.45,
                            add_probability=.8)


@dataclass(frozen=True)
class ShaoWuConfig:
    """Configure product-lag saturation and its GMC verification."""

    tree_config: RandomTreeConfig = field(default_factory=_tree_default)
    target_gain: float = .8
    safety_factor: float = .95
    innovation_std: float = .05
    initial_radius: float = 1.

    def __post_init__(self) -> None:
        if not (0 < self.target_gain < 1 and 0 < self.safety_factor < 1):
            raise ValueError("target_gain and safety_factor must lie inside (0, 1).")
        for name in ("innovation_std", "initial_radius"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class ShaoWuCertificate:
    """L2 lag-block bounds use Euclidean norms, alpha=2 and alpha'=1."""

    lag_gains: tuple[float, ...]
    total_gain: float
    drift_norm_bound: float | None
    anchor_l2: float
    innovation_l2: float

    @property
    def guarantee(self) -> str:
        return "geometric_moment_contraction"

    @property
    def holds(self) -> bool:
        return bool(
            np.isfinite(self.total_gain) and 0 <= self.total_gain < 1
            and (self.drift_norm_bound is None or np.isfinite(self.drift_norm_bound))
            and np.isfinite(self.anchor_l2)
            and np.isfinite(self.innovation_l2)
            and all(np.isfinite(gain) and gain >= 0 for gain in self.lag_gains)
        )


@dataclass(frozen=True)
class ShaoWuMechanism:
    expressions: tuple[ExpressionNode, ...]
    bindings: tuple[MechanismBinding, ...]
    scale: float
    certificate: ShaoWuCertificate
    innovation_std: float
    initial_radius: float
    seed: int
    sample_index: int | None

    @property
    def num_series(self) -> int:
        return len(self.expressions)

    @property
    def order(self) -> int:
        return len(self.certificate.lag_gains)

    @property
    def formulas(self) -> tuple[str, ...]:
        return tuple(f"x_{j}(t) = {to_bound_formula(e, b)} + epsilon_{j}(t)"
                     for j, (e, b) in enumerate(zip(self.expressions, self.bindings)))

    @property
    def trees(self) -> tuple[str, ...]:
        return tuple(format_tree(e) for e in self.expressions)


def global_expression_bounds(
    expression: ExpressionNode, binding: MechanismBinding, *, num_series: int, order: int,
) -> tuple[float, np.ndarray]:
    """Absolute value bound and absolute derivatives [source, lag] on R^(N*L).

    Bounds may be infinite for an unsupported global polynomial. Multiplication
    uses the product rule and explicitly treats a zero derivative as zero,
    including when the other operand's amplitude bound is infinite.
    """

    validate_binding(expression, binding)
    if num_series < 1 or order < 1:
        raise ValueError("num_series and order must be positive.")

    def scaled_derivative(value: float, derivative: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.where(derivative == 0, 0., value * derivative)

    def visit(node: ExpressionNode) -> tuple[float, np.ndarray]:
        zeros = np.zeros((num_series, order))
        if isinstance(node, Parameter):
            value = abs(float(binding.parameter_values[node.slot]))
            if not np.isfinite(value):
                raise ValueError("Coefficients must be finite.")
            return value, zeros
        if isinstance(node, Lag):
            lag = binding.lag_values[node.slot]
            if node.variable >= num_series or lag > order:
                raise ValueError("Lag or variable exceeds the declared state.")
            zeros[node.variable, lag - 1] = 1.
            return np.inf, zeros
        if isinstance(node, Tanh):
            _, derivative = visit(node.operand)
            return 1., derivative
        if not isinstance(node, (Add, Multiply)):
            raise TypeError(type(node))
        a, da = visit(node.left)
        b, db = visit(node.right)
        if isinstance(node, Add):
            return a + b, da + db
        amplitude = 0. if a == 0 or b == 0 else a * b
        return amplitude, scaled_derivative(b, da) + scaled_derivative(a, db)

    return visit(expression)


def saturate_product_lags(node: ExpressionNode) -> ExpressionNode:
    """Wrap lag leaves in tanh only within products of lag-dependent factors.

    A parameter times one lag is linear in the history, so its lag stays bare.
    When both factors of a product depend on lags, every lag within that
    product is wrapped. This intentionally preserves additive bare-lag terms.
    """

    def has_lag(part: ExpressionNode) -> bool:
        return any(isinstance(item, Lag) for item in walk(part))

    def visit(part: ExpressionNode, nonlinear_product: bool) -> ExpressionNode:
        if isinstance(part, Lag):
            return Tanh(part) if nonlinear_product else part
        if isinstance(part, Parameter):
            return part
        if isinstance(part, Tanh):
            return part
        if isinstance(part, Add):
            return Add(visit(part.left, nonlinear_product),
                       visit(part.right, nonlinear_product))
        if isinstance(part, Multiply):
            mark = nonlinear_product or (has_lag(part.left) and has_lag(part.right))
            return Multiply(visit(part.left, mark), visit(part.right, mark))
        raise TypeError(type(part))

    return visit(node, False)


def build_shao_wu_mechanism(
    expressions: tuple[ExpressionNode, ...], bindings: tuple[MechanismBinding, ...],
    config: ShaoWuConfig, *, seed: int = 0, sample_index: int | None = None,
) -> ShaoWuMechanism:
    """Scale a product-lag AST and verify Shao--Wu GMC conditions."""

    n = len(expressions)
    if n < 1 or len(bindings) != n or n != config.tree_config.num_series:
        raise ValueError("One expression and binding per configured series are required.")
    order = max((b.lag_values[node.slot]
                 for e, b in zip(expressions, bindings) for node in walk(e)
                 if isinstance(node, Lag)), default=1)
    bounds = [global_expression_bounds(e, b, num_series=n, order=order)
              for e, b in zip(expressions, bindings)]
    amplitudes = np.array([a for a, _ in bounds])
    derivatives = np.stack([d for _, d in bounds])
    if not np.all(np.isfinite(derivatives)):
        raise ValueError("This constructor requires finite global derivative bounds.")
    gains = np.array([np.linalg.norm(derivatives[:, :, lag], ord=2)
                      for lag in range(order)])
    total = float(gains.sum())
    scale = config.safety_factor * min(1., config.target_gain / total if total else 1.)
    final_expressions, final_bindings = [], []
    for e, b in zip(expressions, bindings):
        slot = "GMC_SCALE"
        while slot in b.parameter_values:
            slot += "_"
        final_expressions.append(Multiply(Parameter(slot), e))
        final_bindings.append(MechanismBinding(
            {**b.parameter_values, slot: scale}, dict(b.lag_values)))
    gains *= scale
    gain = float(gains.sum())
    drift_bound = float(np.linalg.norm(amplitudes * scale)) if np.all(np.isfinite(amplitudes)) else None
    zero_history = np.zeros((n, order))
    anchor_values = np.array([
        evaluate_bound_expression(e, b, time=order, histories=zero_history)
        for e, b in zip(final_expressions, final_bindings)
    ])
    noise_l2 = config.innovation_std * sqrt(n)
    anchor_l2 = float(np.hypot(np.linalg.norm(anchor_values), noise_l2))
    certificate = ShaoWuCertificate(tuple(gains), gain, drift_bound, anchor_l2, noise_l2)
    if not certificate.holds:
        raise ArithmeticError("The selected mathematical verification failed.")
    return ShaoWuMechanism(tuple(final_expressions), tuple(final_bindings), scale,
                           certificate, config.innovation_std, config.initial_radius,
                           int(seed), sample_index)


def sample_shao_wu_mechanism(
    config: ShaoWuConfig, *, seed: int, sample_index: int | None = None,
) -> ShaoWuMechanism:
    trees, bindings = [], []
    for target in range(config.tree_config.num_series):
        child_seed = seed if target == 0 else derive_sample_seed(seed, target)
        base = sample_random_mechanism(config.tree_config, seed=child_seed)
        trees.append(saturate_product_lags(base.expression))
        bindings.append(base.binding)
    return build_shao_wu_mechanism(
        tuple(trees), tuple(bindings), config, seed=seed, sample_index=sample_index)


def sample_shao_wu_mechanisms(
    config: ShaoWuConfig, *, root_seed: int, count: int,
) -> list[ShaoWuMechanism]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    return [sample_shao_wu_mechanism(config, seed=derive_sample_seed(root_seed, i),
                                    sample_index=i) for i in range(count)]


def initialize_shao_wu_history(mechanism: ShaoWuMechanism, *, seed: int) -> np.ndarray:
    """Draw the chronological [series, lag] history directly from a uniform law."""
    return np.random.default_rng(seed).uniform(
        -mechanism.initial_radius, mechanism.initial_radius,
        size=(mechanism.num_series, mechanism.order),
    )


def sample_shao_wu_trajectories(
    mechanism: ShaoWuMechanism, *, trajectory_count: int, generated_steps: int,
    root_seed: int,
) -> np.ndarray:
    """Return [trajectory, series, retained history + generated steps]."""
    if trajectory_count < 1 or generated_steps < 0:
        raise ValueError("trajectory_count must be positive; generated_steps non-negative.")
    result = []
    for i in range(trajectory_count):
        init_seq, noise_seq = np.random.SeedSequence([root_seed, i]).spawn(2)
        history = initialize_shao_wu_history(
            mechanism, seed=int(init_seq.generate_state(1, dtype=np.uint64)[0]))
        noise = np.random.default_rng(noise_seq).normal(
            0., mechanism.innovation_std, size=(mechanism.num_series, generated_steps))
        generated = simulate_innovation_path(mechanism.expressions, mechanism.bindings,
                                             history, noise)
        if generated.shape[1] != generated_steps or not np.all(np.isfinite(generated)):
            raise DivergentTrajectoryError("Non-finite Shao--Wu trajectory.")
        result.append(np.concatenate((history, generated), axis=1))
    return np.stack(result)
