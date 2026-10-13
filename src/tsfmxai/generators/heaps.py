"""Stationary Gaussian VAR(p) using Heaps's symmetric reparameterization.

Independent implementation of Heaps (2023), Appendix pp. 11--12 of
https://arxiv.org/pdf/2004.09455 and supplement S1.3, pp. 7--8.
Matrix order and symmetric roots follow those equations; no inference engine
is required to sample the prior and simulate its resulting mechanism.
"""

from dataclasses import dataclass

import numpy as np

from tsfmxai.generators.random_expression import derive_sample_seed
from tsfmxai.generators.stochastic import simulate_innovation_path
from tsfmxai.generators.trajectory import DivergentTrajectoryError
from tsfmxai.representations.expression import (
    Add, ExpressionNode, Lag, MechanismBinding, Multiply, Parameter, format_tree,
    to_bound_formula,
)


def _sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2


def symmetric_positive_root(matrix: np.ndarray) -> np.ndarray:
    """Symmetric positive root, preserving the paper's coordinate convention."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("Expected a finite square matrix.")
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Expected a symmetric positive-definite matrix.")
    values, vectors = np.linalg.eigh(_sym(a))
    if np.any(values <= 0):
        raise ValueError("Expected a positive-definite matrix.")
    return _sym((vectors * np.sqrt(values)) @ vectors.T)


def unconstrained_to_partial_autocorrelations(matrices: np.ndarray) -> np.ndarray:
    """Heaps Eq. (3): P = (I + U U.T)^(-1/2) U, implemented via its SVD."""
    u = np.asarray(matrices, dtype=float)
    if (u.ndim != 3 or min(u.shape) < 1 or u.shape[1] != u.shape[2]
            or not np.all(np.isfinite(u))):
        raise ValueError("matrices must be finite [order, series, series].")
    left, singular, right = np.linalg.svd(u, full_matrices=False)
    return (left * (singular / np.hypot(1., singular))[:, None, :]) @ right


def partial_autocorrelations_to_var(
    partials: np.ndarray, innovation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse map to coefficients and H[k] = Cov(x[t], x[t-k]).

    H[0..p] uses future-first ordering, the transpose of Heaps's Gamma[k].
    The reverse covariance recursion is S1.3, Eq. S31 and its symmetric-root
    solution. The forward/backward coefficient updates are Appendix step 2.
    """
    pmat = np.asarray(partials, dtype=float)
    if (pmat.ndim != 3 or min(pmat.shape) < 1 or pmat.shape[1] != pmat.shape[2]
            or not np.all(np.isfinite(pmat))):
        raise ValueError("partials must be finite [order, series, series].")
    p, n, _ = pmat.shape
    sigma = np.asarray(innovation_covariance, dtype=float)
    if sigma.shape != (n, n):
        raise ValueError("Innovation covariance shape must match the series count.")
    symmetric_positive_root(sigma)
    if np.any(np.linalg.svd(pmat, compute_uv=False) >= 1.):
        raise ValueError("Partial-autocorrelation singular values must be below one.")
    forward_cov = [np.zeros_like(sigma) for _ in range(p + 1)]
    forward_roots = [np.zeros_like(sigma) for _ in range(p + 1)]
    forward_cov[p] = sigma
    forward_roots[p] = symmetric_positive_root(sigma)
    for s in range(p - 1, -1, -1):
        root = symmetric_positive_root(np.eye(n) - pmat[s] @ pmat[s].T)
        middle = symmetric_positive_root(_sym(root @ forward_cov[s + 1] @ root))
        # R^-1 middle R^-1, using solves rather than explicit inversion.
        value = np.linalg.solve(root, middle)
        forward_roots[s] = _sym(np.linalg.solve(root, value.T).T)
        forward_cov[s] = _sym(forward_roots[s] @ forward_roots[s].T)
    reverse_cov = forward_cov[0].copy()
    forward_coeff = np.empty((0, n, n))
    reverse_coeff = np.empty((0, n, n))
    h = [forward_cov[0]]
    for s in range(p):
        sf = forward_roots[s]
        sr = symmetric_positive_root(reverse_cov)
        last_forward = np.linalg.solve(sr, (sf @ pmat[s]).T).T
        last_reverse = np.linalg.solve(sf, (sr @ pmat[s].T).T).T
        next_h = last_forward @ reverse_cov
        for j in range(s):
            next_h = next_h + forward_coeff[j] @ h[s - j]
        h.append(next_h)
        updated_forward = np.empty((s + 1, n, n))
        updated_reverse = np.empty_like(updated_forward)
        for j in range(s):
            updated_forward[j] = forward_coeff[j] - last_forward @ reverse_coeff[s - j - 1]
            updated_reverse[j] = reverse_coeff[j] - last_reverse @ forward_coeff[s - j - 1]
        updated_forward[s] = last_forward
        updated_reverse[s] = last_reverse
        reverse_cov = _sym(reverse_cov - last_reverse @ forward_cov[s] @ last_reverse.T)
        forward_coeff, reverse_coeff = updated_forward, updated_reverse
    return forward_coeff, np.asarray(h)


def var_companion_matrix(coefficients: np.ndarray) -> np.ndarray:
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 3 or min(a.shape) < 1 or a.shape[1] != a.shape[2]:
        raise ValueError("coefficients must have shape [order, series, series].")
    p, n, _ = a.shape
    result = np.zeros((p * n, p * n))
    result[:n] = np.concatenate(a, axis=1)
    if p > 1:
        result[n:, :-n] = np.eye(n * (p - 1))
    return result


@dataclass(frozen=True)
class HeapsVARConfig:
    num_series: int = 1
    minimum_order: int = 1
    maximum_order: int = 12
    unconstrained_std: float = .45
    innovation_std: float = .05
    mean_minimum: float = 0.
    mean_maximum: float = 0.

    def __post_init__(self) -> None:
        if self.num_series < 1 or self.minimum_order < 1:
            raise ValueError("num_series and minimum_order must be positive.")
        if self.maximum_order < self.minimum_order:
            raise ValueError("maximum_order must be >= minimum_order.")
        if not np.isfinite(self.unconstrained_std) or self.unconstrained_std <= 0:
            raise ValueError("unconstrained_std must be finite and positive.")
        if not np.isfinite(self.innovation_std) or self.innovation_std <= 0:
            raise ValueError("innovation_std must be finite and positive.")
        if not np.isfinite([self.mean_minimum, self.mean_maximum]).all():
            raise ValueError("Mean bounds must be finite.")
        if self.mean_minimum > self.mean_maximum:
            raise ValueError("Mean bounds must be ordered.")


@dataclass(frozen=True)
class HeapsCertificate:
    spectral_radius: float
    maximum_partial_singular_value: float
    minimum_state_covariance_eigenvalue: float
    relative_lyapunov_residual: float

    @property
    def holds(self) -> bool:
        return bool(0 <= self.spectral_radius < 1
                    and self.maximum_partial_singular_value < 1
                    and self.minimum_state_covariance_eigenvalue > 0
                    and self.relative_lyapunov_residual < 1e-8)


def _nested_tuple(a: np.ndarray) -> tuple:
    return tuple(_nested_tuple(row) if isinstance(row, np.ndarray) else float(row)
                 for row in a)


@dataclass(frozen=True)
class HeapsVARMechanism:
    expressions: tuple[ExpressionNode, ...]
    bindings: tuple[MechanismBinding, ...]
    coefficients: tuple
    unconstrained_matrices: tuple
    partial_autocorrelations: tuple
    innovation_covariance: tuple
    stationary_mean: tuple[float, ...]
    stationary_state_covariance: tuple
    stationary_state_factor: tuple
    certificate: HeapsCertificate
    seed: int
    sample_index: int | None

    @property
    def order(self) -> int:
        return len(self.coefficients)

    @property
    def num_series(self) -> int:
        return len(self.expressions)

    @property
    def formulas(self) -> tuple[str, ...]:
        return tuple(f"x_{j}(t) = {to_bound_formula(e, b)} + epsilon_{j}(t)"
                     for j, (e, b) in enumerate(zip(self.expressions, self.bindings)))

    @property
    def trees(self) -> tuple[str, ...]:
        return tuple(format_tree(e) for e in self.expressions)


def build_heaps_var_mechanism(
    unconstrained_matrices: np.ndarray, innovation_covariance: np.ndarray, *,
    stationary_mean: np.ndarray | None = None, seed: int = 0,
    sample_index: int | None = None,
) -> HeapsVARMechanism:
    partials = unconstrained_to_partial_autocorrelations(unconstrained_matrices)
    coefficients, autocov = partial_autocorrelations_to_var(partials, innovation_covariance)
    p, n, _ = coefficients.shape
    mean = np.zeros(n) if stationary_mean is None else np.asarray(stationary_mean, dtype=float)
    if mean.shape != (n,) or not np.all(np.isfinite(mean)):
        raise ValueError("stationary_mean must be a finite vector of num_series values.")
    # State order: newest lag block first, unlike chronological observations.
    covariance = np.block([[autocov[j - i] if j >= i else autocov[i - j].T
                            for j in range(p)] for i in range(p)])
    covariance = _sym(covariance)
    factor = np.linalg.cholesky(covariance)
    companion = var_companion_matrix(coefficients)
    q = np.zeros_like(covariance)
    q[:n, :n] = innovation_covariance
    residual = covariance - companion @ covariance @ companion.T - q
    certificate = HeapsCertificate(
        float(np.max(np.abs(np.linalg.eigvals(companion)))),
        float(np.max(np.linalg.svd(partials, compute_uv=False))),
        float(np.linalg.eigvalsh(covariance)[0]),
        float(np.linalg.norm(residual) / max(np.linalg.norm(covariance), np.finfo(float).tiny)),
    )
    if not certificate.holds:
        raise ArithmeticError("Heaps reconstruction failed its independent numerical certificate.")
    intercept = mean - coefficients.sum(axis=0) @ mean
    trees, bindings = [], []
    for target in range(n):
        tree: ExpressionNode = Parameter("offset")
        parameters = {"offset": float(intercept[target])}
        lags = {}
        for lag in range(p):
            for source in range(n):
                value = float(coefficients[lag, target, source])
                if value == 0:
                    continue
                slot = f"C_{source}_{lag + 1}"
                lag_slot = f"L_{source}_{lag + 1}"
                parameters[slot] = value
                lags[lag_slot] = lag + 1
                tree = Add(tree, Multiply(Parameter(slot), Lag(source, lag_slot)))
        trees.append(tree)
        bindings.append(MechanismBinding(parameters, lags))
    return HeapsVARMechanism(
        tuple(trees), tuple(bindings), _nested_tuple(coefficients),
        _nested_tuple(np.asarray(unconstrained_matrices)), _nested_tuple(partials),
        _nested_tuple(np.asarray(innovation_covariance)), tuple(mean),
        _nested_tuple(covariance), _nested_tuple(factor), certificate, int(seed), sample_index,
    )


def sample_heaps_var_mechanism(
    config: HeapsVARConfig, *, seed: int, sample_index: int | None = None,
) -> HeapsVARMechanism:
    rng = np.random.default_rng(seed)
    order = int(rng.integers(config.minimum_order, config.maximum_order + 1))
    # This Gaussian sampling law is a declared project prior, not a fitted posterior.
    matrices = rng.normal(0, config.unconstrained_std,
                          size=(order, config.num_series, config.num_series))
    mean = rng.uniform(config.mean_minimum, config.mean_maximum, size=config.num_series)
    return build_heaps_var_mechanism(
        matrices, np.eye(config.num_series) * config.innovation_std**2,
        stationary_mean=mean, seed=seed, sample_index=sample_index,
    )


def sample_heaps_var_mechanisms(
    config: HeapsVARConfig, *, root_seed: int, count: int,
) -> list[HeapsVARMechanism]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    return [sample_heaps_var_mechanism(config, seed=derive_sample_seed(root_seed, i),
                                     sample_index=i) for i in range(count)]


def initialize_heaps_history(mechanism: HeapsVARMechanism, *, seed: int) -> np.ndarray:
    """Draw joint invariant Gaussian lag history, returned chronologically."""
    rng = np.random.default_rng(seed)
    state = (np.tile(mechanism.stationary_mean, mechanism.order)
             + np.asarray(mechanism.stationary_state_factor)
             @ rng.standard_normal(mechanism.order * mechanism.num_series))
    return state.reshape(mechanism.order, mechanism.num_series)[::-1].T


def sample_heaps_var_trajectories(
    mechanism: HeapsVARMechanism, *, trajectory_count: int, generated_steps: int,
    root_seed: int,
) -> np.ndarray:
    """Return [trajectory, series, stationary history + generated steps]."""
    if trajectory_count < 1 or generated_steps < 0:
        raise ValueError("trajectory_count must be positive; generated_steps non-negative.")
    result = []
    noise_factor = np.linalg.cholesky(mechanism.innovation_covariance)
    for i in range(trajectory_count):
        init_seq, noise_seq = np.random.SeedSequence([root_seed, i]).spawn(2)
        history = initialize_heaps_history(
            mechanism, seed=int(init_seq.generate_state(1, dtype=np.uint64)[0]))
        noise = noise_factor @ np.random.default_rng(noise_seq).standard_normal(
            (mechanism.num_series, generated_steps))
        generated = simulate_innovation_path(mechanism.expressions, mechanism.bindings,
                                             history, noise)
        if generated.shape[1] != generated_steps or not np.all(np.isfinite(generated)):
            raise DivergentTrajectoryError("Non-finite Heaps VAR trajectory.")
        result.append(np.concatenate((history, generated), axis=1))
    return np.stack(result)
