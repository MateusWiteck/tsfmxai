"""Independent VAR checks avoid merely retesting the reconstruction itself."""

import numpy as np
import pytest
from joblib import Parallel, delayed

from tsfmxai.generators import (
    HeapsVARConfig, build_heaps_var_mechanism, derive_sample_seed,
    evaluate_bound_expression, partial_autocorrelations_to_ar,
    partial_autocorrelations_to_var, sample_heaps_var_mechanism,
    sample_heaps_var_trajectories, solve_stationary_covariance,
    unconstrained_to_partial_autocorrelations, var_companion_matrix,
)


@pytest.mark.parametrize("order", [1, 2, 5, 12])
def test_univariate_map_matches_levinson_recursion(order):
    u = np.random.default_rng(order).normal(0, .4, size=(order, 1, 1))
    partials = unconstrained_to_partial_autocorrelations(u)
    coefficients, h = partial_autocorrelations_to_var(partials, np.array([[.01]]))
    np.testing.assert_allclose(coefficients[:, 0, 0],
                               partial_autocorrelations_to_ar(partials[:, 0, 0]), atol=1e-12)
    mechanism = build_heaps_var_mechanism(u, np.array([[.01]]))
    assert mechanism.certificate.holds
    assert h[0, 0, 0] > 0


def test_multivariate_covariance_matches_independent_lyapunov_solution():
    u = np.random.default_rng(18).normal(0, .3, size=(3, 2, 2))
    sigma = np.array([[.05, .012], [.012, .03]])
    mechanism = build_heaps_var_mechanism(u, sigma, stationary_mean=np.array([.5, -.2]))
    b = var_companion_matrix(np.array(mechanism.coefficients))
    q = np.zeros_like(b)
    q[:2, :2] = sigma
    expected = solve_stationary_covariance(b, q)
    np.testing.assert_allclose(mechanism.stationary_state_covariance, expected, atol=1e-12)
    history = np.random.default_rng(33).normal(size=(2, 3))
    prediction = [evaluate_bound_expression(e, bind, time=3, histories=history)
                  for e, bind in zip(mechanism.expressions, mechanism.bindings)]
    mean = np.asarray(mechanism.stationary_mean)
    direct = mean.copy()
    for lag, a in enumerate(mechanism.coefficients):
        direct += np.asarray(a) @ (history[:, -lag - 1] - mean)
    np.testing.assert_allclose(prediction, direct)


def test_symmetric_mapping_preserves_orthogonal_change_of_coordinates():
    rng = np.random.default_rng(19)
    u = rng.normal(0, .25, (3, 2, 2))
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    sigma = np.diag([.02, .05])
    a = build_heaps_var_mechanism(u, sigma)
    b = build_heaps_var_mechanism(q @ u @ q.T, q @ sigma @ q.T)
    np.testing.assert_allclose(b.coefficients, q @ np.asarray(a.coefficients) @ q.T, atol=1e-12)


def test_zero_partials_are_white_noise_and_invalid_covariances_fail():
    mechanism = build_heaps_var_mechanism(np.zeros((3, 2, 2)), np.eye(2))
    np.testing.assert_array_equal(mechanism.coefficients, 0.)
    np.testing.assert_allclose(mechanism.stationary_state_covariance, np.eye(6))
    assert sample_heaps_var_trajectories(mechanism, trajectory_count=2,
                                        generated_steps=4, root_seed=1).shape == (2, 2, 7)
    with pytest.raises(ValueError, match="positive-definite"):
        build_heaps_var_mechanism(np.zeros((1, 2, 2)), np.diag([1., -1.]))
    with pytest.raises(ValueError, match="below one"):
        partial_autocorrelations_to_var(np.ones((1, 1, 1)), np.ones((1, 1)))


def test_joint_initialization_matches_moments_at_initial_and_later_time():
    mechanism = sample_heaps_var_mechanism(
        HeapsVARConfig(num_series=2, minimum_order=2, maximum_order=2), seed=5)
    samples = sample_heaps_var_trajectories(mechanism, trajectory_count=3000,
                                          generated_steps=8, root_seed=71)
    expected = np.asarray(mechanism.stationary_state_covariance)[:2, :2]
    for t in (mechanism.order - 1, samples.shape[-1] - 1):
        np.testing.assert_allclose(np.cov(samples[:, :, t], rowvar=False, bias=True),
                                   expected, atol=.0005, rtol=.1)


def test_heaps_reproducible_across_global_rng_and_jobs():
    cfg = HeapsVARConfig(num_series=2, maximum_order=3)
    seeds = [derive_sample_seed(123, i) for i in range(4)]
    serial = [sample_heaps_var_mechanism(cfg, seed=s) for s in seeds]
    parallel = Parallel(n_jobs=2)(delayed(sample_heaps_var_mechanism)(cfg, seed=s) for s in seeds)
    assert serial == parallel
    a = sample_heaps_var_trajectories(serial[0], trajectory_count=2,
                                    generated_steps=20, root_seed=1)
    np.random.seed(789)
    np.random.normal(size=500)
    b = sample_heaps_var_trajectories(serial[0], trajectory_count=3,
                                    generated_steps=20, root_seed=1)
    np.testing.assert_array_equal(a, b[:2])
