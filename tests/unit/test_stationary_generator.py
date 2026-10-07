import random

import numpy as np
import pytest

from tsfmxai.generators import (
    StationaryARConfig,
    StationaryVAR1Config,
    ar_companion_matrix,
    ar_lag_polynomial_roots,
    build_stationary_ar_mechanism,
    build_stationary_var1_mechanism,
    generate_multivariate_trajectory,
    generate_univariate_trajectory,
    partial_autocorrelations_to_ar,
    sample_stationary_ar_mechanism,
    sample_stationary_ar_mechanisms,
    sample_stationary_ar_trajectories,
    sample_stationary_var1_mechanism,
    sample_stationary_var1_trajectories,
    solve_stationary_covariance,
)


def test_pacf_parameterization_produces_stationary_ar_coefficients() -> None:
    coefficients = partial_autocorrelations_to_ar([0.60, -0.40, 0.30])
    roots = ar_lag_polynomial_roots(coefficients)

    np.testing.assert_allclose(coefficients, [0.96, -0.652, 0.30])
    assert np.min(np.abs(roots)) > 1.0


def test_stationary_ar_mechanism_contains_certificate_ast_and_lyapunov_covariance() -> None:
    mechanism = build_stationary_ar_mechanism(
        [0.55, -0.25],
        stationary_mean=1.5,
        innovation_std=0.3,
        seed=11,
    )

    transition = ar_companion_matrix(mechanism.coefficients)
    covariance = np.asarray(mechanism.stationary_state_covariance)
    noise = np.zeros((mechanism.order, mechanism.order))
    noise[0, 0] = mechanism.innovation_std**2

    assert mechanism.certificate.holds
    assert mechanism.certificate.margin > 0.0
    assert "x_0(t)" in mechanism.formula
    assert "x_0(t-1)" in mechanism.formula
    assert "x_0(t-2)" in mechanism.formula
    np.testing.assert_allclose(
        covariance,
        transition @ covariance @ transition.T + noise,
        atol=1e-12,
    )
    initial_values = np.array([0.25, -0.75])
    generated = generate_univariate_trajectory(
        mechanism.expression,
        mechanism.binding,
        initial_values=initial_values,
        generated_steps=1,
    )
    intercept = mechanism.stationary_mean * (1.0 - sum(mechanism.coefficients))
    expected_next = intercept + np.dot(mechanism.coefficients, initial_values[::-1])
    np.testing.assert_allclose(generated[-1], expected_next)


def test_stationary_ar_sampling_is_index_stable_and_locally_seeded() -> None:
    config = StationaryARConfig(minimum_order=2, maximum_order=4)

    random.seed(1)
    np.random.seed(2)
    first = sample_stationary_ar_mechanisms(config, root_seed=77, count=5)
    random.seed(999)
    np.random.seed(888)
    second = sample_stationary_ar_mechanisms(config, root_seed=77, count=5)

    assert first == second
    assert [mechanism.sample_index for mechanism in first] == list(range(5))
    assert all(mechanism.certificate.holds for mechanism in first)


def test_stationary_ar_trajectory_has_same_distribution_at_start_and_end() -> None:
    mechanism = build_stationary_ar_mechanism(
        [0.65],
        stationary_mean=1.25,
        innovation_std=0.40,
    )
    trajectories = sample_stationary_ar_trajectories(
        mechanism,
        trajectory_count=4_000,
        generated_steps=30,
        root_seed=123,
    )
    expected_variance = np.asarray(mechanism.stationary_state_covariance)[0, 0]

    assert trajectories.shape == (4_000, 31)
    np.testing.assert_allclose(trajectories[:, 0].mean(), 1.25, atol=0.03)
    np.testing.assert_allclose(trajectories[:, -1].mean(), 1.25, atol=0.03)
    np.testing.assert_allclose(trajectories[:, 0].var(), expected_variance, rtol=0.08)
    np.testing.assert_allclose(trajectories[:, -1].var(), expected_variance, rtol=0.08)


def test_stationary_var_mechanism_has_exact_covariance_and_executable_ast() -> None:
    matrix = np.array([[0.55, 0.20], [-0.10, 0.40]])
    mean = np.array([1.0, -0.5])
    innovation_covariance = np.array([[0.09, 0.02], [0.02, 0.16]])
    mechanism = build_stationary_var1_mechanism(
        matrix,
        stationary_mean=mean,
        innovation_covariance=innovation_covariance,
    )
    covariance = np.asarray(mechanism.stationary_state_covariance)

    assert mechanism.certificate.holds
    np.testing.assert_allclose(
        covariance,
        matrix @ covariance @ matrix.T + innovation_covariance,
        atol=1e-12,
    )

    initial_state = np.array([[2.0], [-1.0]])
    generated = generate_multivariate_trajectory(
        mechanism.expressions,
        mechanism.bindings,
        initial_values=initial_state,
        generated_steps=1,
    )
    expected_next = (np.eye(2) - matrix) @ mean + matrix @ initial_state[:, 0]
    np.testing.assert_allclose(generated[:, 1], expected_next)


def test_stationary_var_sampling_and_trajectories_are_reproducible() -> None:
    config = StationaryVAR1Config(
        num_series=3,
        spectral_radius_minimum=0.4,
        spectral_radius_maximum=0.8,
        edge_probability=0.55,
    )
    first = sample_stationary_var1_mechanism(config, seed=2026)
    second = sample_stationary_var1_mechanism(config, seed=2026)
    assert first == second
    assert 0.4 <= first.certificate.metric_value <= 0.8

    arguments = dict(
        trajectory_count=200,
        generated_steps=12,
        root_seed=991,
    )
    first_trajectories = sample_stationary_var1_trajectories(first, **arguments)
    second_trajectories = sample_stationary_var1_trajectories(second, **arguments)
    np.testing.assert_array_equal(first_trajectories, second_trajectories)
    assert first_trajectories.shape == (200, 3, 13)


def test_stationary_var_accepts_positive_semidefinite_innovation_covariance() -> None:
    mechanism = build_stationary_var1_mechanism(
        [[0.4, 0.0], [0.0, 0.2]],
        innovation_covariance=[[0.1, 0.0], [0.0, 0.0]],
    )

    trajectories = sample_stationary_var1_trajectories(
        mechanism,
        trajectory_count=3,
        generated_steps=4,
        root_seed=19,
    )

    assert trajectories.shape == (3, 2, 5)
    np.testing.assert_array_equal(trajectories[:, 1, :], 0.0)


def test_stationary_var_initialization_matches_stationary_moments() -> None:
    mechanism = build_stationary_var1_mechanism(
        [[0.55, 0.15], [-0.10, 0.35]],
        stationary_mean=[0.75, -0.50],
        innovation_covariance=[[0.12, 0.03], [0.03, 0.08]],
    )
    trajectories = sample_stationary_var1_trajectories(
        mechanism,
        trajectory_count=5_000,
        generated_steps=20,
        root_seed=71,
    )
    expected_mean = np.asarray(mechanism.stationary_mean)
    expected_covariance = np.asarray(mechanism.stationary_state_covariance)

    np.testing.assert_allclose(trajectories[:, :, 0].mean(axis=0), expected_mean, atol=0.03)
    np.testing.assert_allclose(trajectories[:, :, -1].mean(axis=0), expected_mean, atol=0.03)
    np.testing.assert_allclose(
        np.cov(trajectories[:, :, 0], rowvar=False, bias=True),
        expected_covariance,
        atol=0.02,
    )
    np.testing.assert_allclose(
        np.cov(trajectories[:, :, -1], rowvar=False, bias=True),
        expected_covariance,
        atol=0.02,
    )


def test_unstable_transitions_and_invalid_pacf_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"lie in \(-1, 1\)"):
        partial_autocorrelations_to_ar([1.0])
    with pytest.raises(ValueError, match="spectral radius"):
        build_stationary_var1_mechanism([[1.01]])
    with pytest.raises(ValueError, match="spectral radius"):
        solve_stationary_covariance([[1.0]], [[1.0]])


def test_configuration_rejects_open_boundary_violations() -> None:
    with pytest.raises(ValueError, match="Partial autocorrelations"):
        StationaryARConfig(partial_autocorrelation_maximum=1.0)
    with pytest.raises(ValueError, match="Spectral radii"):
        StationaryVAR1Config(spectral_radius_maximum=1.0)
    with pytest.raises(ValueError, match="edge_probability"):
        StationaryVAR1Config(edge_probability=0.0)


def test_sampled_ar_mechanism_is_reproducible() -> None:
    config = StationaryARConfig(minimum_order=3, maximum_order=3)
    assert sample_stationary_ar_mechanism(config, seed=42) == sample_stationary_ar_mechanism(
        config,
        seed=42,
    )
