import random

import numpy as np
import pytest

from tsfmxai.generators import (
    ContractionTreeConfig,
    RandomTreeConfig,
    augmented_jacobian_norm,
    bound_expression_on_box,
    build_contractive_mechanism,
    certify_contractive_expression,
    evaluate_expression_with_gradient,
    sample_contractive_mechanisms,
    sample_contractive_trajectories,
    validate_contraction_monte_carlo,
)
from tsfmxai.representations import Add, Lag, MechanismBinding, Multiply, Parameter, Time


def nonlinear_expression():
    # 0.2 x(t-1) + 0.1 x(t-1) x(t-2)
    expression = Add(
        Multiply(Parameter("a"), Lag(0, "lag_1a")),
        Multiply(
            Parameter("b"),
            Multiply(Lag(0, "lag_1b"), Lag(0, "lag_2")),
        ),
    )
    binding = MechanismBinding(
        parameter_values={"a": 0.2, "b": 0.1},
        lag_values={"lag_1a": 1, "lag_1b": 1, "lag_2": 2},
    )
    return expression, binding


def compact_config() -> ContractionTreeConfig:
    return ContractionTreeConfig(
        tree_config=RandomTreeConfig(
            minimum_operators=1,
            maximum_operators=4,
            num_series=1,
            minimum_lag=1,
            maximum_lag=5,
            parameter_probability=0.55,
            time_probability=0.0,
            lag_probability=0.45,
            add_probability=0.8,
            minimum_parameter_magnitude=0.05,
            maximum_parameter_magnitude=0.8,
            require_lag=True,
            require_time=False,
            coefficient_gate_lags=True,
        ),
        domain_radius=1.0,
        innovation_bound=0.05,
        lag_weight_growth=1.25,
        target_lipschitz=0.9,
    )


def test_interval_and_gradient_bounds_cover_known_nonlinear_expression() -> None:
    expression, binding = nonlinear_expression()
    bound = bound_expression_on_box(expression, binding, domain_radius=1.0)

    assert bound.lower == pytest.approx(-0.3)
    assert bound.upper == pytest.approx(0.3)
    np.testing.assert_allclose(bound.derivative_absolute_bounds, [0.3, 0.1])

    value, gradient = evaluate_expression_with_gradient(
        expression,
        binding,
        [0.5, -0.25],
    )
    assert value == pytest.approx(0.0875)
    np.testing.assert_allclose(gradient, [0.175, 0.05])


def test_augmented_weighted_norm_certifies_shift_and_top_row() -> None:
    expression, binding = nonlinear_expression()
    certificate = certify_contractive_expression(
        expression,
        binding,
        domain_radius=1.0,
        innovation_bound=0.05,
        lag_weight_growth=1.25,
    )

    assert certificate.top_row_gain_bound == pytest.approx(0.425)
    assert certificate.shift_gain == pytest.approx(0.8)
    assert certificate.lipschitz_upper_bound == pytest.approx(0.8)
    assert certificate.domain_margin == pytest.approx(0.65)
    assert certificate.holds


def test_builder_scales_an_unstable_base_expression_into_the_certificate() -> None:
    expression = Add(Lag(0, "l1"), Lag(0, "l2"))
    binding = MechanismBinding(parameter_values={}, lag_values={"l1": 1, "l2": 2})
    mechanism = build_contractive_mechanism(expression, binding, compact_config())

    assert mechanism.scale < 1.0
    assert mechanism.certificate.holds
    assert mechanism.certificate.lipschitz_upper_bound <= 0.9
    assert mechanism.certificate.noisy_output_bound <= 1.0
    assert augmented_jacobian_norm(mechanism, [0.2, -0.3]) <= 0.9


def test_report_worked_example_matches_the_constructor_and_finite_changes() -> None:
    expression, binding = nonlinear_expression()
    config = ContractionTreeConfig(
        tree_config=compact_config().tree_config,
        innovation_bound=np.sqrt(3) * 0.05,
    )
    mechanism = build_contractive_mechanism(expression, binding, config)
    certificate = mechanism.certificate
    assert mechanism.scale == pytest.approx(0.95)
    np.testing.assert_allclose(certificate.derivative_absolute_bounds, [0.285, 0.095])
    assert certificate.top_row_gain_bound == pytest.approx(0.40375)
    assert certificate.lipschitz_upper_bound == pytest.approx(0.8)
    assert certificate.noisy_output_bound == pytest.approx(0.285 + np.sqrt(3) * 0.05)
    assert certificate.holds

    weights = np.asarray(certificate.state_weights)
    rng = np.random.default_rng(71)
    for first, second in rng.uniform(-1, 1, size=(100, 2, 2)):
        noise = rng.uniform(-config.innovation_bound, config.innovation_bound)
        outputs = []
        for state in (first, second):
            value, _ = evaluate_expression_with_gradient(
                mechanism.expression, mechanism.binding, state)
            outputs.append(np.array([value + noise, state[0]]))
        initial_distance = np.max(np.abs(first - second) / weights)
        updated_distance = np.max(np.abs(outputs[0] - outputs[1]) / weights)
        assert updated_distance <= 0.8 * initial_distance + 1e-12
        assert max(np.abs(output).max() for output in outputs) <= 1 + 1e-12


def test_sampled_mechanisms_are_reproducible_and_independent_of_global_rng() -> None:
    config = compact_config()
    random.seed(1)
    np.random.seed(2)
    first = sample_contractive_mechanisms(config, root_seed=77, count=20)
    random.seed(999)
    np.random.seed(888)
    second = sample_contractive_mechanisms(config, root_seed=77, count=20)

    assert first == second
    assert all(mechanism.certificate.holds for mechanism in first)
    assert all(mechanism.sample_index == index for index, mechanism in enumerate(first))


def test_certified_trajectory_generation_respects_domain_and_is_reproducible() -> None:
    mechanism = sample_contractive_mechanisms(compact_config(), root_seed=8, count=1)[0]
    arguments = dict(trajectory_count=32, generated_steps=128, root_seed=91)
    first = sample_contractive_trajectories(mechanism, **arguments)
    second = sample_contractive_trajectories(mechanism, **arguments)

    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first)) <= mechanism.certificate.domain_radius + 1e-12
    assert first.shape == (
        arguments["trajectory_count"],
        mechanism.certificate.state_dimension + arguments["generated_steps"],
    )


def test_monte_carlo_validation_checks_domain_and_local_jacobian() -> None:
    mechanism = sample_contractive_mechanisms(compact_config(), root_seed=19, count=1)[0]
    report = validate_contraction_monte_carlo(
        mechanism,
        trajectory_count=40,
        generated_steps=100,
        root_seed=23,
    )

    assert report.passed
    assert report.evaluated_state_transitions == 4_000
    assert report.maximum_observed_absolute_value <= 1.0 + 1e-12
    assert (
        report.maximum_observed_local_lipschitz
        <= report.certified_lipschitz_upper_bound + 1e-10
    )


def test_domain_singularities_and_noise_assumptions_are_checked() -> None:
    with pytest.raises(ValueError, match="innovation_bound"):
        ContractionTreeConfig(domain_radius=1.0, innovation_bound=1.0)
    with pytest.raises(ValueError, match="shift gain"):
        ContractionTreeConfig(lag_weight_growth=1.1, target_lipschitz=0.9)
    with pytest.raises(ValueError, match="time terminal"):
        ContractionTreeConfig(
            tree_config=RandomTreeConfig(time_probability=0.1, require_time=False)
        )

    expression = Add(Time(), Lag(0, "lag"))
    binding = MechanismBinding(parameter_values={}, lag_values={"lag": 1})
    with pytest.raises(ValueError, match="Time terminals"):
        bound_expression_on_box(expression, binding, domain_radius=1.0)
