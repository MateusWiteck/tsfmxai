"""Analytic bounds, retained mechanisms, stochastic initialization and seeds."""

from dataclasses import replace

import numpy as np
import pytest
from joblib import Parallel, delayed

from tsfmxai.generators import (
    ShaoWuConfig, build_shao_wu_mechanism, derive_sample_seed,
    evaluate_bound_expression, evaluate_expression_with_gradient, global_expression_bounds,
    initialize_shao_wu_history, sample_shao_wu_mechanism, sample_shao_wu_trajectories,
    saturate_product_lags, saturate_product_outputs, amplitude_growth_bounds,
    AmplitudeMomentCertificate,
)
from tsfmxai.representations import (
    Add, Lag, MechanismBinding, Multiply, Parameter, Tanh, Time,
    operator_count, to_bound_formula, walk,
)


def test_tanh_ast_format_and_gradient():
    tree = Add(Tanh(Lag(0, "lag")), Parameter("a"))
    binding = MechanismBinding({"a": .2}, {"lag": 2})
    assert "tanh" in to_bound_formula(tree, binding)
    assert operator_count(tree) == 2
    value, grad = evaluate_expression_with_gradient(tree, binding, [.1, .4])
    assert value == pytest.approx(np.tanh(.4) + .2)
    np.testing.assert_allclose(grad, [0, 1 - np.tanh(.4)**2])
    assert evaluate_bound_expression(tree, binding, time=2, histories={0: [.4, .1]}) == value


def test_product_bounds_and_moment_contraction_certificate():
    tree = Multiply(Parameter("a"), Multiply(Tanh(Lag(0, "l1")), Tanh(Lag(0, "l2"))))
    binding = MechanismBinding({"a": .3}, {"l1": 1, "l2": 2})
    amplitude, derivative = global_expression_bounds(tree, binding, num_series=1, order=2)
    assert amplitude == pytest.approx(.3)
    np.testing.assert_allclose(derivative, [[.3, .3]])
    mechanism = build_shao_wu_mechanism((tree,), (binding,), ShaoWuConfig())
    assert mechanism.certificate.holds
    assert mechanism.certificate.total_gain == pytest.approx(.57)
    assert mechanism.certificate.lag_gains == pytest.approx((.285, .285))
    assert mechanism.certificate.drift_norm_bound == pytest.approx(.285)
    for x in np.random.default_rng(42).normal(size=(50, 2)):
        _, grad = evaluate_expression_with_gradient(mechanism.expressions[0],
                                                   mechanism.bindings[0], x)
        assert np.sum(np.abs(grad)) <= mechanism.certificate.total_gain + 1e-14


def test_reject_unbounded_polynomial_and_time():
    lag = Lag(0, "lag")
    binding = MechanismBinding({}, {"lag": 1})
    with pytest.raises(ValueError, match="finite global"):
        build_shao_wu_mechanism((Multiply(lag, lag),), (binding,), ShaoWuConfig())
    with pytest.raises(ValueError, match="time"):
        global_expression_bounds(Time(), MechanismBinding({}, {}), num_series=1, order=1)


def test_unsaturated_affine_variant_preserves_global_moment_certificate():
    config = ShaoWuConfig(lag_transform="identity")
    mechanism = sample_shao_wu_mechanism(config, seed=17)
    assert mechanism.certificate.holds
    assert mechanism.certificate.drift_norm_bound is None
    assert np.isfinite(mechanism.certificate.anchor_l2)
    assert mechanism.proposal_attempt >= 0
    assert not any(isinstance(node, Tanh) for expression in mechanism.expressions
                   for node in walk(expression))
    assert mechanism == sample_shao_wu_mechanism(config, seed=17)

    tree = Add(Multiply(Parameter("a"), Lag(0, "lag")), Parameter("b"))
    binding = MechanismBinding({"a": .3, "b": .2}, {"lag": 1})
    linear = build_shao_wu_mechanism((tree,), (binding,), config)
    assert linear.certificate.holds
    assert linear.certificate.drift_norm_bound is None
    assert linear.certificate.anchor_l2 == pytest.approx(
        np.hypot(.95 * .2, config.innovation_std)
    )


@pytest.mark.parametrize("lags", [(1, 1), (1, 2), (1, 1, 2), (1, 2, 3, 4),
                                   (1, 1, 1, 1)])
def test_product_only_transform_saturates_nonlinear_lags_but_not_linear_lag(lags):
    product = Lag(0, "p0")
    for index in range(1, len(lags)):
        product = Multiply(product, Lag(0, f"p{index}"))
    tree = Add(Multiply(Parameter("b"), Lag(0, "linear")),
               Multiply(Parameter("a"), product))
    binding = MechanismBinding({"a": .4, "b": .2},
                               {"linear": 5, **{f"p{i}": lag
                                               for i, lag in enumerate(lags)}})
    transformed = saturate_product_lags(tree)
    assert isinstance(transformed.left.right, Lag)
    assert sum(isinstance(node, Tanh) for node in walk(transformed)) == len(lags)
    with pytest.raises(ValueError, match="finite global"):
        build_shao_wu_mechanism((tree,), (binding,), ShaoWuConfig())
    mechanism = build_shao_wu_mechanism(
        (transformed,), (binding,), ShaoWuConfig(lag_transform="product_lags")
    )
    assert mechanism.certificate.holds
    assert mechanism.certificate.drift_norm_bound is None


def test_product_only_sampler_is_deterministic_and_preserves_some_bare_lags():
    config = ShaoWuConfig(lag_transform="product_lags")
    mechanisms = [sample_shao_wu_mechanism(config, seed=s) for s in range(50)]
    assert all(m.certificate.holds and m.proposal_attempt == 0 for m in mechanisms)
    assert any(isinstance(node, Tanh) for m in mechanisms
               for expression in m.expressions for node in walk(expression))
    assert any(isinstance(node, Lag) for m in mechanisms
               for expression in m.expressions for node in walk(expression)
               if isinstance(node, Lag))
    assert mechanisms == [sample_shao_wu_mechanism(config, seed=s) for s in range(50)]


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_product_output_saturation_is_after_the_complete_product(degree):
    product = Lag(0, "l0")
    for i in range(1, degree):
        product = Multiply(product, Lag(0, f"l{i}"))
    linear = Multiply(Parameter("a"), Lag(0, "linear"))
    raw = Add(linear, Multiply(Parameter("b"), product))
    transformed = saturate_product_outputs(raw)
    assert transformed == Add(linear, Multiply(Parameter("b"), Tanh(product)))
    assert sum(isinstance(n, Tanh) for n in walk(transformed)) == 1
    binding = MechanismBinding({"a": .2, "b": .4},
                               {"linear": 1, **{f"l{i}": i+1 for i in range(degree)}})
    history = np.linspace(.5, 2., degree)
    actual = evaluate_bound_expression(transformed, binding, time=degree,
                                       histories={0: history})
    expected = .2 * history[-1] + .4 * np.tanh(np.prod(history))
    assert actual == pytest.approx(expected)


def test_product_output_saturation_does_not_claim_global_contraction():
    tree = Tanh(Multiply(Lag(0, "x"), Lag(0, "y")))
    binding = MechanismBinding({}, {"x": 1, "y": 2})
    for magnitude in (10., 1000.):
        _, gradient = evaluate_expression_with_gradient(tree, binding, [0., magnitude])
        assert gradient[0] == pytest.approx(magnitude)
    with pytest.raises(ValueError, match="finite global"):
        build_shao_wu_mechanism((tree,), (binding,), ShaoWuConfig())


def test_output_saturation_keeps_coefficients_outside_regardless_of_association():
    a, x, y = Parameter("a"), Lag(0, "x"), Lag(0, "y")
    assert saturate_product_outputs(Multiply(a, Multiply(x, y))) == (
        saturate_product_outputs(Multiply(Multiply(a, x), y)))


def test_output_mode_has_an_amplitude_bound_with_unbounded_derivatives():
    tree = Add(Multiply(Parameter("a"), Lag(0, "x")),
               Multiply(Parameter("b"), Tanh(Multiply(Lag(0, "x"), Lag(0, "y")))))
    binding = MechanismBinding({"a": .2, "b": .4}, {"x": 1, "y": 2})
    bound, growth = amplitude_growth_bounds(tree, binding, num_series=1, order=2)
    assert bound == pytest.approx(.4)
    np.testing.assert_allclose(growth, [[.2, 0.]])
    mechanism = build_shao_wu_mechanism((tree,), (binding,),
        ShaoWuConfig(lag_transform="product_output", innovation_std=2.))
    cert = mechanism.certificate
    assert isinstance(cert, AmplitudeMomentCertificate)
    assert cert.holds and cert.guarantee == "uniform_second_moment_bound"
    assert cert.total_gain == pytest.approx(.19)
    assert cert.uniform_l2_bound == pytest.approx((.38 + 2.) / .81)
    for state in np.random.default_rng(99).normal(size=(100, 2)) * 100:
        drift = evaluate_bound_expression(mechanism.expressions[0], mechanism.bindings[0],
                                           time=2, histories={0: state[::-1]})
        assert abs(drift) <= .38 + .19 * abs(state[0]) + 1e-12


def test_random_multivariate_output_mode_is_certified_and_reproducible():
    cfg = ShaoWuConfig(lag_transform="product_output",
        tree_config=replace(ShaoWuConfig().tree_config, num_series=2, maximum_lag=3))
    mechanism = sample_shao_wu_mechanism(cfg, seed=111)
    assert mechanism.certificate.holds
    assert mechanism.proposal_attempt == 0
    assert mechanism == sample_shao_wu_mechanism(cfg, seed=111)
    paths = sample_shao_wu_trajectories(mechanism, trajectory_count=50,
        generated_steps=60, root_seed=19)
    assert np.isfinite(paths).all()
    assert paths.shape == (50, 2, mechanism.order + 60)
    with pytest.raises(ValueError, match="time-homogeneous"):
        amplitude_growth_bounds(Tanh(Time()), MechanismBinding({}, {}), num_series=1, order=1)


def test_zero_gain_and_constant_output_law():
    mechanism = build_shao_wu_mechanism((Parameter("a"),),
                                      (MechanismBinding({"a": .2}, {}),), ShaoWuConfig())
    assert mechanism.certificate.total_gain == 0
    assert mechanism.certificate.drift_norm_bound == pytest.approx(.19)
    series = sample_shao_wu_trajectories(mechanism, trajectory_count=2000,
                                       generated_steps=2, root_seed=39)
    np.testing.assert_allclose(series[:, :, mechanism.order:].mean(axis=(0, 2)), [.19], atol=.004)


def test_multivariate_gain_seeds_and_worker_independence():
    cfg = ShaoWuConfig()
    cfg = replace(cfg, tree_config=replace(cfg.tree_config, num_series=2, maximum_lag=3))
    seeds = [derive_sample_seed(123, i) for i in range(4)]
    serial = [sample_shao_wu_mechanism(cfg, seed=s) for s in seeds]
    parallel = Parallel(n_jobs=2)(delayed(sample_shao_wu_mechanism)(cfg, seed=s) for s in seeds)
    assert serial == parallel
    mechanism = serial[0]
    assert mechanism.num_series == 2 and mechanism.certificate.holds
    a = sample_shao_wu_trajectories(mechanism, trajectory_count=2,
                                  generated_steps=30, root_seed=77)
    np.random.seed(981)
    np.random.normal(size=1000)
    b = sample_shao_wu_trajectories(mechanism, trajectory_count=3,
                                  generated_steps=30, root_seed=77)
    np.testing.assert_array_equal(a, b[:2])
    assert a.shape == (2, 2, mechanism.order + 30)
    assert not np.array_equal(a[0], a[1])


def test_gaussian_noise_and_direct_uniform_initialization():
    cfg = ShaoWuConfig(innovation_std=2.)
    mechanism = sample_shao_wu_mechanism(cfg, seed=11)
    series = sample_shao_wu_trajectories(mechanism, trajectory_count=3,
                                       generated_steps=200, root_seed=19)
    assert np.max(np.abs(series)) > cfg.initial_radius
    initial = initialize_shao_wu_history(mechanism, seed=91)
    assert initial.shape == (mechanism.num_series, mechanism.order)
    assert np.all(np.abs(initial) <= cfg.initial_radius)
    np.testing.assert_array_equal(initial, initialize_shao_wu_history(mechanism, seed=91))


def test_first_generated_value_follows_direct_history_without_presimulation():
    mechanism = sample_shao_wu_mechanism(ShaoWuConfig(), seed=11)
    root_seed = 29
    initial_stream, noise_stream = np.random.SeedSequence([root_seed, 0]).spawn(2)
    initial_seed = int(initial_stream.generate_state(1, dtype=np.uint64)[0])
    expected_history = initialize_shao_wu_history(mechanism, seed=initial_seed)
    expected_noise = np.random.default_rng(noise_stream).normal(
        0., mechanism.innovation_std, size=(mechanism.num_series, 1)
    )
    path = sample_shao_wu_trajectories(
        mechanism, trajectory_count=1, generated_steps=1, root_seed=root_seed
    )[0]
    np.testing.assert_array_equal(path[:, :mechanism.order], expected_history)
    expected_first = evaluate_bound_expression(
        mechanism.expressions[0], mechanism.bindings[0],
        time=mechanism.order, histories={0: expected_history[0]},
    ) + expected_noise[0, 0]
    assert path[0, mechanism.order] == pytest.approx(expected_first)
