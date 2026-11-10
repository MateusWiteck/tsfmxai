"""Analytic bounds, retained mechanisms, stochastic initialization and seeds."""

from dataclasses import replace

import numpy as np
import pytest
from joblib import Parallel, delayed

from tsfmxai.generators import (
    ShaoWuConfig, build_shao_wu_mechanism, derive_sample_seed,
    evaluate_bound_expression, evaluate_expression_with_gradient, global_expression_bounds,
    initialize_shao_wu_history, sample_shao_wu_mechanism, sample_shao_wu_trajectories,
    saturate_product_lags,
)
from tsfmxai.representations import (
    Add, Lag, MechanismBinding, Multiply, Parameter, Tanh,
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
        (transformed,), (binding,), ShaoWuConfig()
    )
    assert mechanism.certificate.holds
    assert mechanism.certificate.drift_norm_bound is None


def test_product_only_sampler_is_deterministic_and_preserves_some_bare_lags():
    config = ShaoWuConfig()
    mechanisms = [sample_shao_wu_mechanism(config, seed=s) for s in range(50)]
    assert all(m.certificate.holds for m in mechanisms)
    assert any(isinstance(node, Tanh) for m in mechanisms
               for expression in m.expressions for node in walk(expression))
    assert any(isinstance(node, Lag) for m in mechanisms
               for expression in m.expressions for node in walk(expression)
               if isinstance(node, Lag))
    assert mechanisms == [sample_shao_wu_mechanism(config, seed=s) for s in range(50)]


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
