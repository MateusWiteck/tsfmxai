"""Analytic bounds, retained mechanisms, stochastic initialization and seeds."""

from dataclasses import replace

import numpy as np
import pytest
from joblib import Parallel, delayed

from tsfmxai.generators import (
    ShaoWuConfig, build_shao_wu_mechanism, derive_sample_seed,
    evaluate_bound_expression, evaluate_expression_with_gradient, global_expression_bounds,
    initialize_shao_wu_history, sample_shao_wu_mechanism, sample_shao_wu_trajectories,
)
from tsfmxai.representations import (
    Add, Lag, MechanismBinding, Multiply, Parameter, Tanh, Time,
    canonicalize_and_relabel, operator_count, to_bound_formula,
)


def test_tanh_ast_format_gradient_and_canonicalization():
    tree = Add(Tanh(Lag(0, "lag")), Parameter("a"))
    binding = MechanismBinding({"a": .2}, {"lag": 2})
    assert "tanh" in to_bound_formula(tree, binding)
    assert operator_count(canonicalize_and_relabel(tree)) == 2
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
