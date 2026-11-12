import numpy as np
import pytest

from tsfmxai.generators.trajectory import evaluate_bound_expression
from tsfmxai.representations.expression import operator_count
from tsfmxai.symbolic_regression.kernel_coverage import (
    FittedRecurrence, SearchBudget, fit_recurrence, gp_local_oracle,
    lagged_rows, relative_rmse,
)


def test_windows_do_not_cross_trajectories():
    x, y = lagged_rows([[1, 2, 3, 4], [10, 20, 30, 40]], 2)
    np.testing.assert_array_equal(x, [[2, 1], [3, 2], [20, 10], [30, 20]])
    np.testing.assert_array_equal(y, [3, 4, 30, 40])


def test_nonlinear_recovery_budget_and_executable_ast():
    rng = np.random.default_rng(41)
    paths = np.zeros((20, 16))
    paths[:, 0] = rng.uniform(-1, 1, 20)
    for t in range(1, 16):
        paths[:, t] = 0.2 + 0.5 * paths[:, t-1]**2
    model = fit_recurrence(paths[:12], paths[12:], SearchBudget(1, 3))
    assert model.validation_mse < 1e-20
    expression, binding = model.as_ast()
    assert operator_count(expression) <= 3
    x, y = lagged_rows(paths[12:], 1)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-10)
    for path in paths[12:]:
        ast_value = evaluate_bound_expression(expression, binding, time=1, histories={0: path})
        assert ast_value == pytest.approx(model.predict([[path[0]]])[0])
    np.testing.assert_allclose(model.rollout(paths[12:, :1], 15), paths[12:, 1:], atol=1e-10)


@pytest.mark.parametrize("budget", [0, 1, 2, 6, 14])
def test_actual_tree_respects_budget(budget):
    data = np.random.default_rng(8).normal(size=(6, 40))
    model = fit_recurrence(data[:4], data[4:], SearchBudget(4, budget))
    expression, binding = model.as_ast()
    assert operator_count(expression) <= budget
    assert max(binding.lag_values.values(), default=0) <= 4


def test_gp_white_noise_and_degenerate_constant():
    paths = np.random.default_rng(3).normal(size=(3, 12))
    means, variances = gp_local_oracle(np.eye(12)*2, paths, 3)
    np.testing.assert_array_equal(means, np.zeros((3, 9)))
    np.testing.assert_array_equal(variances, np.ones(9)*2)
    paths = np.tile([[2], [-3]], (1, 12))
    means, variances = gp_local_oracle(np.ones((12, 12)), paths, 3)
    np.testing.assert_allclose(means, paths[:, 3:])
    np.testing.assert_allclose(variances, 0, atol=1e-12)


def test_gp_ar1_conditional_moments():
    covariance = 0.7**np.abs(np.arange(12)[:, None]-np.arange(12))
    paths = np.random.default_rng(2).normal(size=(4, 12))
    means, variances = gp_local_oracle(covariance, paths, 3)
    np.testing.assert_allclose(means, 0.7*paths[:, 2:-1], atol=1e-12)
    np.testing.assert_allclose(variances, 1-0.7**2, atol=1e-12)


def test_scale_invariance_and_rollout_failure_is_not_hidden():
    assert relative_rmse([1, 2], [2, 4], 2) == pytest.approx(relative_rmse([10, 20], [20, 40], 20))
    model = FittedRecurrence([("product", (1, 1))], np.array([0., 2.]), SearchBudget(1, 3), 0.)
    prediction = model.rollout([[2.]], 12, limit=100)
    assert np.isnan(prediction).any()
    assert np.isinf(relative_rmse(np.zeros_like(prediction), prediction, 1))
    with pytest.raises(ValueError):
        relative_rmse([1], [1], 0)


def test_seeded_innovations():
    model = FittedRecurrence([], np.array([0.]), SearchBudget(1, 0), 0.)
    a = model.rollout([[0.]], 20, seed=10, innovation_std=1)
    np.testing.assert_array_equal(a, model.rollout([[0.]], 20, seed=10, innovation_std=1))
    assert not np.array_equal(a, model.rollout([[0.]], 20, seed=11, innovation_std=1))


def test_complete_pipeline_retains_white_noise_and_manifest():
    from tsfmxai.pipelines.kernel_vocabulary_coverage import CoverageConfig, run_coverage_study
    config = CoverageConfig(length=40, composition_count=0, train_count=2,
                            validation_count=2, test_count=2, lags=(2,), operators=(0,), horizon=4)
    study = run_coverage_study(config)
    assert len(study["manifest"]) == 33
    assert len(study["rows"]) == 66
    assert sum(row["family"] == "white" for row in study["rows"]) == 4
    assert all(row["used_operators"] == 0 for row in study["rows"])
    assert all(len(row["per_path_one_step_nrmse"]) == 2 for row in study["rows"])
