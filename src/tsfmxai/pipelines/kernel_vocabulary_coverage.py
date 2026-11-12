"""Reproducible finite-suite comparison of AR and a bounded AST dictionary."""

from dataclasses import asdict, dataclass

import numpy as np

from tsfmxai.evaluation.process_coverage import distribution_metrics
from tsfmxai.generators.kernel_synth import retained_kernel_cases, stable_factor
from tsfmxai.representations.expression import operator_count, to_bound_formula, to_skeleton_formula
from tsfmxai.symbolic_regression.kernel_coverage import (
    SearchBudget, fit_recurrence, gp_local_oracle, lagged_rows, relative_rmse,
)


@dataclass(frozen=True)
class CoverageConfig:
    seed: int = 20261030
    length: int = 1024
    composition_count: int = 20
    train_count: int = 8
    validation_count: int = 4
    test_count: int = 12
    lags: tuple[int, ...] = (16, 24)
    operators: tuple[int, ...] = (6, 14)
    horizon: int = 64

    def __post_init__(self):
        if (not self.lags or not self.operators or min(self.lags) < 1
                or min(self.operators) < 0 or self.composition_count < 0
                or min(self.train_count, self.validation_count, self.test_count) < 2
                or not 1 <= self.horizon <= self.length-max(self.lags)
                or self.length-max(self.lags) <= 16):
            raise ValueError("Invalid study configuration.")


def run_coverage_study(config=CoverageConfig(), *, progress=None):
    """Fit on independent realizations; return all rows, bindings and examples.

    Distribution distances are calibrated against a same-sized independent GP
    sample, not interpreted as tests of equality. No cases are excluded as noise.
    Test data never select a model, lag, budget, threshold or innovation scale.
    """
    cases = retained_kernel_cases(global_seed=config.seed, length=config.length,
                                  composition_count=config.composition_count)
    start = max(config.lags)
    rows, manifests, examples = [], [], {}
    for index, case in enumerate(cases):
        # Separate streams for independent realizations and generator innovations.
        sample_seed = config.seed + 100_000 + index
        rng = np.random.default_rng(sample_seed)
        factor = stable_factor(case["covariance"])
        # The oracle uses the actual stabilized covariance used for sampling.
        covariance = factor @ factor.T
        counts = (config.train_count, config.validation_count, config.test_count, config.test_count)
        samples = rng.standard_normal((sum(counts), config.length)) @ factor.T
        train, validation, test, calibration = np.split(samples, np.cumsum(counts)[:-1])
        scale = float(np.std(train))
        # Scale only: zero GP prior mean is preserved, and fitting remains well conditioned.
        train, validation, test, calibration = [x / scale for x in (train, validation, test, calibration)]
        covariance = covariance / scale**2
        manifests.append({"id": case["id"], "formula": case["formula"], "family": case["family"],
                          "category": case["category"], "sample_seed": sample_seed,
                          "training_scale": scale})
        calibration_metrics = distribution_metrics(test, calibration, start, max_acf_lag=16)
        for lag in config.lags:
            oracle, conditional_variance = gp_local_oracle(covariance, test, lag, start=start)
            x, target = lagged_rows(test, lag, start)
            for operations in config.operators:
                for nonlinear in (False, True):
                    budget = SearchBudget(lag, operations, nonlinear)
                    fit = fit_recurrence(train, validation, budget, start=start)
                    expression, binding = fit.as_ast()
                    prediction = fit.predict(x).reshape(test[:, start:].shape)
                    tx, ty = lagged_rows(train, lag, start)
                    innovation_std = float(np.sqrt(np.mean((ty-fit.predict(tx))**2)))
                    forecast = fit.rollout(test[:, :-config.horizon], config.horizon)
                    generation_seed = config.seed + 200_000 + index
                    generated = fit.rollout(test[:, :start], config.length-start,
                                            seed=generation_seed, innovation_std=innovation_std)
                    full_generation = np.column_stack([test[:, :start], generated])
                    divergent = ~np.isfinite(generated).all(axis=1)
                    metrics = (distribution_metrics(test, full_generation, start, max_acf_lag=16)
                               if not divergent.any() else {key: float("inf") for key in calibration_metrics})
                    row = {"case": case["id"], "family": case["family"], "category": case["category"],
                           "model": "AST_dictionary" if nonlinear else "sparse_AR",
                           "max_lag": lag, "max_operators": operations,
                           "used_operators": operator_count(expression),
                           "used_max_lag": max(binding.lag_values.values(), default=0),
                           "validation_nrmse": float(np.sqrt(fit.validation_mse)),
                           "one_step_nrmse": relative_rmse(test[:, start:], prediction, 1),
                           "oracle_mean_nrmse": relative_rmse(oracle, prediction, 1),
                           "oracle_observation_nrmse": relative_rmse(test[:, start:], oracle, 1),
                           "irreducible_nrmse": float(np.sqrt(conditional_variance.mean())),
                           "rollout_nrmse": relative_rmse(test[:, -config.horizon:], forecast, 1),
                           "divergent_fraction": float(divergent.mean()),
                           "innovation_std": innovation_std, "generation_seed": generation_seed,
                           "formula": to_bound_formula(expression, binding),
                           "skeleton": to_skeleton_formula(expression),
                           "parameters": dict(binding.parameter_values), "lags": dict(binding.lag_values),
                           **metrics, **{f"gp_calibration_{k}": v for k, v in calibration_metrics.items()}}
                    # Retain each path's score to expose sampling variability.
                    row["per_path_one_step_nrmse"] = np.sqrt(np.mean((test[:, start:]-prediction)**2, axis=1)).tolist()
                    rows.append(row)
                    if lag == max(config.lags) and operations == max(config.operators):
                        examples[(case["id"], row["model"])] = {
                            "reference": test[0], "prediction": prediction[0],
                            "forecast": forecast[0], "generated": full_generation[0],
                            "oracle": oracle[0], "start": start,
                        }
        if progress:
            progress(index+1, len(cases), case["id"])
    return {"config": asdict(config), "manifest": manifests, "rows": rows, "examples": examples}
