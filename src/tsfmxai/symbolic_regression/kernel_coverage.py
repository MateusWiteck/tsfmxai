"""Budgeted, validation-selected recurrence search for GP coverage experiments.

This is a finite dictionary search, not an exhaustive search of the AST grammar.
Coefficients are unrestricted: it tests vocabulary expressivity, not the support
of the default random coefficient sampler or a stationarity certificate.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from tsfmxai.representations.expression import (
    Add, Lag, MechanismBinding, Multiply, Parameter, Tanh, operator_count,
)


@dataclass(frozen=True)
class SearchBudget:
    max_lag: int = 16
    max_operators: int = 6
    nonlinear: bool = True

    def __post_init__(self):
        if self.max_lag < 1 or self.max_operators < 0:
            raise ValueError("Positive max_lag and nonnegative max_operators required.")


def lagged_rows(series, max_lag, start=None):
    """Pool trajectories without ever creating windows across their boundaries."""
    values = np.asarray(series, dtype=float)
    start = max_lag if start is None else start
    if (values.ndim != 2 or not np.isfinite(values).all()
            or not 1 <= max_lag <= start < values.shape[1]):
        raise ValueError("Expected finite (trajectory, time) data and valid window bounds.")
    x = np.stack([values[:, start-lag:values.shape[1]-lag]
                  for lag in range(1, max_lag+1)], axis=-1)
    return x.reshape(-1, max_lag), values[:, start:].reshape(-1)


def _dictionary(max_lag, nonlinear):
    terms = [("lag", (lag,)) for lag in range(1, max_lag+1)]
    if nonlinear:
        terms += [("product", pair) for pair in
                  combinations_with_replacement(range(1, max_lag+1), 2)]
        terms += [("tanh", (lag,)) for lag in range(1, max_lag+1)]
    return terms


def _features(x, terms):
    columns = []
    for kind, lags in terms:
        column = x[:, lags[0]-1]
        if kind == "product":
            column = column * x[:, lags[1]-1]
        elif kind == "tanh":
            column = np.tanh(column)
        columns.append(column)
    return np.column_stack(columns) if columns else np.empty((len(x), 0))


@dataclass
class FittedRecurrence:
    terms: list
    coefficients: np.ndarray
    budget: SearchBudget
    validation_mse: float

    def predict(self, x):
        return self.coefficients[0] + _features(np.asarray(x), self.terms) @ self.coefficients[1:]

    def as_ast(self):
        """Export the actual executable tree, including all coefficient gates."""
        expression = Parameter("P0")
        parameters = {"P0": float(self.coefficients[0])}
        lags = {}
        for index, ((kind, delays), coefficient) in enumerate(zip(self.terms, self.coefficients[1:]), 1):
            nodes = []
            for delay in delays:
                slot = f"L{len(lags)}"
                lags[slot] = int(delay)
                nodes.append(Lag(0, slot))
            term = nodes[0]
            if kind == "product":
                term = Multiply(*nodes)
            elif kind == "tanh":
                term = Tanh(term)
            slot = f"P{index}"
            parameters[slot] = float(coefficient)
            expression = Add(expression, Multiply(Parameter(slot), term))
        assert operator_count(expression) <= self.budget.max_operators
        return expression, MechanismBinding(parameters, lags)

    def rollout(self, histories, steps, *, seed=0, innovation_std=0.0, limit=1e6):
        """Free-running generation; a failed path remains NaN, never clipped."""
        histories = np.asarray(histories, dtype=float)
        if (histories.ndim != 2 or histories.shape[1] < self.budget.max_lag
                or steps < 1 or innovation_std < 0 or not np.isfinite(histories).all()):
            raise ValueError("Invalid histories, horizon, or innovation scale.")
        values = np.full((len(histories), histories.shape[1]+steps), np.nan)
        values[:, :histories.shape[1]] = histories
        alive = np.ones(len(histories), dtype=bool)
        rng = np.random.default_rng(seed)
        for t in range(histories.shape[1], values.shape[1]):
            x = values[:, t-np.arange(1, self.budget.max_lag+1)]
            with np.errstate(over="ignore", invalid="ignore"):
                prediction = self.predict(x) + rng.normal(0, innovation_std, len(values))
            alive &= np.isfinite(prediction) & (np.abs(prediction) <= limit)
            values[alive, t] = prediction[alive]
        return values[:, histories.shape[1]:]


def fit_recurrence(train, validation, budget=SearchBudget(), *, start=None):
    """Forward least-squares dictionary search; select a prefix on validation.

    All candidate additions are scored on training residual reduction. Every
    accepted model is refit by least squares and the best validation prefix is
    retained. The intercept is a terminal; linear/nonlinear terms cost 2/3 ops.
    """
    x, y = lagged_rows(train, budget.max_lag, start)
    vx, vy = lagged_rows(validation, budget.max_lag, start)
    terms = _dictionary(budget.max_lag, budget.nonlinear)
    features = _features(x, terms)
    selected = []
    design = np.ones((len(y), 1))
    coefficients = np.array([y.mean()])
    best = FittedRecurrence([], coefficients.copy(), budget, float(np.mean((vy-y.mean())**2)))
    spent = 0
    costs = np.array([2 if kind == "lag" else 3 for kind, _ in terms])
    while True:
        eligible = [i for i in range(len(terms)) if i not in selected and spent+costs[i] <= budget.max_operators]
        if not eligible:
            break
        q, _ = np.linalg.qr(design, mode="reduced")
        residual_features = features[:, eligible] - q @ (q.T @ features[:, eligible])
        norms = np.sum(residual_features**2, axis=0)
        residual = y - design @ coefficients
        gains = np.divide((residual_features.T @ residual)**2, norms,
                          out=np.full(len(eligible), -np.inf), where=norms > 1e-16)
        if not np.isfinite(gains).any():
            break
        chosen = eligible[int(np.argmax(gains))]
        selected.append(chosen)
        spent += costs[chosen]
        design = np.column_stack([np.ones(len(y)), features[:, selected]])
        coefficients = np.linalg.lstsq(design, y, rcond=1e-10)[0]
        model = FittedRecurrence([terms[i] for i in selected], coefficients.copy(), budget, 0.0)
        model.validation_mse = float(np.mean((vy-model.predict(vx))**2))
        if model.validation_mse < best.validation_mse:
            best = model
    best.as_ast()
    return best


def relative_rmse(target, prediction, training_scale):
    """RMSE / training standard deviation; failures score infinity."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or not np.isfinite(target).all():
        raise ValueError("Targets must be finite and shapes must match.")
    if not np.isfinite(training_scale) or training_scale <= 0:
        raise ValueError("Training scale must be finite and positive.")
    if not np.isfinite(prediction).all():
        return float("inf")
    return float(np.sqrt(np.mean((target-prediction)**2))/training_scale)


def gp_local_oracle(covariance, series, max_lag, *, start=None):
    """Exact zero-mean GP conditional moments with the same observed lag window.

    Coefficients may vary with time. This privileged reference is not a fitted
    homogeneous AR model. Pseudoinverses support degenerate constant kernels.
    """
    values = np.asarray(series, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    start = max_lag if start is None else start
    lagged_rows(values, max_lag, start)
    if covariance.shape != (values.shape[1], values.shape[1]) or not np.isfinite(covariance).all():
        raise ValueError("Covariance must match the series length and be finite.")
    predictions, variances = [], []
    for t in range(start, values.shape[1]):
        indices = t-np.arange(1, max_lag+1)
        cross = covariance[t, indices]
        weights = np.linalg.pinv(covariance[np.ix_(indices, indices)], rcond=1e-12) @ cross
        predictions.append(values[:, indices] @ weights)
        variances.append(max(0.0, float(covariance[t, t]-cross @ weights)))
    return np.asarray(predictions).T, np.asarray(variances)
