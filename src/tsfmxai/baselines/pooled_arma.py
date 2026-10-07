"""Pooled conditional-sum-of-squares ARIMA and seasonal ARIMA baselines.

Seasonal models use the expanded lag support of the multiplicative seasonal
polynomials. Coefficients on that support are estimated without nonlinear
product constraints, so experiment reports identify the estimator as pooled
CSS with expanded multiplicative lag support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PooledARMAFit:
    intercept: np.ndarray
    ar_lags: tuple[int, ...]
    ma_lags: tuple[int, ...]
    ar_coefficients: np.ndarray
    ma_coefficients: np.ndarray
    innovation_covariance: np.ndarray
    difference_polynomial: np.ndarray
    channels: int
    scalar_input: bool
    css_iterations: int

    @property
    def difference_lag(self) -> int:
        return len(self.difference_polynomial) - 1

    @property
    def model_lag(self) -> int:
        return max((0, *self.ar_lags, *self.ma_lags))

    @property
    def start(self) -> int:
        return self.difference_lag + self.model_lag


def difference_polynomial(d: int = 0, D: int = 0,
                          seasonal_period: int | None = None) -> np.ndarray:
    if d < 0 or D < 0:
        raise ValueError("Differencing orders must be non-negative.")
    if D and (seasonal_period is None or seasonal_period < 2):
        raise ValueError("Seasonal differencing requires a period >= 2.")
    polynomial = np.array([1.0])
    for _ in range(d):
        polynomial = np.convolve(polynomial, np.array([1.0, -1.0]))
    for _ in range(D):
        seasonal = np.zeros(int(seasonal_period) + 1)
        seasonal[0], seasonal[-1] = 1.0, -1.0
        polynomial = np.convolve(polynomial, seasonal)
    return polynomial


def multiplicative_lag_support(nonseasonal_order: int, seasonal_order: int = 0,
                               seasonal_period: int | None = None) -> tuple[int, ...]:
    if nonseasonal_order < 0 or seasonal_order < 0:
        raise ValueError("AR and MA orders must be non-negative.")
    base = set(range(1, nonseasonal_order + 1))
    if seasonal_order == 0:
        return tuple(sorted(base))
    if seasonal_period is None or seasonal_period < 2:
        raise ValueError("Seasonal terms require a period >= 2.")
    seasonal = {seasonal_period * index for index in range(1, seasonal_order + 1)}
    return tuple(sorted(base | seasonal | {left + right for left in base for right in seasonal}))


def _as_3d(series: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(series, dtype=float)
    if values.ndim == 2:
        return values[:, :, None], True
    if values.ndim == 3:
        return values, False
    raise ValueError("Expected [trajectory, time] or [trajectory, time, channel].")


def _apply_difference(values: np.ndarray, polynomial: np.ndarray) -> np.ndarray:
    lag = len(polynomial) - 1
    output_length = values.shape[1] - lag
    if output_length <= 0:
        raise ValueError("Differencing consumes the complete trajectory.")
    output = np.zeros((values.shape[0], output_length, values.shape[2]))
    for index, coefficient in enumerate(polynomial):
        output += coefficient * values[:, lag - index:values.shape[1] - index]
    return output


def _design(transformed: np.ndarray, residuals: np.ndarray, ar_lags: tuple[int, ...],
            ma_lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    maximum_lag = max((0, *ar_lags, *ma_lags))
    rows = transformed.shape[1] - maximum_lag
    if rows <= 0:
        raise ValueError("Model lags consume the complete transformed trajectory.")
    features = [transformed[:, maximum_lag - lag:-lag or None] for lag in ar_lags]
    features += [residuals[:, maximum_lag - lag:-lag or None] for lag in ma_lags]
    if features:
        x = np.concatenate(features, axis=2).reshape(-1, len(features) * transformed.shape[2])
    else:
        x = np.empty((transformed.shape[0] * rows, 0))
    y = transformed[:, maximum_lag:].reshape(-1, transformed.shape[2])
    return x, y


def fit_pooled_arma(series: np.ndarray, *, order: tuple[int, int, int] = (1, 0, 0),
                    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
                    ridge: float = 1e-5, css_iterations: int = 4) -> PooledARMAFit:
    values, scalar_input = _as_3d(series)
    p, d, q = order
    P, D, Q, period = seasonal_order
    seasonal_period = int(period) if period else None
    polynomial = difference_polynomial(d, D, seasonal_period)
    transformed = _apply_difference(values, polynomial)
    ar_lags = multiplicative_lag_support(p, P, seasonal_period)
    ma_lags = multiplicative_lag_support(q, Q, seasonal_period)
    maximum_lag = max((0, *ar_lags, *ma_lags))
    if maximum_lag >= transformed.shape[1] - 1:
        raise ValueError("Model lags leave fewer than two transformed observations.")
    residuals = np.zeros_like(transformed)
    coefficients = None
    actual_iterations = 1 if not ma_lags else max(1, css_iterations)
    for _ in range(actual_iterations):
        x, y = _design(transformed, residuals, ar_lags, ma_lags)
        design = np.column_stack([np.ones(len(x)), x])
        penalty = ridge * np.eye(design.shape[1]); penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        prediction = design @ coefficients
        updated = np.zeros_like(transformed)
        updated[:, maximum_lag:] = (y - prediction).reshape(
            transformed.shape[0], transformed.shape[1] - maximum_lag, transformed.shape[2])
        residuals = updated
    if coefficients is None:
        raise RuntimeError("CSS fitting did not produce coefficients.")
    channels = values.shape[2]
    ar_size = len(ar_lags) * channels
    ar_coefficients = coefficients[1:1 + ar_size].reshape(len(ar_lags), channels, channels)
    ma_coefficients = coefficients[1 + ar_size:].reshape(len(ma_lags), channels, channels)
    valid_residuals = residuals[:, maximum_lag:].reshape(-1, channels)
    innovation_covariance = valid_residuals.T @ valid_residuals / max(len(valid_residuals), 1) \
        + 1e-8 * np.eye(channels)
    return PooledARMAFit(np.asarray(coefficients[0]), ar_lags, ma_lags, ar_coefficients,
                         ma_coefficients, innovation_covariance, polynomial, channels,
                         scalar_input, actual_iterations)


def _linear_prediction(fit: PooledARMAFit, transformed: np.ndarray,
                       residuals: np.ndarray, time_index: int) -> np.ndarray:
    prediction = np.broadcast_to(fit.intercept, (transformed.shape[0], fit.channels)).copy()
    for coefficient, lag in zip(fit.ar_coefficients, fit.ar_lags):
        prediction += transformed[:, time_index - lag] @ coefficient
    for coefficient, lag in zip(fit.ma_coefficients, fit.ma_lags):
        prediction += residuals[:, time_index - lag] @ coefficient
    return prediction


def simulate_pooled_arma(fit: PooledARMAFit, initials: np.ndarray, *, length: int,
                         rng: np.random.Generator, clip: float = 1e8) -> np.ndarray:
    initial_values, scalar_input = _as_3d(initials)
    if initial_values.shape[2] != fit.channels or initial_values.shape[1] < fit.start:
        raise ValueError(f"Initial values must contain {fit.channels} channels and {fit.start} steps.")
    output = np.zeros((initial_values.shape[0], length, fit.channels))
    output[:, :fit.start] = initial_values[:, :fit.start]
    transformed = np.zeros((initial_values.shape[0], length - fit.difference_lag, fit.channels))
    known = _apply_difference(output[:, :fit.start], fit.difference_polynomial)
    transformed[:, :known.shape[1]] = known
    innovations = rng.multivariate_normal(np.zeros(fit.channels), fit.innovation_covariance,
                                          size=(initial_values.shape[0], transformed.shape[1]))
    residuals = np.zeros_like(transformed)
    for transformed_index in range(fit.model_lag, transformed.shape[1]):
        prediction = _linear_prediction(fit, transformed, residuals, transformed_index)
        residuals[:, transformed_index] = innovations[:, transformed_index]
        transformed[:, transformed_index] = np.clip(prediction + residuals[:, transformed_index], -clip, clip)
        original_index = fit.difference_lag + transformed_index
        reconstructed = transformed[:, transformed_index].copy()
        for lag, coefficient in enumerate(fit.difference_polynomial[1:], start=1):
            reconstructed -= coefficient * output[:, original_index - lag]
        output[:, original_index] = np.clip(reconstructed, -clip, clip)
    return output[:, :, 0] if scalar_input and fit.scalar_input else output


def one_step_nmse(fit: PooledARMAFit, series: np.ndarray) -> float:
    values, _ = _as_3d(series)
    transformed = _apply_difference(values, fit.difference_polynomial)
    residuals = np.zeros_like(transformed)
    predictions, targets = [], []
    for time_index in range(fit.model_lag, transformed.shape[1]):
        transformed_prediction = _linear_prediction(fit, transformed, residuals, time_index)
        residuals[:, time_index] = transformed[:, time_index] - transformed_prediction
        original_index = fit.difference_lag + time_index
        prediction = transformed_prediction.copy()
        for lag, coefficient in enumerate(fit.difference_polynomial[1:], start=1):
            prediction -= coefficient * values[:, original_index - lag]
        predictions.append(prediction); targets.append(values[:, original_index])
    predicted, target = np.stack(predictions, axis=1), np.stack(targets, axis=1)
    return float(np.mean((target - predicted) ** 2) / (np.var(target) + 1e-12))
