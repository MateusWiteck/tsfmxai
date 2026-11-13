"""Forecast accuracy and probabilistic calibration metrics."""

from __future__ import annotations

import numpy as np


def _aligned(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have matching [..., horizon] shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("forecast arrays must be finite")
    return prediction, target


def point_forecast_metrics(prediction, target) -> dict[str, float]:
    """Return MAE, RMSE and MASE using the context first-difference scale."""
    prediction, target = _aligned(prediction, target)
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def mase(prediction, target, context, *, seasonal_period: int = 1) -> float:
    """Mean absolute forecast error scaled by a context-only naive forecast."""
    prediction, target = _aligned(prediction, target)
    context = np.asarray(context, dtype=float)
    if context.ndim != prediction.ndim or context.shape[:-1] != prediction.shape[:-1]:
        raise ValueError("context leading dimensions must match forecast arrays")
    if seasonal_period < 1 or context.shape[-1] <= seasonal_period:
        raise ValueError("context must exceed the positive seasonal_period")
    scale = np.mean(np.abs(np.diff(context, n=seasonal_period, axis=-1)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return float("nan")
    return float(np.mean(np.abs(prediction - target)) / scale)


def quantile_metrics(quantile_predictions, target, levels) -> dict[str, float]:
    """Pinball loss and empirical coverage per requested quantile level.

    ``quantile_predictions`` has shape [quantile, ..., horizon].
    """
    target = np.asarray(target, dtype=float)
    forecasts = np.asarray(quantile_predictions, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if forecasts.ndim != target.ndim + 1 or forecasts.shape[0] != len(levels):
        raise ValueError("quantile predictions must have [quantile, ...target shape]")
    if forecasts.shape[1:] != target.shape or not np.isfinite(forecasts).all():
        raise ValueError("quantile forecast shape mismatch or non-finite values")
    if np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    error = target[None, ...] - forecasts
    pinball = np.maximum(levels.reshape((-1,) + (1,) * target.ndim) * error,
                         (levels.reshape((-1,) + (1,) * target.ndim) - 1) * error)
    result = {"mean_pinball": float(pinball.mean())}
    for i, level in enumerate(levels):
        result[f"coverage_q{level:g}"] = float(np.mean(target <= forecasts[i]))
        result[f"pinball_q{level:g}"] = float(pinball[i].mean())
    return result
