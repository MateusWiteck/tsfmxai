"""Distributional coverage metrics for stochastic time-series mechanisms."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram
from scipy.stats import wasserstein_distance


DEFAULT_THRESHOLDS = {"acf_mae": 0.10, "psd_jsd": 0.10,
                      "abs_log_variance_ratio": 0.25, "standardized_wasserstein": 0.25}


def acf_vector(values: np.ndarray, max_lag: int = 64) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    denominator = np.mean(centered**2, axis=1) + 1e-12
    return np.array([np.mean(np.mean(centered[:, :-lag] * centered[:, lag:], axis=1) / denominator)
                     for lag in range(1, max_lag + 1)])


def psd_distribution(values: np.ndarray) -> np.ndarray:
    _, power = periodogram(values, axis=1, detrend=False)
    power = np.mean(power, axis=0) + 1e-12
    return power / power.sum()


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)
    return float(0.5 * np.sum(left * np.log(left / middle))
                 + 0.5 * np.sum(right * np.log(right / middle)))


def distribution_metrics(reference: np.ndarray, generated: np.ndarray, start: int, *,
                         max_acf_lag: int = 64) -> dict[str, float]:
    ref, gen = np.asarray(reference[:, start:], dtype=float), np.asarray(generated[:, start:], dtype=float)
    ref_scale = float(np.std(ref) + 1e-12)
    return {
        "acf_mae": float(np.mean(np.abs(acf_vector(ref, max_acf_lag) - acf_vector(gen, max_acf_lag)))),
        "psd_jsd": js_divergence(psd_distribution(ref), psd_distribution(gen)),
        "abs_log_variance_ratio": float(abs(np.log((np.var(gen) + 1e-12) / (np.var(ref) + 1e-12)))),
        "standardized_wasserstein": float(wasserstein_distance(ref.ravel(), gen.ravel()) / ref_scale),
    }


def passes_coverage(metrics: dict[str, float], thresholds: dict[str, float] | None = None) -> bool:
    limits = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    return all(np.isfinite(metrics[key]) and metrics[key] <= limit for key, limit in limits.items())


def violation_score(row: dict[str, float], *, thresholds: dict[str, float] | None = None,
                    cross_correlation_threshold: float | None = None) -> float:
    limits = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    score = sum(max(float(row[key]) / limit - 1.0, 0.0) for key, limit in limits.items())
    if cross_correlation_threshold is not None:
        score += max(float(row["cross_corr_frobenius"]) / cross_correlation_threshold - 1.0, 0.0)
    return float(score)


def channel_metrics(reference: np.ndarray, generated: np.ndarray, start: int, *,
                    max_acf_lag: int = 64) -> dict[str, float]:
    individual = [distribution_metrics(reference[:, :, channel], generated[:, :, channel], start,
                                       max_acf_lag=max_acf_lag)
                  for channel in range(reference.shape[2])]
    metrics = {key: float(np.mean([item[key] for item in individual])) for key in DEFAULT_THRESHOLDS}
    ref = reference[:, start:].reshape(-1, reference.shape[2])
    gen = generated[:, start:].reshape(-1, generated.shape[2])
    metrics["cross_corr_frobenius"] = float(
        np.linalg.norm(np.corrcoef(ref, rowvar=False) - np.corrcoef(gen, rowvar=False), ord="fro")
        / reference.shape[2])
    return metrics


def passes_channel_coverage(metrics: dict[str, float], *, thresholds: dict[str, float] | None = None,
                            cross_correlation_threshold: float = 0.10) -> bool:
    return passes_coverage(metrics, thresholds) and np.isfinite(metrics["cross_corr_frobenius"]) \
        and metrics["cross_corr_frobenius"] <= cross_correlation_threshold


def dominant_periods(series: np.ndarray, *, count: int = 2, minimum: int = 2,
                     maximum: int = 256) -> list[int]:
    """Select deterministic candidate periods from the mean periodogram."""
    values = np.asarray(series, dtype=float)
    if values.ndim == 3:
        values = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    _, power = periodogram(values, axis=1, detrend="linear")
    mean_power = np.mean(power, axis=0)
    frequencies = np.fft.rfftfreq(values.shape[1])
    candidates = []
    for index in range(1, len(frequencies)):
        period = int(round(1.0 / frequencies[index]))
        if minimum <= period <= maximum:
            candidates.append((float(mean_power[index]), period))
    selected: list[int] = []
    for _, period in sorted(candidates, reverse=True):
        if period not in selected:
            selected.append(period)
        if len(selected) == count:
            break
    return selected
