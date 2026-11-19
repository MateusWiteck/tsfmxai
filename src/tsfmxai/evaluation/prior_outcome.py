"""Unit-free outcomes per dataset for paired zero-shot comparisons."""
from __future__ import annotations

import math

import numpy as np

from tsfmxai.evaluation.real_world import observed_metrics


def seasonal_naive(context: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    """Repeat the preceding daily cycle, using earlier same-hour observations for gaps."""
    context = np.asarray(context, dtype=float)
    if context.ndim != 3 or period < 1 or horizon < 1 or context.shape[1] < period:
        raise ValueError("Expected [window, history, channel] and a valid period/horizon")
    result = np.empty((len(context), horizon, context.shape[2]), dtype=float)
    for window in range(len(context)):
        for channel in range(context.shape[2]):
            history = context[window, :, channel]
            valid = np.flatnonzero(np.isfinite(history))
            if len(valid) == 0:
                raise ValueError("Seasonal naive needs a past observation per channel")
            for step in range(horizon):
                position = len(history) - period + step % period
                while position >= 0 and not np.isfinite(history[position]):
                    position -= period
                result[window, step, channel] = (history[position] if position >= 0
                                                 else history[valid[-1]])
    return result


def _domain_log_score(target, channels, predictions):
    if "seasonal" not in predictions or "lmc" not in predictions or "own" not in predictions:
        raise ValueError("Need paired seasonal, lmc, and own predictions")
    metrics = {name: observed_metrics(pred, target, channels)["per_channel"]
               for name, pred in predictions.items()}
    scores = {}
    for arm in ("lmc", "own"):
        ratios = []
        for channel in channels:
            denominator = metrics["seasonal"][channel]["mae"]
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"Seasonal MAE is zero/undefined for {channel}")
            numerator = metrics[arm][channel]["mae"]
            ratios.append(max(numerator, np.finfo(float).tiny) / denominator)
        scores[arm] = float(np.mean(np.log(ratios)))
    return scores, metrics


def per_dataset_relative_mae(domains):
    """Geometric mean of channel MAE ratios, separately for each dataset."""
    if not domains:
        raise ValueError("At least one domain is required")
    details = {}
    for domain, item in domains.items():
        scores, metrics = _domain_log_score(item["target"], item["channels"],
                                             item["predictions"])
        relative = {arm: math.exp(value) for arm, value in scores.items()}
        details[domain] = {"per_channel": metrics,
                           "relative_mae_per_channel": {
                               arm: {channel: metrics[arm][channel]["mae"] /
                                     metrics["seasonal"][channel]["mae"]
                                     for channel in item["channels"]}
                               for arm in ("lmc", "own")},
                           "relative_geomean": relative,
                           "own_over_lmc": relative["own"] / relative["lmc"]}
    return {"per_domain": details}


def paired_block_interval(domains, *, block_windows=7, replicates=1000, seed=2021,
                          alpha=0.05):
    """Resample contiguous origin blocks, keeping models paired in each domain."""
    if block_windows < 1 or replicates < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid block bootstrap settings")
    rng = np.random.default_rng(seed)
    ratios = {domain: np.empty(replicates) for domain in domains}
    for replicate in range(replicates):
        sampled = {}
        for domain, item in domains.items():
            length = len(item["target"])
            if length < block_windows:
                raise ValueError(f"{domain} has fewer origins than one block")
            starts = rng.integers(0, length - block_windows + 1,
                                  size=math.ceil(length / block_windows))
            indices = np.concatenate([np.arange(start, start + block_windows)
                                      for start in starts])[:length]
            sampled[domain] = {"target": item["target"][indices],
                               "channels": item["channels"],
                               "predictions": {name: pred[indices] for name, pred
                                               in item["predictions"].items()}}
        scores = per_dataset_relative_mae(sampled)["per_domain"]
        for domain in domains:
            ratios[domain][replicate] = scores[domain]["own_over_lmc"]
    intervals = {}
    for domain, values in ratios.items():
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        intervals[domain] = {"low": float(low), "high": float(high),
                             "replicates": replicates, "block_windows": block_windows,
                             "seed": seed,
                             "decision": ("own" if high < 1 else "lmc" if low > 1
                                          else "inconclusive")}
    return intervals
