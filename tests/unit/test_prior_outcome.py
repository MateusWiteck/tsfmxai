import numpy as np
import pytest

from tsfmxai.evaluation.prior_outcome import (seasonal_naive,
                                               per_dataset_relative_mae,
                                               paired_block_interval)


def test_seasonal_repeats_past_day_and_fills_missing_from_previous_day():
    context = np.arange(48, dtype=float).reshape(1, 48, 1)
    context[0, 24, 0] = np.nan
    forecast = seasonal_naive(context, 26)
    assert forecast.shape == (1, 26, 1)
    assert forecast[0, 0, 0] == 0
    assert forecast[0, 1, 0] == 25
    assert forecast[0, 24, 0] == 0


def test_relative_mae_is_separate_for_each_dataset():
    target = np.ones((8, 2, 2))
    domains = {
        "air": {"target": target, "channels": ("a", "b"),
                "predictions": {"seasonal": np.zeros_like(target),
                                "lmc": np.full_like(target, 1.5),
                                "own": np.full_like(target, 1.25)}},
        "energy": {"target": target[:, :, :1], "channels": ("c",),
                   "predictions": {"seasonal": np.zeros_like(target[:, :, :1]),
                                   "lmc": np.full_like(target[:, :, :1], 1.5),
                                   "own": np.full_like(target[:, :, :1], 1.25)}},
    }
    result = per_dataset_relative_mae(domains)
    assert set(result) == {"per_domain"}
    assert result["per_domain"]["air"]["own_over_lmc"] == pytest.approx(0.5)
    assert result["per_domain"]["energy"]["own_over_lmc"] == pytest.approx(0.5)
    interval = paired_block_interval(domains, block_windows=2, replicates=25)
    assert interval["air"]["decision"] == "own"
    assert interval["energy"]["decision"] == "own"
    assert interval["air"]["low"] == pytest.approx(0.5)


def test_opposite_dataset_directions_are_kept_separate():
    target = np.ones((8, 2, 1))
    domains = {
        "air": {"target": target, "channels": ("x",),
                "predictions": {"seasonal": np.zeros_like(target),
                                "lmc": target + 0.5, "own": target + 0.25}},
        "energy": {"target": target, "channels": ("y",),
                   "predictions": {"seasonal": np.zeros_like(target),
                                   "lmc": target + 0.25, "own": target + 0.5}},
    }
    scores = per_dataset_relative_mae(domains)["per_domain"]
    assert scores["air"]["own_over_lmc"] == pytest.approx(0.5)
    assert scores["energy"]["own_over_lmc"] == pytest.approx(2.0)
    intervals = paired_block_interval(domains, block_windows=2, replicates=25)
    assert intervals["air"]["decision"] == "own"
    assert intervals["energy"]["decision"] == "lmc"


def test_zero_seasonal_denominator_rejected():
    target = np.ones((8, 1, 1))
    domain = {"target": target, "channels": ("x",),
              "predictions": {"seasonal": target, "lmc": target+1, "own": target+1}}
    with pytest.raises(ValueError, match="Seasonal MAE"):
        per_dataset_relative_mae({"domain": domain})
