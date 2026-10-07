import numpy as np
import pytest

from tsfmxai.evaluation import (
    StabilityDiagnosticConfig,
    diagnose_stability_prefixes,
    first_explosion_step,
    wilson_interval,
)


def test_first_explosion_step_detects_amplitude_and_nonfinite_values() -> None:
    assert first_explosion_step([0.0, -2.0, 10.1], explosion_limit=10.0) == 3
    assert first_explosion_step([0.0, np.nan], explosion_limit=10.0) == 2
    assert first_explosion_step([0.0, np.inf], explosion_limit=10.0) == 2
    assert first_explosion_step([0.0, 10.0], explosion_limit=10.0) is None


def test_diagnostics_are_horizon_specific_and_exclusive() -> None:
    config = StabilityDiagnosticConfig(
        horizons=(4, 8),
        explosion_limit=10.0,
        near_zero_radius=0.15,
        near_zero_tail_steps=2,
    )
    diagnostics = diagnose_stability_prefixes(
        [1.0, 0.5, 0.1, 0.1, 0.2, 0.3, 0.4, 12.0],
        config,
    )

    assert diagnostics[0].near_zero
    assert not diagnostics[0].exploded
    assert not diagnostics[0].neither
    assert diagnostics[1].exploded
    assert not diagnostics[1].near_zero
    assert not diagnostics[1].neither
    assert diagnostics[1].explosion_step == 8


def test_nonexploded_trajectory_must_cover_largest_horizon() -> None:
    config = StabilityDiagnosticConfig(
        horizons=(4, 8),
        explosion_limit=10.0,
        near_zero_radius=0.15,
        near_zero_tail_steps=2,
    )

    with pytest.raises(ValueError, match="largest horizon"):
        diagnose_stability_prefixes([0.0] * 7, config)


def test_wilson_interval_contains_observed_proportion() -> None:
    low, high = wilson_interval(40, 100)

    assert low < 0.4 < high
    assert wilson_interval(0, 100)[0] == pytest.approx(0.0)
    assert wilson_interval(100, 100)[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "arguments",
    [
        {"horizons": (), "explosion_limit": 10.0, "near_zero_radius": 0.1,
         "near_zero_tail_steps": 1},
        {"horizons": (2, 1), "explosion_limit": 10.0, "near_zero_radius": 0.1,
         "near_zero_tail_steps": 1},
        {"horizons": (2,), "explosion_limit": 0.0, "near_zero_radius": 0.1,
         "near_zero_tail_steps": 1},
        {"horizons": (2,), "explosion_limit": 10.0, "near_zero_radius": -0.1,
         "near_zero_tail_steps": 1},
        {"horizons": (2,), "explosion_limit": 10.0, "near_zero_radius": 0.1,
         "near_zero_tail_steps": 3},
    ],
)
def test_invalid_diagnostic_configuration_is_rejected(arguments: dict) -> None:
    with pytest.raises(ValueError):
        StabilityDiagnosticConfig(**arguments)
