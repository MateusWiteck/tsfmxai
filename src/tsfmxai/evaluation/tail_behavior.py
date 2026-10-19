"""Finite-window flatness, persistence, and initial-to-terminal change diagnostics.

Windows are measured on signed channels separately. Centering a magnitude
envelope would incorrectly classify sign-changing trajectories as constant.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from tsfmxai.evaluation.stability import StabilityDiagnosticConfig, diagnose_stability_prefixes


@dataclass(frozen=True)
class TailDiagnosticConfig:
    """Explicit operational tolerances, rather than an asymptotic convergence test."""

    window_steps: int
    persistence_windows: int
    flat_tolerance: float
    decline_ratio: float = 0.5

    def __post_init__(self) -> None:
        for name, minimum in (("window_steps", 2), ("persistence_windows", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite(self.flat_tolerance) or self.flat_tolerance < 0:
            raise ValueError("flat_tolerance must be finite and non-negative.")
        if not np.isfinite(self.decline_ratio) or not 0 < self.decline_ratio < 1:
            raise ValueError("decline_ratio must lie strictly between zero and one.")

    def validate_horizons(self, horizons: tuple[int, ...]) -> None:
        if min(horizons) < (self.persistence_windows + 1) * self.window_steps:
            raise ValueError("The initial window and terminal persistence windows must not overlap.")


@dataclass(frozen=True)
class TailWindowStatistics:
    """Per-channel measurements; failed prefixes have no eligible tail statistics."""

    horizon: int
    tail_eligible: bool
    initial_mean: tuple[float, ...] | None = None
    initial_variation: tuple[float, ...] | None = None
    initial_max_abs: tuple[float, ...] | None = None
    tail_mean: tuple[float, ...] | None = None
    tail_variation: tuple[float, ...] | None = None
    recent_window_means: tuple[tuple[float, ...], ...] | None = None
    recent_window_variations: tuple[tuple[float, ...], ...] | None = None
    persistent_mean: tuple[float, ...] | None = None
    persistent_variation: tuple[float, ...] | None = None
    persistent_max_abs: tuple[float, ...] | None = None


TAIL_LABELS = (
    "flat_tail", "persistent_flat_tail", "quiet_initial_and_tail",
    "flattening", "flat_nonzero", "persistent_near_zero", "amplitude_decline_near_zero",
)


def measure_tail_prefixes(
    generated_values: np.ndarray,
    stability_config: StabilityDiagnosticConfig,
    tail_config: TailDiagnosticConfig,
) -> tuple[TailWindowStatistics, ...]:
    """Measure each successful prefix using an early window and terminal windows.

    Input is [series, time], or a scalar series. Explosions use the shared
    evaluator, including retained terminal failures. The persistence window
    contains ``persistence_windows * window_steps`` consecutive observations.
    Its pooled variation additionally detects changes between flat plateaus.
    """

    tail_config.validate_horizons(stability_config.horizons)
    values = np.asarray(generated_values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("generated_values must have shape [series, time] with at least one series.")
    diagnostics = diagnose_stability_prefixes(np.max(np.abs(values), axis=0), stability_config)
    width, windows = tail_config.window_steps, tail_config.persistence_windows

    def summarize(block: np.ndarray) -> tuple[tuple[float, ...], ...]:
        mean = block.mean(axis=1)
        variation = np.max(np.abs(block - mean[:, None]), axis=1)
        amplitude = np.max(np.abs(block), axis=1)
        return tuple(mean), tuple(variation), tuple(amplitude)

    results = []
    for diagnostic in diagnostics:
        h = diagnostic.horizon
        if diagnostic.exploded:
            results.append(TailWindowStatistics(h, False))
            continue
        early_mean, early_variation, early_amplitude = summarize(values[:, :width])
        tail_mean, tail_variation, _ = summarize(values[:, h-width:h])
        persistent = values[:, h-windows*width:h]
        mean, variation, amplitude = summarize(persistent)
        blocks = [summarize(persistent[:, j*width:(j+1)*width]) for j in range(windows)]
        results.append(TailWindowStatistics(
            horizon=h, tail_eligible=True,
            initial_mean=early_mean, initial_variation=early_variation,
            initial_max_abs=early_amplitude, tail_mean=tail_mean, tail_variation=tail_variation,
            recent_window_means=tuple(block[0] for block in blocks),
            recent_window_variations=tuple(block[1] for block in blocks),
            persistent_mean=mean, persistent_variation=variation, persistent_max_abs=amplitude,
        ))
    return tuple(results)


def classify_tail_windows(
    statistics: TailWindowStatistics,
    config: TailDiagnosticConfig,
    *,
    near_zero_radius: float,
) -> dict[str, bool]:
    """Classify saved statistics, also allowing tolerance sensitivity without regeneration.

    All channels must be quiet for flatness. A reduction requires at least one
    initially active channel and the configured drop in every such channel.
    Labels overlap; every rate uses all attempted trajectories unless explicitly
    conditioned on ``tail_eligible``. Failed prefixes contribute false labels.
    """

    if not np.isfinite(near_zero_radius) or near_zero_radius < 0:
        raise ValueError("near_zero_radius must be finite and non-negative.")
    if not statistics.tail_eligible:
        return dict.fromkeys(TAIL_LABELS, False)
    early = np.asarray(statistics.initial_variation)
    persistent = np.asarray(statistics.persistent_variation)
    initial_amplitude = np.asarray(statistics.initial_max_abs)
    final_amplitude = np.asarray(statistics.persistent_max_abs)
    delta, ratio = config.flat_tolerance, config.decline_ratio
    flat = bool(np.all(np.asarray(statistics.tail_variation) <= delta))
    persistent_flat = bool(
        flat and np.all(np.asarray(statistics.recent_window_variations) <= delta)
        and np.all(persistent <= delta)
    )
    active_variation = early > delta
    active_amplitude = initial_amplitude > near_zero_radius
    persistent_near_zero = bool(np.all(final_amplitude <= near_zero_radius))
    return dict(
        flat_tail=flat,
        persistent_flat_tail=persistent_flat,
        quiet_initial_and_tail=bool(persistent_flat and not np.any(active_variation)),
        flattening=bool(persistent_flat and np.any(active_variation)
                        and np.all(persistent[active_variation] <= ratio * early[active_variation])),
        flat_nonzero=bool(persistent_flat
                         and np.any(np.abs(statistics.tail_mean) > near_zero_radius)),
        persistent_near_zero=persistent_near_zero,
        amplitude_decline_near_zero=bool(
            persistent_near_zero and np.any(active_amplitude)
            and np.all(final_amplitude[active_amplitude] <= ratio * initial_amplitude[active_amplitude])
        ),
    )
