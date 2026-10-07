"""Finite-horizon stability diagnostics for generated time-series trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StabilityDiagnosticConfig:
    """Define the operational rules used by a finite-horizon stability study."""

    horizons: tuple[int, ...]
    explosion_limit: float
    near_zero_radius: float
    near_zero_tail_steps: int

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon < 1 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers.")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be strictly increasing and unique.")
        if not np.isfinite(self.explosion_limit) or self.explosion_limit <= 0.0:
            raise ValueError("explosion_limit must be finite and positive.")
        if not np.isfinite(self.near_zero_radius) or self.near_zero_radius < 0.0:
            raise ValueError("near_zero_radius must be finite and non-negative.")
        if self.near_zero_tail_steps < 1:
            raise ValueError("near_zero_tail_steps must be positive.")
        if self.near_zero_tail_steps > min(self.horizons):
            raise ValueError("near_zero_tail_steps cannot exceed the smallest horizon.")


@dataclass(frozen=True)
class StabilityDiagnostic:
    """Classify one generated trajectory prefix at one evaluation horizon."""

    horizon: int
    exploded: bool
    near_zero: bool
    explosion_step: int | None

    @property
    def neither(self) -> bool:
        """Return whether the trajectory is neither exploded nor near zero."""

        return not self.exploded and not self.near_zero


def first_explosion_step(
    generated_values: Sequence[float],
    *,
    explosion_limit: float,
) -> int | None:
    """Return the one-based first non-finite or over-limit generated step."""

    if not np.isfinite(explosion_limit) or explosion_limit <= 0.0:
        raise ValueError("explosion_limit must be finite and positive.")
    values = np.asarray(generated_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("generated_values must be one-dimensional.")
    failing = np.flatnonzero(~np.isfinite(values) | (np.abs(values) > explosion_limit))
    return None if failing.size == 0 else int(failing[0] + 1)


def diagnose_stability_prefixes(
    generated_values: Sequence[float],
    config: StabilityDiagnosticConfig,
) -> tuple[StabilityDiagnostic, ...]:
    """Classify explosion and near-zero behavior at every configured horizon.

    ``generated_values`` may stop at its first explosive value. It must otherwise
    cover the largest requested horizon. Near-zero is an operational finite-window
    label: every value in the final configured tail must lie inside the radius.
    """

    values = np.asarray(generated_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("generated_values must be one-dimensional.")
    explosion_step = first_explosion_step(
        values,
        explosion_limit=config.explosion_limit,
    )
    if explosion_step is None and values.size < max(config.horizons):
        raise ValueError("A non-exploded trajectory must cover the largest horizon.")

    diagnostics: list[StabilityDiagnostic] = []
    for horizon in config.horizons:
        exploded = explosion_step is not None and explosion_step <= horizon
        if exploded:
            near_zero = False
        else:
            if values.size < horizon:
                raise ValueError(
                    "The trajectory ended before a requested pre-explosion horizon."
                )
            tail = values[horizon - config.near_zero_tail_steps : horizon]
            near_zero = bool(np.max(np.abs(tail)) <= config.near_zero_radius)
        diagnostics.append(
            StabilityDiagnostic(
                horizon=horizon,
                exploded=exploded,
                near_zero=near_zero,
                explosion_step=explosion_step,
            )
        )
    return tuple(diagnostics)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z_score: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if trials < 1:
        raise ValueError("trials must be positive.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie between zero and trials.")
    if not np.isfinite(z_score) or z_score <= 0.0:
        raise ValueError("z_score must be finite and positive.")

    proportion = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (proportion + z_score**2 / (2.0 * trials)) / denominator
    half_width = (
        z_score
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_score**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return center - half_width, center + half_width
