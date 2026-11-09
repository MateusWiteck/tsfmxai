"""Startup laws shared by training generation and initialization experiments.

Histories are chronological. Cosine prefixes are not resampled or padded.
"""

from dataclasses import dataclass
import numpy as np
from tsfmxai.generators.heaps import HeapsVARMechanism, initialize_heaps_history

@dataclass(frozen=True)
class HistorySpec:
    """An explicit chronological startup law, separate from the recurrence.

    Cosine histories are prefixes of one cycle sampled at integer k. A fraction
    of 1/2 keeps the *first* half of that same cycle, without padding, resampling,
    or changing its period. Each prefix must cover the required lag history.
    """

    name: str
    kind: str
    amplitude: float = 1.
    offset: float = 0.
    period: int = 48
    fraction: float = 1.

    def __post_init__(self):
        if self.kind not in {"zero", "constant", "uniform", "stationary", "cosine"}:
            raise ValueError("Unknown history kind.")
        if not np.isfinite([self.amplitude, self.offset]).all() or self.amplitude < 0:
            raise ValueError("History amplitude must be finite and nonnegative; offset finite.")
        if self.kind == "cosine":
            if (not isinstance(self.period, int) or self.period < 4
                    or self.fraction not in (1., .5, .25)
                    or not float(self.period * self.fraction).is_integer()):
                raise ValueError("Cosine requires an integer period and a whole number of prefix samples.")


def make_initial_history(spec, mechanism, unit_history, stationary_seed):
    """Construct the declared history without clipping or changing the mechanism."""
    if spec.kind == "zero":
        return np.zeros_like(unit_history) + spec.offset
    if spec.kind == "constant":
        return np.full_like(unit_history, spec.offset)
    if spec.kind == "uniform":
        return spec.offset + spec.amplitude * unit_history
    if spec.kind == "stationary":
        if not isinstance(mechanism, HeapsVARMechanism):
            raise ValueError("Joint stationary initializer available only for Heaps.")
        return spec.offset + spec.amplitude * initialize_heaps_history(mechanism, seed=stationary_seed)
    length = int(spec.period * spec.fraction)
    if length < unit_history.shape[1]:
        raise ValueError("Cosine prefix is shorter than the required history.")
    values = spec.offset + spec.amplitude * np.cos(2 * np.pi * np.arange(length) / spec.period)
    return np.broadcast_to(values, (unit_history.shape[0], length)).copy()
