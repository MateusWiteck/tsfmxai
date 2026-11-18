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


FAMILIES = ("constant", "noise", "cosine")


def _interval(value, name, *, integer=False, minimum=0.):
    low, high = value
    if integer and not all(isinstance(v, int) and not isinstance(v, bool) for v in (low, high)):
        raise ValueError(f"{name} must be an integer interval.")
    if not np.isfinite([low, high]).all() or not minimum <= low <= high:
        raise ValueError(f"{name} must be a finite interval with {minimum} <= low <= high.")


@dataclass(frozen=True)
class RandomHistorySpec:
    """Per-channel startup assembled from randomly present families.

    For every channel, each family appears independently with its probability;
    the parameters of every family are then drawn from their intervals and the
    present components are summed. When no family appears the history is zero.

    * constant: a level ``s * m`` with a random sign ``s`` and ``m ~ U(constant_magnitude)``;
    * noise: i.i.d. white noise ``U(-r, r)`` per lag with ``r ~ U(noise_radius)``;
    * cosine: ``A cos(2 pi k / T + phi)`` with ``A ~ U(cosine_amplitude)``,
      integer ``T ~ U{cosine_period}`` and phase ``phi ~ U(0, 2 pi)``. It is
      evaluated directly on the required history length, so no prefix has to
      cover the lag memory.

    Each retry draws families and parameters again. The Heaps stationary law is
    deliberately excluded: it exists only for Heaps VAR mechanisms.
    """

    name: str = "random families"
    constant_probability: float = .5
    constant_magnitude: tuple[float, float] = (0., 1.)
    noise_probability: float = .5
    noise_radius: tuple[float, float] = (.1, 1.)
    cosine_probability: float = .5
    cosine_amplitude: tuple[float, float] = (.1, 1.)
    cosine_period: tuple[int, int] = (4, 48)

    kind = "random"

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ValueError("name must be a string; pass family settings by keyword.")
        for family in FAMILIES:
            p = getattr(self, f"{family}_probability")
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or not 0. <= p <= 1.:
                raise ValueError(f"{family}_probability must lie in [0, 1].")
        _interval(self.constant_magnitude, "constant_magnitude")
        _interval(self.noise_radius, "noise_radius")
        _interval(self.cosine_amplitude, "cosine_amplitude")
        _interval(self.cosine_period, "cosine_period", integer=True, minimum=2)


def random_history_bound(spec):
    """Largest absolute value a ``RandomHistorySpec`` history can reach.

    Families that can appear (positive probability) add their maximum magnitude;
    the bound is attained only if all of them line up, so it is conservative.
    """
    parts = ((spec.constant_probability, spec.constant_magnitude[1]),
             (spec.noise_probability, spec.noise_radius[1]),
             (spec.cosine_probability, spec.cosine_amplitude[1]))
    return float(sum(high for p, high in parts if p > 0))


def sample_random_history(spec, channels, length, rng):
    """Return a ``(channels, length)`` history and the components of each channel.

    The draw order is fixed (presence of all families, then all parameters), so
    the parameters of one family do not depend on which other families appear.
    """
    history = np.zeros((channels, length), dtype=np.float64)
    components = []
    k = np.arange(length)
    for channel in range(channels):
        present = {f: bool(rng.random() < getattr(spec, f"{f}_probability")) for f in FAMILIES}
        sign = -1. if rng.random() < .5 else 1.
        level = sign * rng.uniform(*spec.constant_magnitude)
        radius = rng.uniform(*spec.noise_radius)
        noise = rng.uniform(-1., 1., length) * radius
        amplitude = rng.uniform(*spec.cosine_amplitude)
        period = int(rng.integers(spec.cosine_period[0], spec.cosine_period[1] + 1))
        phase = rng.uniform(0., 2 * np.pi)
        parts = []
        if present["constant"]:
            history[channel] += level
            parts.append(dict(family="constant", level=float(level)))
        if present["noise"]:
            history[channel] += noise
            parts.append(dict(family="noise", radius=float(radius)))
        if present["cosine"]:
            history[channel] += amplitude * np.cos(2 * np.pi * k / period + phase)
            parts.append(dict(family="cosine", amplitude=float(amplitude), period=period,
                              phase=float(phase)))
        components.append(parts)
    return history, components


def make_initial_history(spec, mechanism, unit_history, stationary_seed):
    """Construct the declared history without clipping or changing the mechanism."""
    if isinstance(spec, RandomHistorySpec):
        channels, length = unit_history.shape
        return sample_random_history(spec, channels, length,
                                     np.random.default_rng(stationary_seed))[0]
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
