"""Experimental AST/KernelSynth channels with explicit supervision availability.

The channel-family choice is separate from the AST treatment. New families can
be added through the sampler and path-drawer dispatch below.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import numpy as np

from tsfmxai.generators.kernel_synth import official_kernel_bank, covariance_matrix, stable_factor
from tsfmxai.generators.multiseries import (
    MultiseriesConfig, MultiseriesSystem, draw_choice, sample_multiseries_system,
    stream_seed, validate_probabilities,
)
from tsfmxai.generators.trajectory import compile_bound_expression


CHANNEL_FAMILIES = ("ast", "kernel_synth")


@dataclass(frozen=True)
class HybridConfig:
    ast: MultiseriesConfig
    family_probs: Mapping[str, float]

    def __post_init__(self):
        validate_probabilities(self.family_probs, choices=CHANNEL_FAMILIES)


@dataclass(frozen=True)
class HybridSystem:
    base: MultiseriesSystem
    channel_families: tuple[str, ...]
    kernel_specs: tuple[tuple[tuple[int, ...], tuple[str, ...]] | None, ...]
    config: HybridConfig | None = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)


def sample_hybrid_system(config: HybridConfig, *, seed: int) -> HybridSystem:
    """Draw channel families first, then sample ASTs only for AST channels.

    Family draws use their own per-channel streams, so they do not depend on
    the channel count. KernelSynth channels receive a zero placeholder instead
    of a tree; AST channels are identical to the plain multiseries sampler.
    """
    max_channels = max(c for c, p in config.ast.channel_count_probs.items() if p > 0)
    families, specs = [], []
    for channel in range(max_channels):
        rng = np.random.default_rng(stream_seed(seed, 91, channel))
        family = draw_choice(rng, config.family_probs)
        families.append(family)
        if family == "kernel_synth":
            count = int(rng.integers(1, 6))
            indices = tuple(int(v) for v in rng.integers(0, 33, size=count))
            operators = tuple(str(v) for v in rng.choice(("+", "*"), size=count - 1))
            specs.append((indices, operators))
        else:
            specs.append(None)
    exogenous = frozenset(c for c, family in enumerate(families) if family == "kernel_synth")
    base = sample_multiseries_system(config.ast, seed=seed, exogenous_channels=exogenous)
    n = base.num_channels
    return HybridSystem(base, tuple(families[:n]), tuple(specs[:n]), config)


def kernel_formula(spec: tuple[tuple[int, ...], tuple[str, ...]]) -> str:
    """Readable KernelSynth composition using the official bank identifiers."""
    bank = official_kernel_bank()
    indices, operators = spec
    parts = [bank[indices[0]]["id"]]
    for operator, index in zip(operators, indices[1:]):
        parts.extend((operator, bank[index]["id"]))
    return " ".join(parts)


@lru_cache(maxsize=64)
def _factor(spec: tuple[tuple[int, ...], tuple[str, ...]], length: int) -> np.ndarray:
    bank = official_kernel_bank(length)
    indices, operators = spec
    matrix = covariance_matrix(bank[indices[0]]["kernel"], length)
    for operator, index in zip(operators, indices[1:]):
        component = covariance_matrix(bank[index]["kernel"], length)
        matrix = matrix + component if operator == "+" else matrix * component
    return stable_factor(matrix)


def _draw_kernel_synth(spec, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = _factor(spec, length) @ rng.standard_normal(length)
    scale = float(np.std(values))
    return (values - np.mean(values)) / max(scale, 1e-8)


def simulate_hybrid_path(system: HybridSystem, initial: np.ndarray,
                         innovations: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Use full GP prefixes as lag history; return only post-prefix steps."""
    length, steps = initial.shape[1], innovations.shape[1]
    history = np.array(initial, copy=True)
    exogenous = {}
    for channel, family in enumerate(system.channel_families):
        if family == "kernel_synth":
            values = _draw_kernel_synth(system.kernel_specs[channel], length + steps,
                                        stream_seed(seed, channel))
            history[channel] = values[:length]
            exogenous[channel] = values[length:]
    # Python floats keep the compiled equations bitwise equal to the recursive
    # evaluator; tolist() preserves every value exactly.
    histories = [row.tolist() for row in history]
    channels = [(channel, exogenous[channel].tolist() if family == "kernel_synth" else None,
                 None if family == "kernel_synth" else compile_bound_expression(
                     system.expressions[channel], system.bindings[channel]),
                 innovations[channel].tolist())
                for channel, family in enumerate(system.channel_families)]
    result = np.empty((system.num_channels, steps), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(steps):
            time = length + step
            for channel, external, equation, noise in channels:
                result[channel, step] = (external[step] if equation is None
                                         else equation(histories, time) + noise[step])
            column = result[:, step]
            if not np.isfinite(column).all():
                return result[:, :step + 1], history
            for channel, value in enumerate(column):
                histories[channel].append(float(value))
    return result, history
