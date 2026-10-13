"""Shared AST execution with explicit innovations for controlled experiments."""

from collections.abc import Sequence

import numpy as np

from tsfmxai.generators.trajectory import (
    evaluate_bound_expression,
    required_multivariate_history_length,
)
from tsfmxai.representations.expression import ExpressionNode, MechanismBinding


def simulate_innovation_path(
    expressions: Sequence[ExpressionNode],
    bindings: Sequence[MechanismBinding],
    initial_values: np.ndarray,
    innovations: np.ndarray,
    *,
    stop_threshold: float | None = None,
) -> np.ndarray:
    """Return generated values [series, time], retaining the first failing step.

    Histories are chronological; simultaneous equations use the same past.
    A non-finite output (or a threshold crossing when requested) stops all
    channels together. Keeping that value makes failure diagnostics auditable.
    The supplied innovations are explicit, so no global RNG is consulted.
    """

    required = required_multivariate_history_length(expressions, bindings)
    initial = np.asarray(initial_values, dtype=float)
    noise = np.asarray(innovations, dtype=float)
    n = len(expressions)
    if initial.ndim != 2 or initial.shape[0] != n or initial.shape[1] < required:
        raise ValueError("initial_values must have shape [series, history >= required lag].")
    if noise.ndim != 2 or noise.shape[0] != n:
        raise ValueError("innovations must have shape [series, generated steps].")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(noise)):
        raise ValueError("Histories and innovations must be finite.")
    if stop_threshold is not None and (
        not np.isfinite(stop_threshold) or stop_threshold <= 0
    ):
        raise ValueError("stop_threshold must be finite and positive.")
    length = initial.shape[1]
    histories = [list(row) for row in initial]
    result = np.empty_like(noise)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(noise.shape[1]):
            for target, (tree, binding) in enumerate(zip(expressions, bindings)):
                result[target, step] = evaluate_bound_expression(
                    tree, binding, time=length + step, histories=histories,
                ) + noise[target, step]
            column = result[:, step]
            if not np.all(np.isfinite(column)) or (
                stop_threshold is not None and np.any(np.abs(column) > stop_threshold)
            ):
                return result[:, : step + 1]
            for target, value in enumerate(column):
                histories[target].append(float(value))
    return result
