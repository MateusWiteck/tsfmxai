"""Reproducible random sampling of factorized expression trees."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np

from tsfmxai.representations.expression import (
    Add,
    ExpressionNode,
    Lag,
    MechanismBinding,
    Multiply,
    Parameter,
    Time,
    canonicalize_and_relabel,
    format_tree,
    lag_slots,
    operator_count,
    parameter_slots,
    to_bound_formula,
    to_skeleton_formula,
)


@dataclass(frozen=True)
class RandomTreeConfig:
    """Controls topology, terminals, operators, and separately bound values."""

    minimum_operators: int = 1
    maximum_operators: int = 6
    num_series: int = 1
    minimum_lag: int = 1
    maximum_lag: int = 12
    parameter_probability: float = 0.40
    time_probability: float = 0.15
    lag_probability: float = 0.45
    add_probability: float = 0.65
    minimum_parameter_magnitude: float = 0.05
    maximum_parameter_magnitude: float = 0.90
    require_lag: bool = True
    require_time: bool = False

    def __post_init__(self) -> None:
        if self.minimum_operators < 0:
            raise ValueError("minimum_operators must be non-negative.")
        if self.maximum_operators < self.minimum_operators:
            raise ValueError("maximum_operators must be at least minimum_operators.")
        if self.num_series < 1:
            raise ValueError("num_series must be positive.")
        if self.minimum_lag < 1 or self.maximum_lag < self.minimum_lag:
            raise ValueError("The lag interval must contain positive integers.")
        terminal_total = (
            self.parameter_probability + self.time_probability + self.lag_probability
        )
        if terminal_total <= 0:
            raise ValueError("At least one terminal probability must be positive.")
        if any(
            probability < 0
            for probability in (
                self.parameter_probability,
                self.time_probability,
                self.lag_probability,
            )
        ):
            raise ValueError("Terminal probabilities must be non-negative.")
        if not 0 <= self.add_probability <= 1:
            raise ValueError("add_probability must lie in [0, 1].")
        if not 0 < self.minimum_parameter_magnitude <= self.maximum_parameter_magnitude:
            raise ValueError("Parameter magnitudes must define a positive interval.")
        minimum_leaf_count = self.minimum_operators + 1
        required_kinds = int(self.require_lag) + int(self.require_time)
        if required_kinds > minimum_leaf_count:
            raise ValueError("The minimum tree size cannot contain all required terminals.")


@dataclass(frozen=True)
class SampledMechanism:
    """A sampled structure together with separately sampled numeric bindings."""

    expression: ExpressionNode
    binding: MechanismBinding
    seed: int
    sample_index: int | None

    @property
    def skeleton_formula(self) -> str:
        return to_skeleton_formula(self.expression)

    @property
    def formula(self) -> str:
        return to_bound_formula(self.expression, self.binding)

    @property
    def tree(self) -> str:
        return format_tree(self.expression)


@lru_cache(maxsize=None)
def full_binary_tree_shape_count(operator_nodes: int) -> int:
    """Return the Catalan number counting full binary shapes with n operators."""

    if operator_nodes < 0:
        raise ValueError("operator_nodes must be non-negative.")
    return comb(2 * operator_nodes, operator_nodes) // (operator_nodes + 1)


def derive_sample_seed(root_seed: int, sample_index: int) -> int:
    """Derive an order-independent seed suitable for parallel sample generation."""

    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    sequence = np.random.SeedSequence([int(root_seed), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _sample_terminal_kinds(
    rng: np.random.Generator,
    leaf_count: int,
    config: RandomTreeConfig,
) -> list[str]:
    probabilities = np.asarray(
        [config.parameter_probability, config.time_probability, config.lag_probability],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    kinds = list(
        rng.choice(
            np.asarray(["parameter", "time", "lag"], dtype=object),
            size=leaf_count,
            p=probabilities,
        )
    )

    reserved_indices: set[int] = set()
    if config.require_lag:
        if "lag" in kinds:
            lag_index = kinds.index("lag")
        else:
            lag_index = int(rng.integers(leaf_count))
            kinds[lag_index] = "lag"
        reserved_indices.add(lag_index)
    if config.require_time:
        if "time" in kinds and kinds.index("time") not in reserved_indices:
            time_index = kinds.index("time")
        else:
            choices = [index for index in range(leaf_count) if index not in reserved_indices]
            time_index = int(rng.choice(choices))
            kinds[time_index] = "time"
        reserved_indices.add(time_index)
    return kinds


def _make_terminals(
    rng: np.random.Generator,
    kinds: list[str],
    config: RandomTreeConfig,
) -> list[ExpressionNode]:
    terminals: list[ExpressionNode] = []
    parameter_index = 0
    lag_index = 0
    for kind in kinds:
        if kind == "parameter":
            terminals.append(Parameter(f"temporary_parameter_{parameter_index}"))
            parameter_index += 1
        elif kind == "time":
            terminals.append(Time())
        elif kind == "lag":
            variable = int(rng.integers(config.num_series))
            terminals.append(Lag(variable, f"temporary_lag_{lag_index}"))
            lag_index += 1
        else:
            raise ValueError(f"Unknown terminal kind: {kind}")
    return terminals


def _sample_full_binary_tree(
    rng: np.random.Generator,
    operator_nodes: int,
    terminals: list[ExpressionNode],
    add_probability: float,
) -> ExpressionNode:
    if operator_nodes == 0:
        if len(terminals) != 1:
            raise ValueError("A terminal subtree must contain exactly one terminal.")
        return terminals[0]

    possible_left_counts = np.arange(operator_nodes)
    shape_weights = np.asarray(
        [
            full_binary_tree_shape_count(int(left_count))
            * full_binary_tree_shape_count(operator_nodes - 1 - int(left_count))
            for left_count in possible_left_counts
        ],
        dtype=float,
    )
    shape_weights /= shape_weights.sum()
    left_operator_count = int(rng.choice(possible_left_counts, p=shape_weights))
    right_operator_count = operator_nodes - 1 - left_operator_count
    left_leaf_count = left_operator_count + 1

    left = _sample_full_binary_tree(
        rng,
        left_operator_count,
        terminals[:left_leaf_count],
        add_probability,
    )
    right = _sample_full_binary_tree(
        rng,
        right_operator_count,
        terminals[left_leaf_count:],
        add_probability,
    )
    if rng.random() < add_probability:
        return Add(left, right)
    return Multiply(left, right)


def _sample_parameter_value(rng: np.random.Generator, config: RandomTreeConfig) -> float:
    magnitude = rng.uniform(
        config.minimum_parameter_magnitude,
        config.maximum_parameter_magnitude,
    )
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(sign * magnitude)


def sample_random_mechanism(
    config: RandomTreeConfig,
    *,
    seed: int,
    sample_index: int | None = None,
) -> SampledMechanism:
    """Sample one canonical tree and independent coefficient/lag bindings."""

    rng = np.random.default_rng(seed)
    binary_operators = int(
        rng.integers(config.minimum_operators, config.maximum_operators + 1)
    )
    terminal_kinds = _sample_terminal_kinds(rng, binary_operators + 1, config)
    terminals = _make_terminals(rng, terminal_kinds, config)
    expression = _sample_full_binary_tree(
        rng,
        binary_operators,
        terminals,
        config.add_probability,
    )
    expression = canonicalize_and_relabel(expression)

    parameter_values = {
        slot: _sample_parameter_value(rng, config) for slot in parameter_slots(expression)
    }
    lag_values = {
        slot: int(rng.integers(config.minimum_lag, config.maximum_lag + 1))
        for slot in lag_slots(expression)
    }
    binding = MechanismBinding(parameter_values=parameter_values, lag_values=lag_values)
    return SampledMechanism(
        expression=expression,
        binding=binding,
        seed=int(seed),
        sample_index=sample_index,
    )


def sample_random_mechanisms(
    config: RandomTreeConfig,
    *,
    root_seed: int,
    count: int,
) -> list[SampledMechanism]:
    """Sample an index-stable batch that can later be generated in parallel."""

    if count < 0:
        raise ValueError("count must be non-negative.")
    return [
        sample_random_mechanism(
            config,
            seed=derive_sample_seed(root_seed, sample_index),
            sample_index=sample_index,
        )
        for sample_index in range(count)
    ]


def sampled_operator_count(mechanism: SampledMechanism) -> int:
    """Convenience helper for experiment summaries."""

    return operator_count(mechanism.expression)
