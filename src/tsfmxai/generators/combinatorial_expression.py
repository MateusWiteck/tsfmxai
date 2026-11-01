"""Count and uniformly sample finite, ordered polynomial AST structures.

Every lag atom (series, delay) is a discrete terminal label. Each parameter
occurrence is one placeholder whose continuous value is bound after sampling.
Uniformity is over ASTs before polynomial canonicalization, not over supports.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
import random

import numpy as np

from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.representations.expression import (
    Add, ExpressionNode, Lag, MechanismBinding, Multiply, Parameter,
)


@dataclass(frozen=True)
class CombinatorialTreeConfig:
    minimum_operators: int = 1
    maximum_operators: int = 6
    num_series: int = 1
    minimum_lag: int = 1
    maximum_lag: int = 12
    max_degree: int = 4
    require_lag: bool = True
    minimum_parameter_magnitude: float = 0.05
    maximum_parameter_magnitude: float = 0.90

    def __post_init__(self) -> None:
        for name in ("minimum_operators", "maximum_operators", "num_series",
                     "minimum_lag", "maximum_lag", "max_degree"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer.")
        if not 0 <= self.minimum_operators <= self.maximum_operators:
            raise ValueError("Operator bounds must satisfy 0 <= minimum <= maximum.")
        if self.num_series < 1 or not 1 <= self.minimum_lag <= self.maximum_lag:
            raise ValueError("Positive series count and valid positive lag bounds required.")
        if self.max_degree < 0 or (self.require_lag and self.max_degree == 0):
            raise ValueError("Degree bound must allow the requested terminals.")
        if not isinstance(self.require_lag, bool):
            raise TypeError("require_lag must be a boolean.")
        if not (isfinite(self.minimum_parameter_magnitude)
                and isfinite(self.maximum_parameter_magnitude)
                and 0 < self.minimum_parameter_magnitude <= self.maximum_parameter_magnitude):
            raise ValueError("Parameter magnitudes must be finite and positive.")


@dataclass(frozen=True)
class StructureCounts:
    """Totals over a class of ordered ASTs, using exact Python integers."""

    trees: int = 0
    addition_nodes: int = 0
    parameter_leaves: int = 0


class CombinatorialTreeSampler:
    """Precompute completion counts and sample by uniform integer unranking.

    For n operators and degree d, addition uses max(d_left, d_right),
    multiplication uses d_left + d_right. Each decomposition receives the
    product of its left and right completion counts. All root states within
    the configured bounds are weighted by their counts, including size.
    """

    def __init__(self, config: CombinatorialTreeConfig = CombinatorialTreeConfig()):
        self.config = config
        self.lag_count = config.maximum_lag - config.minimum_lag + 1
        self.atom_count = config.num_series * self.lag_count
        self.counts: dict[tuple[int, int], StructureCounts] = {}
        self._branches: dict[tuple[int, int], list[tuple[str, int, int, int]]] = {}
        self._cumulative: dict[tuple[int, int], list[int]] = {}
        for degree in range(config.max_degree + 1):
            self.counts[0, degree] = (
                StructureCounts(1, 0, 1) if degree == 0 else
                StructureCounts(self.atom_count, 0, 0) if degree == 1 else
                StructureCounts()
            )
        for n in range(1, config.maximum_operators + 1):
            for degree in range(config.max_degree + 1):
                count = additions = parameters = 0
                branches, cumulative = [], []
                for left_n in range(n):
                    right_n = n - 1 - left_n
                    for left_d in range(config.max_degree + 1):
                        left = self.counts[left_n, left_d]
                        if not left.trees:
                            continue
                        for right_d in range(config.max_degree + 1):
                            right = self.counts[right_n, right_d]
                            if not right.trees:
                                continue
                            for op, result_degree in (("+", max(left_d, right_d)),
                                                      ("*", left_d + right_d)):
                                if result_degree != degree:
                                    continue
                                weight = left.trees * right.trees
                                count += weight
                                additions += (left.addition_nodes * right.trees
                                              + left.trees * right.addition_nodes
                                              + (weight if op == "+" else 0))
                                parameters += (left.parameter_leaves * right.trees
                                               + left.trees * right.parameter_leaves)
                                branches.append((op, left_n, left_d, right_d))
                                cumulative.append(count)
                self.counts[n, degree] = StructureCounts(count, additions, parameters)
                self._branches[n, degree] = branches
                self._cumulative[n, degree] = cumulative
        self.root_states: tuple[tuple[int, int], ...] = tuple(
            (n, degree)
            for n in range(config.minimum_operators, config.maximum_operators + 1)
            for degree in range(int(config.require_lag), config.max_degree + 1)
            if self.counts[n, degree].trees
        )
        self._root_cumulative = []
        total = 0
        for state in self.root_states:
            total += self.counts[state].trees
            self._root_cumulative.append(total)
        self.total_structures = total

    def size_counts(self) -> dict[int, int]:
        return {
            n: sum(self.counts[n, d].trees for d in range(
                int(self.config.require_lag), self.config.max_degree + 1))
            for n in range(self.config.minimum_operators, self.config.maximum_operators + 1)
        }

    def marginal_profile(self) -> dict[str, float]:
        """Pooled-node fractions; local sampling probabilities depend on state."""

        operators = sum(n * self.counts[n, d].trees for n, d in self.root_states)
        leaves = operators + self.total_structures
        additions = sum(self.counts[s].addition_nodes for s in self.root_states)
        parameters = sum(self.counts[s].parameter_leaves for s in self.root_states)
        return {
            "mean_operators": operators / self.total_structures,
            "addition_fraction": additions / operators if operators else 0.0,
            "parameter_fraction": parameters / leaves,
        }

    def unrank(self, rank: int, *, seed: int = 0,
               sample_index: int | None = None) -> SampledMechanism:
        """Map each integer in [0, total_structures) to one discrete AST.

        The seed affects continuous parameter values only. Left/right order
        and repeated lag leaves are retained. Each parameter leaf is distinct.
        """

        if type(rank) is not int or not 0 <= rank < self.total_structures:
            raise ValueError("Rank is outside the counted structure space.")
        root_index = bisect_right(self._root_cumulative, rank)
        root_previous = self._root_cumulative[root_index - 1] if root_index else 0
        n, degree = self.root_states[root_index]
        parameter_slots: list[str] = []
        lag_values: dict[str, int] = {}

        def build(n: int, degree: int, local_rank: int) -> ExpressionNode:
            if n == 0:
                if degree == 0:
                    slot = f"P{len(parameter_slots)}"
                    parameter_slots.append(slot)
                    return Parameter(slot)
                series, lag_offset = divmod(local_rank, self.lag_count)
                slot = f"L{len(lag_values)}"
                lag_values[slot] = self.config.minimum_lag + lag_offset
                return Lag(series, slot)
            cumulative = self._cumulative[n, degree]
            branch_index = bisect_right(cumulative, local_rank)
            previous = cumulative[branch_index - 1] if branch_index else 0
            op, left_n, left_d, right_d = self._branches[n, degree][branch_index]
            right_n = n - 1 - left_n
            left_rank, right_rank = divmod(local_rank - previous,
                                          self.counts[right_n, right_d].trees)
            left = build(left_n, left_d, left_rank)
            right = build(right_n, right_d, right_rank)
            return Add(left, right) if op == "+" else Multiply(left, right)

        expression = build(n, degree, rank - root_previous)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261030]))
        parameters = {}
        for slot in parameter_slots:
            magnitude = rng.uniform(self.config.minimum_parameter_magnitude,
                                    self.config.maximum_parameter_magnitude)
            parameters[slot] = float(magnitude * (-1 if rng.random() < 0.5 else 1))
        return SampledMechanism(expression, MechanismBinding(parameters, lag_values),
                                int(seed), sample_index)

    def sample(self, *, seed: int, sample_index: int | None = None) -> SampledMechanism:
        """Draw one AST with replacement, without float-rounded count weights."""

        rank = random.Random(int(seed)).randrange(self.total_structures)
        return self.unrank(rank, seed=int(seed), sample_index=sample_index)
