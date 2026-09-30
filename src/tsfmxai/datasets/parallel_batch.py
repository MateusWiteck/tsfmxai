"""Parallel generation of diverse forecast batches from symbolic mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from tsfmxai.generators import (
    DivergentTrajectoryError,
    RandomTreeConfig,
    SampledMechanism,
    required_history_length,
    sample_random_mechanism,
    sample_univariate_trajectories,
)


@dataclass(frozen=True)
class ParallelGenerationConfig:
    """Configuration shared by mechanism jobs and the diverse batch builder."""

    job_count: int
    trajectories_per_job: int
    context_length: int
    prediction_length: int
    burn_in: int = 32
    innovation_std: float = 0.05
    initial_value_minimum: float = -1.0
    initial_value_maximum: float = 1.0
    maximum_absolute_value: float = 100.0
    minimum_window_std: float = 1e-4
    maximum_attempts_per_job: int = 100

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "job_count": self.job_count,
            "trajectories_per_job": self.trajectories_per_job,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "maximum_attempts_per_job": self.maximum_attempts_per_job,
        }
        for name, value in positive_integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.innovation_std < 0:
            raise ValueError("innovation_std must be non-negative.")
        if self.initial_value_maximum < self.initial_value_minimum:
            raise ValueError("The initial-value interval is invalid.")
        if self.maximum_absolute_value <= 0:
            raise ValueError("maximum_absolute_value must be positive.")
        if self.minimum_window_std < 0:
            raise ValueError("minimum_window_std must be non-negative.")


@dataclass(frozen=True)
class MechanismJobResult:
    """One accepted symbolic mechanism and its independently sampled trajectories."""

    job_index: int
    attempt_index: int
    mechanism: SampledMechanism
    trajectories: np.ndarray
    mechanism_seed: int
    trajectory_seed: int

    @property
    def mechanism_id(self) -> str:
        return f"mechanism_{self.job_index:04d}"


@dataclass(frozen=True)
class SyntheticForecastBatch:
    """Forecast windows plus the mechanism and trajectory identity of every row."""

    past_values: np.ndarray
    future_values: np.ndarray
    series_ids: tuple[str, ...]
    mechanism_ids: tuple[str, ...]
    job_indices: np.ndarray
    trajectory_indices: np.ndarray
    formulas: tuple[str, ...]
    skeletons: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.past_values.shape[0])

    @property
    def distinct_mechanism_count(self) -> int:
        return len(set(self.skeletons))


def derive_job_seed(root_seed: int, job_index: int, attempt_index: int, stream: int) -> int:
    """Derive a seed independent of job scheduling and completion order."""

    if job_index < 0 or attempt_index < 0 or stream < 0:
        raise ValueError("Seed coordinates must be non-negative.")
    sequence = np.random.SeedSequence(
        [int(root_seed), int(job_index), int(attempt_index), int(stream)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def generate_mechanism_job(
    tree_config: RandomTreeConfig,
    generation_config: ParallelGenerationConfig,
    *,
    root_seed: int,
    job_index: int,
) -> MechanismJobResult:
    """Sample one stable tree and generate several trajectories from that same tree."""

    retained_length = generation_config.context_length + generation_config.prediction_length
    generated_steps = generation_config.burn_in + retained_length

    for attempt_index in range(generation_config.maximum_attempts_per_job):
        mechanism_seed = derive_job_seed(root_seed, job_index, attempt_index, stream=0)
        trajectory_seed = derive_job_seed(root_seed, job_index, attempt_index, stream=1)
        mechanism = sample_random_mechanism(
            tree_config,
            seed=mechanism_seed,
            sample_index=job_index,
        )
        history_length = required_history_length(mechanism.expression, mechanism.binding)
        if history_length > generation_config.context_length:
            continue

        try:
            complete = sample_univariate_trajectories(
                mechanism.expression,
                mechanism.binding,
                trajectory_count=generation_config.trajectories_per_job,
                generated_steps=generated_steps,
                initial_value_minimum=generation_config.initial_value_minimum,
                initial_value_maximum=generation_config.initial_value_maximum,
                innovation_std=generation_config.innovation_std,
                root_seed=trajectory_seed,
                maximum_absolute_value=generation_config.maximum_absolute_value,
            )
        except DivergentTrajectoryError:
            continue

        trajectories = complete[:, -retained_length:]
        if not np.isfinite(trajectories).all():
            continue
        if np.any(np.std(trajectories, axis=1) < generation_config.minimum_window_std):
            continue

        return MechanismJobResult(
            job_index=job_index,
            attempt_index=attempt_index,
            mechanism=mechanism,
            trajectories=trajectories,
            mechanism_seed=mechanism_seed,
            trajectory_seed=trajectory_seed,
        )

    raise RuntimeError(
        f"Job {job_index} did not produce an accepted mechanism after "
        f"{generation_config.maximum_attempts_per_job} attempts."
    )


def generate_mechanism_jobs(
    tree_config: RandomTreeConfig,
    generation_config: ParallelGenerationConfig,
    *,
    root_seed: int,
    n_jobs: int = 1,
    backend: Literal["loky", "threading"] = "loky",
) -> list[MechanismJobResult]:
    """Run index-stable jobs sequentially or with joblib workers."""

    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero.")
    if backend not in {"loky", "threading"}:
        raise ValueError("backend must be 'loky' or 'threading'.")

    arguments = range(generation_config.job_count)
    if n_jobs == 1:
        return [
            generate_mechanism_job(
                tree_config,
                generation_config,
                root_seed=root_seed,
                job_index=job_index,
            )
            for job_index in arguments
        ]

    results = Parallel(n_jobs=n_jobs, backend=backend, max_nbytes=None)(
        delayed(generate_mechanism_job)(
            tree_config,
            generation_config,
            root_seed=root_seed,
            job_index=job_index,
        )
        for job_index in arguments
    )
    return sorted(results, key=lambda result: result.job_index)


def assemble_diverse_batch(
    job_results: list[MechanismJobResult],
    generation_config: ParallelGenerationConfig,
    *,
    batch_size: int,
    minimum_distinct_mechanisms: int,
) -> SyntheticForecastBatch:
    """Assemble windows while guaranteeing a minimum number of unique AST skeletons."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if minimum_distinct_mechanisms < 1:
        raise ValueError("minimum_distinct_mechanisms must be positive.")
    if minimum_distinct_mechanisms > batch_size:
        raise ValueError("The diversity minimum cannot exceed batch_size.")

    unique_jobs: list[MechanismJobResult] = []
    seen_skeletons: set[str] = set()
    for result in sorted(job_results, key=lambda item: item.job_index):
        skeleton = result.mechanism.skeleton_formula
        if skeleton not in seen_skeletons:
            seen_skeletons.add(skeleton)
            unique_jobs.append(result)

    if len(unique_jobs) < minimum_distinct_mechanisms:
        raise ValueError(
            f"Only {len(unique_jobs)} distinct AST skeletons are available; "
            f"the batch requires {minimum_distinct_mechanisms}."
        )
    capacity = len(unique_jobs) * generation_config.trajectories_per_job
    if batch_size > capacity:
        raise ValueError(
            f"The unique mechanisms provide {capacity} trajectories, fewer than "
            f"batch_size={batch_size}."
        )

    selected: list[tuple[MechanismJobResult, int]] = []
    for trajectory_index in range(generation_config.trajectories_per_job):
        for result in unique_jobs:
            selected.append((result, trajectory_index))
            if len(selected) == batch_size:
                break
        if len(selected) == batch_size:
            break

    split = generation_config.context_length
    windows = np.stack([result.trajectories[index] for result, index in selected])
    mechanism_ids = tuple(result.mechanism_id for result, _ in selected)
    trajectory_indices = np.asarray([index for _, index in selected], dtype=int)
    series_ids = tuple(
        f"{mechanism_id}_trajectory_{trajectory_index:03d}"
        for mechanism_id, trajectory_index in zip(mechanism_ids, trajectory_indices)
    )
    batch = SyntheticForecastBatch(
        past_values=windows[:, :split],
        future_values=windows[:, split:],
        series_ids=series_ids,
        mechanism_ids=mechanism_ids,
        job_indices=np.asarray([result.job_index for result, _ in selected], dtype=int),
        trajectory_indices=trajectory_indices,
        formulas=tuple(result.mechanism.formula for result, _ in selected),
        skeletons=tuple(result.mechanism.skeleton_formula for result, _ in selected),
    )
    if batch.distinct_mechanism_count < minimum_distinct_mechanisms:
        raise AssertionError("The batch diversity invariant was not preserved.")
    return batch
