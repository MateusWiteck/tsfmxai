"""Deterministic batch-index planning for online synthetic training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from tsfmxai.datasets.multiseries import (
    TrainingBatch,
    TrainingBatchStream,
    TrainingFormat,
    TrajectoryConfig,
    generate_training_batches,
)
from tsfmxai.generators.multiseries import MultiseriesConfig, stream_seed


@dataclass(frozen=True)
class CurriculumStage:
    """Generator configuration active over a contiguous range of batch IDs."""

    first_batch: int
    config: MultiseriesConfig
    systems_per_batch: int
    trajectories_per_system: int = 1

    def __post_init__(self) -> None:
        if type(self.first_batch) is not int or self.first_batch < 0:
            raise ValueError("first_batch must be a non-negative integer")
        if type(self.systems_per_batch) is not int or self.systems_per_batch < 1:
            raise ValueError("systems_per_batch must be positive")
        if type(self.trajectories_per_system) is not int or self.trajectories_per_system < 1:
            raise ValueError("trajectories_per_system must be positive")


@dataclass(frozen=True)
class OnlineBatchPlan:
    """Immutable description of a finite, reproducible online batch stream."""

    stages: tuple[CurriculumStage, ...]
    total_batches: int
    root_seed: int
    trajectory: TrajectoryConfig
    training: TrainingFormat

    def __post_init__(self) -> None:
        if not self.stages or self.stages[0].first_batch != 0:
            raise ValueError("Stages must start at batch zero")
        if tuple(sorted(self.stages, key=lambda stage: stage.first_batch)) != self.stages:
            raise ValueError("Stages must be ordered by first_batch")
        if len({stage.first_batch for stage in self.stages}) != len(self.stages):
            raise ValueError("Stage boundaries must be unique")
        if type(self.total_batches) is not int or self.total_batches < 1:
            raise ValueError("total_batches must be positive")
        if type(self.root_seed) is not int or self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")

    def stage_for(self, batch_id: int) -> CurriculumStage:
        if type(batch_id) is not int or not 0 <= batch_id < self.total_batches:
            raise IndexError(batch_id)
        selected = self.stages[0]
        for stage in self.stages[1:]:
            if stage.first_batch > batch_id:
                break
            selected = stage
        return selected

    def generate(self, batch_id: int) -> TrainingBatch:
        """Generate exactly one batch; output depends on its global ID, not worker."""
        stage = self.stage_for(batch_id)
        seed = stream_seed(self.root_seed, 701, batch_id)
        stream = generate_training_batches(
            config=stage.config,
            num_systems=stage.systems_per_batch,
            trajectories_per_system=stage.trajectories_per_system,
            trajectory=self.trajectory,
            training=self.training,
            seed=seed,
        )
        try:
            batch = next(stream)
        except StopIteration as error:
            raise RuntimeError(
                f"Batch {batch_id} produced no valid windows; inspect trajectory configuration"
            ) from error
        expected_examples = self.training.batch_size
        if batch.context.shape[0] != expected_examples:
            raise RuntimeError(
                f"Batch {batch_id} has {batch.context.shape[0]} examples; "
                f"expected exactly {expected_examples}. Increase systems_per_batch "
                "or choose trajectory length/stride that guarantees enough windows."
            )
        return batch


class OnlineBatchDataset(Dataset):
    """Map-style PyTorch dataset yielding pre-collated batches by global ID."""

    def __init__(self, plan: OnlineBatchPlan):
        self.plan = plan

    def __len__(self) -> int:
        return self.plan.total_batches

    def __getitem__(self, batch_id: int) -> dict[str, object]:
        batch = self.plan.generate(int(batch_id))
        return {
            "context": batch.context,
            "future": batch.future,
            "channel_mask": batch.channel_mask,
            "equation_targets": batch.equation_targets,
            "equation_mask": batch.equation_mask,
            "noise_targets": batch.noise_targets,
            "metadata": batch.metadata,
        }


def worker_seed_init(worker_id: int) -> None:
    """Seed incidental worker RNG use; scientific randomness is keyed by batch ID."""
    seed = int(np.random.SeedSequence([torch_worker_seed(), worker_id]).generate_state(1)[0])
    np.random.seed(seed)


def torch_worker_seed() -> int:
    """Return PyTorch's per-worker seed without importing torch at module import time."""
    import torch

    return int(torch.initial_seed())


def make_dataloader(plan: OnlineBatchPlan, *, num_workers: int = 0,
                    prefetch_factor: int = 2, pin_memory: bool = True):
    """Create a DataLoader that keeps workers alive and bounds queued batches."""
    import torch
    from torch.utils.data import DataLoader

    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if num_workers and (type(prefetch_factor) is not int or prefetch_factor < 1):
        raise ValueError("prefetch_factor must be positive when workers are enabled")
    kwargs = dict(
        dataset=OnlineBatchDataset(plan), batch_size=None, num_workers=num_workers,
        pin_memory=pin_memory, worker_init_fn=worker_seed_init,
        generator=torch.Generator().manual_seed(plan.root_seed),
    )
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(**kwargs)
