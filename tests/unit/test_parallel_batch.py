import hashlib
import json
import random

import numpy as np
import pytest

from tsfmxai.datasets import (
    ParallelGenerationConfig,
    assemble_diverse_batch,
    derive_job_seed,
    generate_mechanism_jobs,
)
from tsfmxai.generators import RandomTreeConfig


def batch_sha256(batch) -> str:
    """Hash numerical contents and semantic identities of one forecast batch."""

    digest = hashlib.sha256()
    arrays = {
        "past_values": batch.past_values,
        "future_values": batch.future_values,
        "job_indices": batch.job_indices,
        "trajectory_indices": batch.trajectory_indices,
    }
    for name, values in arrays.items():
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))

    metadata = {
        "series_ids": batch.series_ids,
        "mechanism_ids": batch.mechanism_ids,
        "formulas": batch.formulas,
        "skeletons": batch.skeletons,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def test_job_seeds_depend_on_every_coordinate() -> None:
    seeds = {
        derive_job_seed(42, job_index, attempt_index, stream)
        for job_index in range(2)
        for attempt_index in range(2)
        for stream in range(2)
    }

    assert len(seeds) == 8


def test_parallel_jobs_are_reproducible_and_keep_one_tree_per_job() -> None:
    tree_config = RandomTreeConfig(
        minimum_operators=1,
        maximum_operators=2,
        num_series=1,
        maximum_lag=4,
        parameter_probability=0.6,
        time_probability=0.0,
        lag_probability=0.4,
        add_probability=0.8,
        require_lag=True,
    )
    generation_config = ParallelGenerationConfig(
        job_count=4,
        trajectories_per_job=3,
        context_length=24,
        prediction_length=6,
        burn_in=8,
        maximum_absolute_value=20.0,
    )

    sequential = generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=20260929,
        n_jobs=1,
    )
    threaded = generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=20260929,
        n_jobs=2,
        backend="threading",
    )

    assert [result.mechanism for result in sequential] == [
        result.mechanism for result in threaded
    ]
    for first, second in zip(sequential, threaded):
        assert first.trajectories.shape == (3, 30)
        np.testing.assert_array_equal(first.trajectories, second.trajectories)


def test_batch_guarantees_distinct_tree_skeletons() -> None:
    tree_config = RandomTreeConfig(
        minimum_operators=1,
        maximum_operators=3,
        num_series=1,
        maximum_lag=6,
        parameter_probability=0.5,
        time_probability=0.0,
        lag_probability=0.5,
        add_probability=0.75,
        require_lag=True,
    )
    generation_config = ParallelGenerationConfig(
        job_count=8,
        trajectories_per_job=3,
        context_length=32,
        prediction_length=8,
        burn_in=8,
        maximum_absolute_value=30.0,
    )
    jobs = generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=77,
        n_jobs=1,
    )

    batch = assemble_diverse_batch(
        jobs,
        generation_config,
        batch_size=12,
        minimum_distinct_mechanisms=4,
    )

    assert batch.past_values.shape == (12, 32)
    assert batch.future_values.shape == (12, 8)
    assert batch.distinct_mechanism_count >= 4
    assert len(set(batch.series_ids)) == batch.size
    assert set(batch.mechanism_ids).issubset({job.mechanism_id for job in jobs})


def test_batch_rejects_an_unreachable_diversity_requirement() -> None:
    config = ParallelGenerationConfig(
        job_count=1,
        trajectories_per_job=2,
        context_length=8,
        prediction_length=2,
    )

    with pytest.raises(ValueError, match="diversity minimum"):
        assemble_diverse_batch([], config, batch_size=1, minimum_distinct_mechanisms=2)


def test_batch_hash_is_independent_of_worker_count_and_callers_random_seeds() -> None:
    tree_config = RandomTreeConfig(
        minimum_operators=1,
        maximum_operators=3,
        num_series=1,
        maximum_lag=6,
        parameter_probability=0.5,
        time_probability=0.0,
        lag_probability=0.5,
        add_probability=0.75,
        require_lag=True,
    )
    generation_config = ParallelGenerationConfig(
        job_count=8,
        trajectories_per_job=3,
        context_length=32,
        prediction_length=8,
        burn_in=8,
        maximum_absolute_value=30.0,
    )

    random.seed(101)
    np.random.seed(202)
    one_worker_jobs = generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=77,
        n_jobs=1,
    )
    one_worker_batch = assemble_diverse_batch(
        one_worker_jobs,
        generation_config,
        batch_size=12,
        minimum_distinct_mechanisms=4,
    )

    random.seed(303)
    np.random.seed(404)
    two_worker_jobs = generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=77,
        n_jobs=2,
        backend="threading",
    )
    two_worker_batch = assemble_diverse_batch(
        two_worker_jobs,
        generation_config,
        batch_size=12,
        minimum_distinct_mechanisms=4,
    )

    assert batch_sha256(one_worker_batch) == batch_sha256(two_worker_batch)


def test_generation_does_not_advance_callers_global_random_states() -> None:
    tree_config = RandomTreeConfig(
        maximum_operators=2,
        num_series=1,
        time_probability=0.0,
        require_lag=True,
    )
    generation_config = ParallelGenerationConfig(
        job_count=2,
        trajectories_per_job=2,
        context_length=12,
        prediction_length=3,
        burn_in=4,
    )

    random.seed(505)
    np.random.seed(606)
    expected_python_draw = random.random()
    expected_numpy_draw = float(np.random.random())

    random.seed(505)
    np.random.seed(606)
    generate_mechanism_jobs(
        tree_config,
        generation_config,
        root_seed=77,
        n_jobs=1,
    )

    assert random.random() == expected_python_draw
    assert float(np.random.random()) == expected_numpy_draw
