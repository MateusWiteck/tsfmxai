"""Measure AST batch delivery at the frozen 16 x 96/96 x 6 training shape."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.datasets.multiseries import TrainingFormat, TrajectoryConfig
from tsfmxai.datasets.online import CurriculumStage, OnlineBatchPlan, make_dataloader
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import HistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--measured", type=int, default=64)
    args = parser.parse_args()
    if min(args.warmup, args.measured) < 1:
        raise ValueError("Warmup and measured batch counts must be positive")
    config = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=16),
        channel_count_probs={6: 1.}, dependency_probs={"random": 1.},
        max_external_sources=2,
    )
    trajectory = TrajectoryConfig(192, HistorySpec("uniform", "uniform"))
    training = TrainingFormat()
    assert (training.context_length, training.prediction_length,
            training.stride, training.batch_size) == (96, 96, 96, 16)
    rows = []
    for topology in ("independent", "random", "linked_chain"):
        selected = replace(config, dependency_probs={topology: 1.})
        plan = OnlineBatchPlan(
            (CurriculumStage(0, selected, systems_per_batch=16,
                             trajectories_per_system=1),),
            args.warmup + args.measured, 20261119, trajectory, training,
        )
        loader = make_dataloader(plan, num_workers=args.workers,
                                 prefetch_factor=2, pin_memory=False)
        started = time.perf_counter()
        iterator = iter(loader)
        waits = []
        steady_start = None
        for index in range(args.warmup + args.measured):
            if index == args.warmup:
                steady_start = time.perf_counter()
            before = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - before
            assert tuple(batch["context"].shape) == (16, 96, 6)
            assert tuple(batch["future"].shape) == (16, 96, 6)
            if index >= args.warmup:
                waits.append(elapsed)
        steady_elapsed = time.perf_counter() - steady_start
        total_elapsed = time.perf_counter() - started
        del iterator, loader
        row = dict(topology=topology, measured_batches=len(waits),
                   sustained_seconds_per_batch=steady_elapsed / len(waits),
                   batches_per_second=len(waits) / steady_elapsed,
                   median_next_batch_seconds=statistics.median(waits),
                   p95_next_batch_seconds=float(np.percentile(waits, 95)),
                   elapsed_including_startup_seconds=total_elapsed)
        rows.append(row)
        print(json.dumps(row), flush=True)
    files = ["scripts/benchmark_ast_96_96.py",
             "src/tsfmxai/datasets/multiseries.py",
             "src/tsfmxai/datasets/online.py",
             "src/tsfmxai/generators/multiseries.py"]
    record = dict(
        question="How quickly does the AST generator deliver training-shape batches online?",
        hypothesis="Throughput differs by dependency topology at fixed batch dimensions.",
        configuration=dict(seed=20261119, workers=args.workers, prefetch_factor=2,
                           pin_memory=False, blas_threads=1, warmup_batches=args.warmup,
                           measured_batches=args.measured, systems_per_batch=16,
                           trajectories_per_system=1, generator=asdict(config),
                           trajectory=asdict(trajectory), training=asdict(training),
                           topologies=["independent", "random", "linked_chain"]),
        provenance=dict(git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                           for path in files},
            python=platform.python_version(), numpy=np.__version__,
            platform=platform.platform(), cpu=platform.processor(),
            logical_cpus=os.cpu_count()),
        results=rows,
        scientific_summary=("CPU-only batch delivery with one run per topology. "
                            "The 64/16 historical experiment uses a different workload; "
                            "this run fixes the prospective 16-example, 96/96, six-channel "
                            "contract. Delivery latency includes prefetch and serialization "
                            "but excludes Chronos optimization and GPU stall."),
        reproduction=("From repository root with Python 3.11 and project dependencies: "
                      "python scripts/benchmark_ast_96_96.py --output "
                      "experiments/multivariate_generation/07_ast_batch_96_96/record.json "
                      "--workers 8 --warmup 16 --measured 64"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
