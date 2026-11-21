"""Measure delivered AST/KernelSynth batch latency with a fixed worker count."""

import argparse
from dataclasses import asdict
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfmxai.datasets.multiseries import TrainingFormat, TrajectoryConfig
from tsfmxai.datasets.online import CurriculumStage, OnlineBatchPlan, make_dataloader
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.hybrid import HybridConfig
from tsfmxai.generators.initialization import HistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--measured", type=int, default=64)
    args = parser.parse_args()
    # The executed record used the removed independent treatment draw; rerun the
    # record's git_head to reproduce it. This script now uses the dependency rule.
    base = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=16),
        channel_count_probs={6: 1.}, dependency_probs={"random": 1.},
        max_external_sources=2,
    )
    trajectory = TrajectoryConfig(192, HistorySpec("uniform", "uniform"))
    training = TrainingFormat(96, 96, 96, batch_size=16)
    rows = []
    for p in (0., .25, .5, .75):
        config = HybridConfig(base, {"ast": 1. - p, "kernel_synth": p})
        plan = OnlineBatchPlan((CurriculumStage(0, config, 16),),
                               args.warmup + args.measured, 20261119,
                               trajectory, training)
        loader = make_dataloader(plan, num_workers=args.workers, prefetch_factor=2,
                                 pin_memory=False)
        iterator = iter(loader)
        times, actual_kernel, actual_total = [], 0, 0
        started = time.perf_counter()
        measured_started = None
        for index in range(args.warmup + args.measured):
            if index == args.warmup:
                measured_started = time.perf_counter()
            before = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - before
            if index >= args.warmup:
                times.append(elapsed)
                for item in batch["metadata"]:
                    families = item["channel_families"]
                    actual_kernel += sum(f == "kernel_synth" for f in families)
                    actual_total += len(families)
        measured_elapsed = time.perf_counter() - measured_started
        total_elapsed = time.perf_counter() - started
        del iterator, loader
        row = dict(kernel_probability=p, actual_kernel_fraction=actual_kernel / actual_total,
                   batch_seconds=times, median_seconds=statistics.median(times),
                   mean_seconds=statistics.mean(times), p95_seconds=float(np.percentile(times, 95)),
                   sustained_seconds_per_batch=measured_elapsed / args.measured,
                   elapsed_including_startup_seconds=total_elapsed,
                   measured_batches=len(times))
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "batch_seconds"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        config=dict(workers=args.workers, warmup=args.warmup, measured=args.measured,
                    seed=20261119, batch_size=16, channels=6, context=96, horizon=96,
                    generated_steps=192, self_only_treatment_probs=dict(base.self_only_treatment_probs),
                    dependency="random", kernel_composition="1 to 5 official-bank kernels",
                    factor_cache_size=64, blas_threads=1,
                    base_multiseries=asdict(base), trajectory=asdict(trajectory),
                    training=asdict(training), prefetch_factor=2, pin_memory=False),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         cpu=platform.processor(), logical_cpus=os.cpu_count()),
        results=rows), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
