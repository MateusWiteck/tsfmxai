"""Compare LMC training with 10, 12, and 14 DataLoader workers.

Run while the full training process is paused. The independent arm uses the
same populated trajectory cache for every worker count. The LMC arm uses a
separate cold cache per count so that each generates the same trajectories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments_local/chronos-forecasting/src"))

from tsfmxai.datasets.lmc_synth import LMCSpec, TimePFNBatches, batch_loader, verify_timepfn
from tsfmxai.pipelines.chronos2_lmc import _inputs, _sync, scratch_model


class GPUSampler:
    def __init__(self):
        self.samples: list[tuple[float, float]] = []
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-l", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            bufsize=1,
        )

        def collect():
            for line in self.process.stdout:
                try:
                    a, b = (float(part.strip()) for part in line.split(","))
                except (ValueError, TypeError):
                    continue
                self.samples.append((a, b))

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.thread.join(timeout=5)
        return {
            "samples": len(self.samples),
            "utilization_mean_percent": statistics.mean(x[0] for x in self.samples) if self.samples else None,
            "utilization_max_percent": max((x[0] for x in self.samples), default=None),
            "memory_mean_mib": statistics.mean(x[1] for x in self.samples) if self.samples else None,
            "memory_max_mib": max((x[1] for x in self.samples), default=None),
        }


def benchmark(spec, upstream, cache, workers, family, steps, start, warmup):
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    model = scratch_model(seed=spec.seed).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4 if family == "independent" else 1e-4,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    dataset = TimePFNBatches(spec, upstream, cache, family, start=start,
                             limit=start + warmup + steps)
    loader = batch_loader(dataset, workers=workers, pin_memory=True)
    begin = time.perf_counter()
    iterator = iter(loader)
    waits, trains = [], []
    sampler = GPUSampler()
    try:
        for step in range(warmup + steps):
            if step == warmup:
                _sync("cuda")
                torch.cuda.reset_peak_memory_stats()
                timed_begin = time.perf_counter()
                sampler.start()
            _sync("cuda")
            before = time.perf_counter()
            batch = next(iterator)
            ready = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(**_inputs(batch, model, "cuda", 0, spec.batch_size))
            result.loss.backward()
            optimizer.step()
            _sync("cuda")
            end = time.perf_counter()
            if step >= warmup:
                waits.append(ready - before)
                trains.append(end - ready)
        timed_wall = time.perf_counter() - timed_begin
        peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    finally:
        gpu = sampler.stop()
        del iterator, loader, model, optimizer
        torch.cuda.empty_cache()
    trajectory_dir = dataset.cache
    trajectory_count = len(list(trajectory_dir.glob("*.npy")))
    result = {
        "family": family, "workers": workers, "steps": steps, "start_update": start,
        "warmup_steps": warmup, "wall_seconds": timed_wall,
        "seconds_per_update": timed_wall / steps,
        "next_batch_seconds_mean": statistics.mean(waits),
        "next_batch_seconds_median": statistics.median(waits),
        "train_seconds_mean": statistics.mean(trains),
        "exposed_input_wait_fraction": sum(waits) / (sum(waits) + sum(trains)),
        "gpu": gpu, "pytorch_peak_allocated_mib": peak_allocated_mib,
        "trajectory_count_after": trajectory_count,
        "dataset_cache": str(trajectory_dir),
        "invocation_seconds": time.perf_counter() - begin,
    }
    print(json.dumps(result), flush=True)
    return result


def write_results(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-steps", type=int, default=500)
    parser.add_argument("--lmc-steps", type=int, default=20)
    parser.add_argument("--independent-start", type=int, default=10000)
    parser.add_argument("--remaining-independent", type=int, required=True)
    args = parser.parse_args()
    if min(args.independent_steps, args.lmc_steps, args.remaining_independent) < 1:
        parser.error("step counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    shared_cache = args.shared_cache.resolve(strict=True)
    upstream = ROOT / "experiments_local/TimePFN"
    hashes = verify_timepfn(upstream)
    spec = LMCSpec(dirichlet_min=.1, dirichlet_max=2.,
                   weibull_shape=2., weibull_scale=2.)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "workers_order": [10, 12, 14],
        "seed": spec.seed, "data": asdict(spec), "upstream_sha256": hashes,
        "gpu_name": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "shared_cache": str(shared_cache),
        "comparison": "Same Chronos-2 random initialization, 16 examples/update, 16/example microbatch, and identical window IDs. Independent uses one warm cache; each LMC arm has an isolated cold cache.",
        "timing_limit": "No validation/checkpoint time; nvidia-smi utilization is sampled every second. Exposed next_batch wait is not an exact GPU idle metric. Cold LMC sample may not represent the full stage after cache reuse.",
        "results": [],
    }
    path = args.output / "results.json"
    write_results(path, record)
    for workers in (10, 12, 14):
        arms = {}
        arms["independent_warm"] = benchmark(
            spec, upstream, shared_cache, workers, "independent",
            args.independent_steps, args.independent_start, warmup=10)
        cold_cache = args.output / f"workers-{workers}" / "cold-cache"
        arms["lmc_cold"] = benchmark(
            spec, upstream, cold_cache, workers, "lmc",
            args.lmc_steps, 0, warmup=0)
        independent = arms["independent_warm"]
        lmc = arms["lmc_cold"]
        remaining_lmc = spec.updates("lmc")
        warm_lower_bound = (args.remaining_independent + remaining_lmc) * independent["seconds_per_update"]
        naive_cold_extrapolation = (args.remaining_independent * independent["seconds_per_update"]
                                    + remaining_lmc * lmc["seconds_per_update"])
        arms["projection"] = {
            "remaining_independent_updates": args.remaining_independent,
            "remaining_lmc_updates": remaining_lmc,
            "warm_compute_floor_hours": warm_lower_bound / 3600,
            "naive_cold_lmc_scenario_hours": naive_cold_extrapolation / 3600,
            "method": "Floor assumes warm-cache LMC matches independent rate; cold scenario naively extends initial LMC batch cost and is deliberately an upper scenario, not a calibrated ETA.",
        }
        record["results"].append(arms)
        write_results(path, record)
    record["status"] = "complete"
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_results(path, record)
    print("COMPLETE", path, flush=True)


if __name__ == "__main__":
    main()
