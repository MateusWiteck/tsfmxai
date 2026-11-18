"""Prospective, forecasting-only comparison of LMC and the project generator.

The runner keeps the historical TimePFN experiment intact. Controlled mode
materializes equal-sized batch corpora before training. Time mode charges
generation, batch wait, optimization and periodic checkpoints to one clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tsfmxai.datasets.lmc_synth import LMCSpec, TimePFNBatches, batch_loader, verify_timepfn
from tsfmxai.datasets.online import CurriculumStage, OnlineBatchPlan, OnlineBatchDataset
from tsfmxai.datasets.multiseries import TrainingFormat, TrajectoryConfig
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import HistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig, NoiseConfig
from tsfmxai.pipelines.chronos2_lmc import _inputs, _sync, evaluate, scratch_model


@dataclass(frozen=True)
class PriorProtocol:
    seed: int
    batch_size: int
    context: int
    horizon: int
    channels: int
    independent_fraction: float
    independent_lr: float
    correlated_lr: float
    controlled_updates: int
    time_hours: float
    max_updates_per_stage: int
    checkpoint_interval: int
    validation_batches: int
    microbatch: int
    workers: int
    independent_bank: int
    correlated_bank: int
    dirichlet_min: float
    dirichlet_max: float
    weibull_shape: float
    weibull_scale: float
    own_min_operators: int
    own_max_operators: int
    own_max_lag: int
    own_max_degree: int
    own_generated_steps: int
    own_noise_min: float
    own_noise_max: float
    own_treatment: str
    own_dependency: str

    def __post_init__(self):
        if (self.batch_size, self.context, self.horizon, self.channels) != (16, 96, 96, 6):
            raise ValueError("Comparison contract is batch=16, context=96, horizon=96, channels=6")
        if not 0 < self.independent_fraction < 1 or not 0 < self.correlated_lr < self.independent_lr:
            raise ValueError("Invalid stage fraction or learning rates")
        if min(self.controlled_updates, self.max_updates_per_stage, self.checkpoint_interval,
               self.microbatch, self.own_generated_steps, self.validation_batches) < 1 or self.workers < 0:
            raise ValueError("Invalid size or worker count")
        if self.controlled_updates < 2 or self.time_hours <= 0:
            raise ValueError("Require at least two updates and a positive time budget")
        if self.batch_size % self.microbatch:
            raise ValueError("microbatch must divide effective batch")
        if not self.context + self.horizon <= self.own_generated_steps < self.context + 2 * self.horizon:
            raise ValueError("Own trajectory must yield exactly one window at stride=horizon")
        if self.own_dependency == "independent":
            raise ValueError("The correlated stage must allow cross-series dependencies")

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def stage_updates(self):
        first = round(self.controlled_updates * self.independent_fraction)
        return max(1, min(self.controlled_updates - 1, first)), self.controlled_updates - max(1, min(self.controlled_updates - 1, first))


def lmc_spec(protocol):
    return LMCSpec(protocol.dirichlet_min, protocol.dirichlet_max,
                   protocol.weibull_shape, protocol.weibull_scale,
                   channels=protocol.channels, context=protocol.context,
                   horizon=protocol.horizon, batch_size=protocol.batch_size,
                   independent_bank=protocol.independent_bank,
                   correlated_bank=protocol.correlated_bank, seed=protocol.seed)


def own_plan(protocol, stage, batches, *, split=0):
    independent = stage == "independent"
    config = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(
            minimum_operators=protocol.own_min_operators,
            maximum_operators=protocol.own_max_operators,
            minimum_lag=1, maximum_lag=protocol.own_max_lag,
            max_degree=protocol.own_max_degree),
        channel_count_probs={protocol.channels: 1.0},
        dependency_probs={"independent" if independent else protocol.own_dependency: 1.0},
        treatment_probs={protocol.own_treatment: 1.0},
        max_external_sources=0 if independent else 2,
        noise=NoiseConfig(std_range=(protocol.own_noise_min, protocol.own_noise_max)))
    return OnlineBatchPlan((CurriculumStage(0, config, protocol.batch_size),),
                           batches, protocol.seed + (101 if independent else 202) + 10000 * split,
                           TrajectoryConfig(protocol.own_generated_steps,
                                            HistorySpec("uniform", "uniform")),
                           TrainingFormat(protocol.context, protocol.horizon,
                                          protocol.horizon, protocol.batch_size,
                                          dtype="float32"))


class SavedBatches(Dataset):
    def __init__(self, folder, count):
        self.folder, self.count = Path(folder), count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with np.load(self.folder / f"{index:08d}.npz", allow_pickle=False) as sample:
            return {key: torch.from_numpy(sample[key].copy()) for key in ("context", "future")}


class RepeatedBatches(Dataset):
    """Repeat the finite LMC window order without regenerating its raw bank."""

    def __init__(self, source, start, stop):
        self.source, self.start, self.stop = source, start, stop
        self.spec = source.spec

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        return self.source[(self.start + index) % len(self.source)]


def _batch_tensors(batch):
    return {key: torch.as_tensor(batch[key], dtype=torch.float32) for key in ("context", "future")}


def _dataset(protocol, arm, stage, upstream, cache, count, start=0, *, split=0):
    if arm == "lmc":
        spec = lmc_spec(protocol)
        family = "independent" if stage == "independent" else "lmc"
        if count > spec.updates(family):
            raise ValueError(f"Requested {count} {family} updates; finite LMC bank has {spec.updates(family)}")
        return TimePFNBatches(spec, upstream, cache, family, split=split, limit=count, start=start)
    plan = own_plan(protocol, stage, count, split=split)
    return Subset(OnlineBatchDataset(plan), range(start, count))


def _loader(protocol, arm, dataset, device):
    if arm == "lmc":
        return batch_loader(dataset, protocol.workers, pin_memory=device.startswith("cuda"))
    # Subset preserves deterministic global batch IDs through resume.
    kwargs = dict(dataset=dataset, batch_size=None, num_workers=protocol.workers,
                  pin_memory=device.startswith("cuda"))
    if protocol.workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2,
                      multiprocessing_context="spawn")
    return DataLoader(**kwargs)


def prepare_corpus(protocol, arm, upstream, root):
    """Materialize exactly the same number of complete batches per arm."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Corpus directory must be empty: {root}")
    # np.savez is uncompressed: two float32 arrays of [16,96,6] per update.
    estimated_bytes = protocol.controlled_updates * protocol.batch_size * (
        protocol.context + protocol.horizon) * protocol.channels * 4
    if arm == "lmc":
        estimated_bytes += (protocol.independent_bank + protocol.correlated_bank *
                            protocol.channels) * 1024 * 4
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < estimated_bytes * 1.2:
        raise OSError(f"Need at least {estimated_bytes * 1.2 / 1e9:.1f} GB free for the controlled corpus")
    root.mkdir(parents=True, exist_ok=True)
    counts = protocol.stage_updates()
    started = time.perf_counter()
    manifest = {"arm": arm, "counts": counts, "seed": protocol.seed,
                "estimated_batch_bytes": estimated_bytes,
                "protocol": asdict(protocol)}
    if arm == "lmc":
        manifest["upstream_sha256"] = verify_timepfn(upstream)
    for stage, count in zip(("independent", "correlated"), counts):
        folder = root / stage
        folder.mkdir()
        dataset = _dataset(protocol, arm, stage, upstream, root / "raw_cache", count)
        for index, item in enumerate(_loader(protocol, arm, dataset, "cpu")):
            batch = _batch_tensors(item)
            temporary = folder / f"{index:08d}.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, **{key: tensor.numpy() for key, tensor in batch.items()})
            os.replace(temporary, folder / f"{index:08d}.npz")
    manifest["preparation_seconds"] = time.perf_counter() - started
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _cycle(progress, peak):
    """Two-stage OneCycle-shaped LR and beta1 as functions of stage progress."""
    p = min(1.0, max(0.0, progress))
    if p <= 0.3:
        fraction = p / 0.3
        lr = peak / 25 + (peak - peak / 25) * (1 - math.cos(math.pi * fraction)) / 2
        beta1 = 0.95 - 0.10 * (1 - math.cos(math.pi * fraction)) / 2
    else:
        fraction = (p - 0.3) / 0.7
        lr = peak / 250000 + (peak - peak / 250000) * (1 + math.cos(math.pi * fraction)) / 2
        beta1 = 0.85 + 0.10 * (1 - math.cos(math.pi * fraction)) / 2
    return lr, beta1


def _environment(device, output):
    packages = {}
    for package in ("torch", "chronos-forecasting", "numpy", "gluonts", "tsfmxai"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    gpu = torch.cuda.get_device_name(device) if device.startswith("cuda") else None
    cpu_model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in
                          cpuinfo.read_text(encoding="utf-8").splitlines()
                          if line.startswith("model name")), cpu_model)
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
        ram_bytes = psutil.virtual_memory().total
    except ImportError:
        physical_cores = None
        meminfo = Path("/proc/meminfo")
        ram_bytes = (int(next(line.split()[1] for line in
                              meminfo.read_text().splitlines() if line.startswith("MemTotal:"))) * 1024
                     if meminfo.exists() else None)
    driver = power_limit = None
    if device.startswith("cuda"):
        try:
            hardware = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version,power.limit",
                 "--format=csv,noheader", "--id=0"], text=True,
                stderr=subprocess.DEVNULL, timeout=3).strip().split(",", 1)
            driver, power_limit = (value.strip() for value in hardware)
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    source_root = Path(__file__).resolve().parents[2]
    code_hashes = {}
    for name in ("tsfmxai/pipelines/prior_baselines.py",
                 "tsfmxai/pipelines/chronos2_lmc.py",
                 "tsfmxai/datasets/lmc_synth.py", "tsfmxai/datasets/online.py",
                 "tsfmxai/datasets/multiseries.py",
                 "tsfmxai/generators/multiseries.py"):
        code_hashes[name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
    return {"repository_commit": commit, "packages": packages,
            "torch_cuda": torch.version.cuda, "gpu": gpu,
            "hostname": platform.node(), "cpu_model": cpu_model,
            "logical_cores": os.cpu_count(), "physical_cores": physical_cores,
            "ram_total_bytes": ram_bytes,
            "disk_total_bytes": shutil.disk_usage(output).total,
            "disk_device": os.stat(output).st_dev,
            "nvidia_driver": driver, "gpu_power_limit": power_limit,
            "code_sha256": code_hashes}


RESUME_IDENTITY_FIELDS = (
    "repository_commit", "code_sha256", "hostname", "cpu_model", "gpu",
    "packages", "torch_cuda",
)


def _compare_resume_environment(saved, current):
    """Reject identity changes; retain transient hardware differences as diagnostics."""
    if not isinstance(saved, dict) or any(saved.get(key) != current.get(key)
                                          for key in RESUME_IDENTITY_FIELDS):
        raise ValueError("Resume environment identity does not match")
    return {key: {"checkpoint": saved.get(key), "resume": current.get(key)}
            for key in saved.keys() | current.keys()
            if key not in RESUME_IDENTITY_FIELDS and saved.get(key) != current.get(key)}


def _gpu_sample(device):
    if not device.startswith("cuda"):
        return {}
    sample = {"cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
              "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device)}
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "--id=0"], text=True,
            stderr=subprocess.DEVNULL, timeout=3)
        utilization, memory = output.strip().split(",", 1)
        sample.update(gpu_utilization_percent=float(utilization),
                      gpu_memory_used_mib=float(memory))
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return sample


def _reconcile_log(path, completed):
    """Keep only telemetry backed by the restored model checkpoint."""
    if not path.exists():
        return 0.0, 0, 0.0
    retained, discarded = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        (retained if row["update"] <= completed else discarded).append(row)
    if discarded:
        temporary = path.with_suffix(".reconciled")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8")
        os.replace(temporary, path)
    latest = max((row["elapsed_seconds"] for row in discarded), default=0.0)
    wasted_wait = sum(row.get("batch_wait_seconds", 0.0) for row in discarded)
    return latest, len(discarded), wasted_wait


def run(protocol, arm, study, upstream, output, *, corpus=None, resume=False,
        device="cuda", tiny=False):
    """Run one arm. A checkpoint contains the complete optimizer/clock state."""
    if arm not in ("lmc", "own") or study not in ("controlled", "time"):
        raise ValueError("arm must be lmc/own; study must be controlled/time")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if study == "controlled" and corpus is None:
        raise ValueError("Controlled study requires a prepared corpus")
    if arm == "lmc":
        provenance = verify_timepfn(upstream)
    else:
        provenance = None
    output = Path(output)
    checkpoint = output / "latest.pt"
    if resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not resume and checkpoint.exists():
        raise FileExistsError(checkpoint)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"arm": arm, "study": study, "protocol": asdict(protocol),
                 "upstream": provenance, "tiny": tiny}
    environment = _environment(device, output)
    disk_free_at_start = shutil.disk_usage(output).free
    if study == "controlled":
        manifest = json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))
        if manifest["arm"] != arm or manifest["protocol"] != asdict(protocol):
            raise ValueError("Corpus provenance does not match the run")
        signature["corpus_manifest_sha256"] = hashlib.sha256(
            (Path(corpus) / "manifest.json").read_bytes()).hexdigest()
    elif (output / "cache").exists() and not resume:
        raise FileExistsError("Time-budget run requires a cold, empty cache")
    random.seed(protocol.seed)
    np.random.seed(protocol.seed)
    torch.manual_seed(protocol.seed)
    model = scratch_model(seed=protocol.seed, tiny=tiny).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.independent_lr,
                                 betas=(.95, .999), eps=1e-8)
    completed = 0
    active_before = 0.0
    stage = 0
    stage_update = 0
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume signature does not match")
        environment_differences = _compare_resume_environment(
            saved.get("environment"), environment)
        if environment_differences:
            with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"environment_differences": environment_differences,
                                         "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed, active_before = saved["updates"], saved["active_seconds"]
        stage, stage_update = saved["stage"], saved["stage_update"]
        clock_path = output / "latest_clock.json"
        if clock_path.exists():
            clock = json.loads(clock_path.read_text(encoding="utf-8"))
            if (clock["updates"], clock["stage"], clock["stage_update"]) == (
                    completed, stage, stage_update):
                active_before = max(active_before, clock["active_seconds"])
        # Updates after the atomic checkpoint consumed active time but their
        # model state is unavailable. Charge that measured work and replay it.
        latest, discarded, wasted_wait = _reconcile_log(output / "telemetry.jsonl", completed)
        _reconcile_log(output / "validation.jsonl", completed)
        active_before = max(active_before, latest)
        if discarded:
            with (output / "resume_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"replayed_updates": discarded,
                                         "replayed_batch_wait_seconds": wasted_wait,
                                         "charged_elapsed_seconds": active_before,
                                         "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    began = time.perf_counter()
    elapsed = lambda: active_before + time.perf_counter() - began
    if study == "controlled":
        counts = protocol.stage_updates()
    else:
        counts = (protocol.max_updates_per_stage,) * 2
    limit = protocol.time_hours * 3600
    boundary = limit * protocol.independent_fraction
    wait_seconds = 0.0
    stage_names = ("independent", "correlated")
    interrupted = False

    def validate(name):
        folder = output / "validation_corpus" / name
        if not (folder / "complete").exists():
            folder.mkdir(parents=True, exist_ok=True)
            source = _dataset(protocol, arm, name, upstream, output / "validation_cache",
                              protocol.validation_batches, split=1)
            for index in range(protocol.validation_batches):
                item = _batch_tensors(source[index])
                temporary = folder / f"{index:08d}.tmp"
                with temporary.open("wb") as stream:
                    np.savez(stream, **{key: tensor.numpy() for key, tensor in item.items()})
                os.replace(temporary, folder / f"{index:08d}.npz")
            (folder / "complete").write_text("fixed validation batches\n", encoding="utf-8")
        loader = DataLoader(SavedBatches(folder, protocol.validation_batches), batch_size=None,
                            num_workers=0, pin_memory=device.startswith("cuda"))
        metrics = evaluate(model, loader, device, protocol.microbatch)
        with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "update": completed,
                                     "elapsed_seconds": elapsed(), **metrics}) + "\n")

    def save():
        _sync(device)
        payload = {"signature": signature, "model": model.state_dict(),
                   "environment": environment,
                   "optimizer": optimizer.state_dict(), "updates": completed,
                   "stage": stage, "stage_update": stage_update,
                   "active_seconds": elapsed(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
        temporary = output / "latest.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        clock_temporary = output / "latest_clock.tmp"
        clock_temporary.write_text(json.dumps({"updates": completed, "stage": stage,
                                               "stage_update": stage_update,
                                               "active_seconds": elapsed()}), encoding="utf-8")
        os.replace(clock_temporary, output / "latest_clock.json")

    model.train()
    try:
        for stage in range(stage, 2):
            name = stage_names[stage]
            stage_limit = boundary if stage == 0 else limit
            stage_begin = 0 if stage == 0 else boundary
            count = counts[stage]
            if study == "controlled":
                dataset = Subset(SavedBatches(Path(corpus) / name, count), range(stage_update, count))
                loader = DataLoader(dataset, batch_size=None, num_workers=protocol.workers,
                                    pin_memory=device.startswith("cuda"))
            else:
                if arm == "lmc":
                    bank_updates = lmc_spec(protocol).updates("independent" if stage == 0 else "lmc")
                    source = _dataset(protocol, arm, name, upstream, output / "cache", bank_updates)
                    dataset = RepeatedBatches(source, stage_update, count)
                else:
                    dataset = _dataset(protocol, arm, name, upstream, output / "cache", count,
                                       start=stage_update)
                loader = _loader(protocol, arm, dataset, device)
            iterator = iter(loader)
            for index in range(stage_update, count):
                if study == "time" and elapsed() >= stage_limit:
                    break
                _sync(device)
                waiting = time.perf_counter()
                batch = _batch_tensors(next(iterator))
                ready = time.perf_counter()
                wait_seconds += ready - waiting
                if study == "time" and elapsed() >= stage_limit:
                    break
                if any(tuple(batch[key].shape) != (16, 96, 6) for key in ("context", "future")):
                    raise ValueError("Expected full [16,96,6] context/future batches")
                progress = ((index / count) if study == "controlled" else
                            (elapsed() - stage_begin) / (stage_limit - stage_begin))
                peak = protocol.independent_lr if stage == 0 else protocol.correlated_lr
                lr, beta1 = _cycle(progress, peak)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["betas"] = (beta1, .999)
                optimizer.zero_grad(set_to_none=True)
                loss = 0.0
                for offset in range(0, protocol.batch_size, protocol.microbatch):
                    inputs = _inputs(batch, model, device, offset, offset + protocol.microbatch)
                    result = model(**inputs)
                    if not torch.isfinite(result.loss):
                        raise FloatingPointError("Nonfinite forecasting loss")
                    weight = protocol.microbatch / protocol.batch_size
                    (result.loss * weight).backward()
                    loss += result.loss.detach().item() * weight
                optimizer.step()
                _sync(device)
                completed += 1
                stage_update = index + 1
                row = {"stage": name, "update": completed, "stage_update": stage_update,
                       "elapsed_seconds": elapsed(), "batch_wait_seconds": ready-waiting,
                       "lr": lr, "loss": loss, "windows_seen": completed * protocol.batch_size}
                if study == "time" and arm == "lmc":
                    row["window_reuses_stage"] = max(0, stage_update - bank_updates) * protocol.batch_size
                row["estimated_remaining_hours"] = (
                    max(0., limit - row["elapsed_seconds"]) / 3600 if study == "time"
                    else max(0., protocol.controlled_updates - completed) *
                    row["elapsed_seconds"] / completed / 3600)
                with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if completed == 1 or completed % 100 == 0:
                    row.update(_gpu_sample(device))
                    print(json.dumps({"progress": row}), flush=True)
                if completed % protocol.checkpoint_interval == 0:
                    validate(name)
                    save()
            del iterator, loader
            if study == "time" and elapsed() < stage_limit and stage_update >= count:
                raise RuntimeError("Safety update limit reached before time boundary; restart both arms with a larger frozen limit")
            if stage == 0:
                # Retain the measured boundary, including its weight state.
                validate(name)
                model.save_pretrained(output / "independent_end")
                stage = 1
                stage_update = 0
                save()
        reason = "budget_elapsed" if study == "time" and elapsed() >= limit else "data_exhausted"
    except KeyboardInterrupt:
        interrupted = True
        reason = "interrupted"
    save()
    if not interrupted:
        model.save_pretrained(output / "model_final")
    stage_counts = {name: 0 for name in stage_names}
    if completed:
        wait_seconds = 0.0
        with (output / "telemetry.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                wait_seconds += row["batch_wait_seconds"]
                stage_counts[row["stage"]] += 1
    cache_counts = {}
    if arm == "lmc":
        root = output / "cache" if study == "time" else Path(corpus) / "raw_cache"
        for name in stage_names:
            family = "independent" if name == "independent" else "lmc"
            cache_counts[name] = sum(1 for path in root.rglob("*.npy") if path.parent.name == family)
    summary = {"arm": arm, "study": study, "updates": completed,
               "elapsed_seconds": elapsed(), "batch_wait_seconds": wait_seconds,
               "stage_updates": stage_counts,
               "batch_wait_fraction": wait_seconds / elapsed() if elapsed() else None,
               "updates_per_active_hour": completed * 3600 / elapsed() if elapsed() else None,
               "stop_reason": reason, "signature": signature,
               "environment": environment,
               "disk_free_bytes_at_start": disk_free_at_start,
               "windows_seen": completed * protocol.batch_size,
               "raw_cached_trajectories": cache_counts if arm == "lmc" else None,
               "repeated_lmc_windows": (sum(max(0, stage_counts[name] - lmc_spec(protocol).updates(
                   "independent" if name == "independent" else "lmc")) * protocol.batch_size
                   for name in stage_names) if arm == "lmc" and study == "time" else None),
               "own_accepted_trajectories": completed * protocol.batch_size if arm == "own" else None,
               "gpu": _gpu_sample(device),
               "parameters": sum(p.numel() for p in model.parameters())}
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
