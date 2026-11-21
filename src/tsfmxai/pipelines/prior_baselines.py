"""Prospective, forecasting-only comparison of LMC and the project generator.

The runner keeps the historical TimePFN experiment intact. Controlled mode
materializes equal-sized batch corpora before training. Time mode charges
generation, batch wait, optimization and periodic checkpoints to one clock.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from tsfmxai.generators.initialization import RandomHistorySpec
from tsfmxai.generators.hybrid import HybridConfig
from tsfmxai.generators.multiseries import MultiseriesConfig, NoiseConfig
from tsfmxai.pipelines.chronos2_lmc import _inputs, _sync, evaluate, scratch_model
from tsfmxai.pipelines.equation_targets import EquationTargetTeacher
from tsfmxai.models.chronos2_equation import Chronos2EquationModel, equation_training_loss


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
    own_self_only_treatment_probs: dict[str, float]
    own_dependency: str
    own_kernel_probability: float = 0.0
    # Defaults reproduce the historical LMC-era protocol; current configs set 0.05-1.0.
    own_min_parameter_magnitude: float = 0.05
    own_max_parameter_magnitude: float = 0.90
    own_difficulty_policy: str = "none"
    # Backbone: defaults reproduce the completed LMC runs (16-step patches, no
    # channel index); current configs use one-step patches and the index.
    backbone_patch_size: int = 16
    backbone_channel_embedding: bool = False

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
            raise ValueError("The proposed generator must permit cross-series dependencies from the start")
        if not 0 <= self.own_kernel_probability < 1:
            raise ValueError("own_kernel_probability must be in [0, 1)")
        if self.own_difficulty_policy not in ("none", "mixed", "curriculum"):
            raise ValueError("Unknown own_difficulty_policy")
        if type(self.backbone_patch_size) is not int or self.backbone_patch_size < 1:
            raise ValueError("backbone_patch_size must be a positive integer")
        if type(self.backbone_channel_embedding) is not bool:
            raise ValueError("backbone_channel_embedding must be boolean")

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        # Historical LMC configs carry treatment fields for an own arm that was
        # never run with them; the treatment rule is now fixed by dependency.
        for legacy in ("own_treatment", "own_treatment_probs", "own_treatment_policy"):
            values.pop(legacy, None)
        values.setdefault("own_self_only_treatment_probs", {"jacobian": .5, "heaps": .5})
        return cls(**values)

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
    config = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(
            minimum_operators=protocol.own_min_operators,
            maximum_operators=protocol.own_max_operators,
            minimum_lag=1, maximum_lag=protocol.own_max_lag,
            max_degree=protocol.own_max_degree,
            minimum_parameter_magnitude=protocol.own_min_parameter_magnitude,
            maximum_parameter_magnitude=protocol.own_max_parameter_magnitude),
        channel_count_probs={protocol.channels: 1.0},
        # The AST generator allows cross-series dependencies from the first
        # update. The two runner phases retain the LMC budget/LR boundaries.
        dependency_probs={protocol.own_dependency: 1.0},
        self_only_treatment_probs=protocol.own_self_only_treatment_probs,
        max_external_sources=2,
        noise=NoiseConfig(std_range=(protocol.own_noise_min, protocol.own_noise_max)))
    generator_config = (HybridConfig(config, {"ast": 1 - protocol.own_kernel_probability,
                                               "kernel_synth": protocol.own_kernel_probability})
                        if protocol.own_kernel_probability else config)
    difficulty_configs = ()
    if protocol.own_difficulty_policy != "none":
        def operation_config(maximum_operators):
            ast_tree = replace(config.ast_config,
                               maximum_operators=min(protocol.own_max_operators,
                                                     max(protocol.own_min_operators,
                                                         maximum_operators)))
            ast = replace(config, ast_config=ast_tree)
            return (HybridConfig(ast, generator_config.family_probs)
                    if isinstance(generator_config, HybridConfig) else ast)
        difficulty_configs = tuple(operation_config(n) for n in (2, 4, 6))
    return OnlineBatchPlan((CurriculumStage(0, generator_config, protocol.batch_size),),
                           batches, protocol.seed + (101 if stage == "independent" else 202) + 10000 * split,
                           TrajectoryConfig(protocol.own_generated_steps,
                                            RandomHistorySpec()),
                           TrainingFormat(protocol.context, protocol.horizon,
                                          protocol.horizon, protocol.batch_size,
                                          dtype="float32"),
                           difficulty_policy=protocol.own_difficulty_policy,
                           difficulty_configs=difficulty_configs)


class SavedBatches(Dataset):
    def __init__(self, folder, count):
        self.folder, self.count = Path(folder), count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with np.load(self.folder / f"{index:08d}.npz", allow_pickle=False) as sample:
            return {key: torch.from_numpy(sample[key].copy()) for key in sample.files}


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
    result = {key: torch.as_tensor(batch[key], dtype=torch.float32) for key in ("context", "future")}
    if "channel_mask" in batch:
        result["channel_mask"] = torch.as_tensor(batch["channel_mask"], dtype=torch.bool)
    for key in ("teacher_embedding", "noise_std"):
        if key in batch:
            result[key] = torch.as_tensor(batch[key], dtype=torch.float32)
    for key in ("equation_target_mask", "noise_mask"):
        if key in batch:
            result[key] = torch.as_tensor(batch[key], dtype=torch.bool)
    return result


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


CORPUS_FIELDS = ("seed", "batch_size", "context", "horizon", "channels",
                 "independent_fraction", "controlled_updates")
LMC_CORPUS_FIELDS = ("independent_bank", "correlated_bank", "dirichlet_min", "dirichlet_max",
                     "weibull_shape", "weibull_scale")


def corpus_fields(protocol, arm):
    """Protocol fields that determine an arm's prepared batches.

    Backbone, optimizer and logging settings do not change the data, so a run
    with a different backbone can reuse a corpus whose data fields match.
    """
    values = protocol if isinstance(protocol, dict) else asdict(protocol)
    names = CORPUS_FIELDS + (LMC_CORPUS_FIELDS if arm == "lmc" else
                             tuple(sorted(name for name in values if name.startswith("own_"))))
    return {name: values.get(name) for name in names}


def _channel_families(item, channels):
    """Per-example channel families of an own-generator batch, padded to ``channels``."""
    rows = []
    for metadata in item.get("metadata", ()):
        system = metadata["system"]
        families = list(metadata.get("channel_families", ("ast",) * system.num_channels))
        rows.append(families + ["padding"] * (channels - len(families)))
    return rows


def _count_families(rows, counts):
    for families in rows:
        for family in families:
            if family != "padding":
                counts[family] = counts.get(family, 0) + 1


def prepare_corpus(protocol, arm, upstream, root, *, equation_head=False,
                   teacher_source=None, teacher_revision=None):
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
    if equation_head and arm != "own":
        raise ValueError("Equation head is defined only for the proposed generator")
    teacher = (EquationTargetTeacher(teacher_source, teacher_revision or "",
                                    channels=protocol.channels, max_lag=protocol.own_max_lag,
                                    max_degree=protocol.own_max_degree)
               if equation_head else None)
    if teacher:
        estimated_bytes += (protocol.controlled_updates * protocol.batch_size *
                            protocol.channels * (teacher.dimension * 4 + 4 + 2))
    root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root.parent).free < estimated_bytes * 1.2:
        raise OSError(f"Need at least {estimated_bytes * 1.2 / 1e9:.1f} GB free for the controlled corpus")
    root.mkdir(parents=True, exist_ok=True)
    counts = protocol.stage_updates()
    started = time.perf_counter()
    manifest = {"arm": arm, "counts": counts, "seed": protocol.seed,
                "estimated_batch_bytes": estimated_bytes,
                "protocol": asdict(protocol), "equation_head": equation_head,
                "teacher": teacher.provenance if teacher else None}
    if arm == "lmc":
        manifest["upstream_sha256"] = verify_timepfn(upstream)
    covered_equations = covered_noise = 0
    coverage, families = {}, {}
    for stage, count in zip(("independent", "correlated"), counts):
        folder = root / stage
        folder.mkdir()
        dataset = _dataset(protocol, arm, stage, upstream, root / "raw_cache", count)
        for index, item in enumerate(_loader(protocol, arm, dataset, "cpu")):
            batch = _batch_tensors(item)
            if arm == "own":
                rows = _channel_families(item, protocol.channels)
                _count_families(rows, families)
                batch["kernel_mask"] = torch.as_tensor(
                    [[family == "kernel_synth" for family in row] for row in rows])
            if teacher:
                batch.update({key: torch.as_tensor(value) for key, value in
                              teacher.targets(item, coverage=coverage).items()})
                covered_equations += int(batch["equation_target_mask"].sum())
                covered_noise += int(batch["noise_mask"].sum())
            temporary = folder / f"{index:08d}.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, **{key: tensor.numpy() for key, tensor in batch.items()})
            os.replace(temporary, folder / f"{index:08d}.npz")
    manifest["preparation_seconds"] = time.perf_counter() - started
    if arm == "own":
        total = sum(families.values())
        manifest["channel_families"] = families
        manifest["kernel_synth_fraction"] = families.get("kernel_synth", 0) / total if total else None
    if teacher:
        manifest["auxiliary_target_counts"] = {"skeleton": covered_equations,
                                               "noise": covered_noise}
        manifest["auxiliary_target_coverage"] = coverage
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
                 "tsfmxai/datasets/multiseries.py", "tsfmxai/datasets/hybrid.py",
                 "tsfmxai/generators/multiseries.py", "tsfmxai/generators/hybrid.py",
                 "tsfmxai/pipelines/equation_targets.py",
                 "tsfmxai/models/chronos2_equation.py",
                 "tsfmxai/models/chronos2_indexed.py"):
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
        device="cuda", tiny=False, equation_head=False, teacher_source=None,
        teacher_revision=None, embedding_weight=1.0, noise_weight=1.0):
    """Run one arm. A checkpoint contains the complete optimizer/clock state."""
    if arm not in ("lmc", "own") or study not in ("controlled", "time"):
        raise ValueError("arm must be lmc/own; study must be controlled/time")
    if equation_head and arm != "own":
        raise ValueError("Equation head is defined only for the proposed generator")
    if equation_head and not protocol.backbone_channel_embedding:
        # The skeleton target names series by absolute channel index, which the
        # backbone can only represent with the channel-index embedding.
        raise ValueError("The equation head requires backbone_channel_embedding=true")
    if min(embedding_weight, noise_weight) < 0:
        raise ValueError("Auxiliary loss weights must be nonnegative")
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
                 "upstream": provenance, "tiny": tiny, "equation_head": equation_head,
                 "teacher_source": teacher_source if equation_head else None,
                 "teacher_revision": teacher_revision if equation_head else None,
                 "embedding_weight": embedding_weight if equation_head else None,
                 "noise_weight": noise_weight if equation_head else None}
    environment = _environment(device, output)
    disk_free_at_start = shutil.disk_usage(output).free
    if study == "controlled":
        manifest = json.loads((Path(corpus) / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["arm"] != arm
                or corpus_fields(manifest["protocol"], arm) != corpus_fields(protocol, arm)
                or manifest.get("equation_head", False) != equation_head):
            raise ValueError("Corpus provenance does not match the run")
        if equation_head and (manifest["teacher"]["source"] != teacher_source or
                              manifest["teacher"]["revision"] != (teacher_revision or "")):
            raise ValueError("Teacher provenance does not match the corpus")
        signature["corpus_manifest_sha256"] = hashlib.sha256(
            (Path(corpus) / "manifest.json").read_bytes()).hexdigest()
    elif (output / "cache").exists() and not resume:
        raise FileExistsError("Time-budget run requires a cold, empty cache")
    random.seed(protocol.seed)
    np.random.seed(protocol.seed)
    torch.manual_seed(protocol.seed)
    forecaster = scratch_model(
        seed=protocol.seed, tiny=tiny, patch_size=protocol.backbone_patch_size,
        channel_index_embedding=protocol.channels if protocol.backbone_channel_embedding else None,
        horizon=protocol.horizon).to(device)
    if equation_head:
        if not teacher_source:
            raise ValueError("Equation head requires a pinned skeleton teacher")
        teacher = EquationTargetTeacher(teacher_source, teacher_revision or "",
                                        channels=protocol.channels, max_lag=protocol.own_max_lag,
                                        max_degree=protocol.own_max_degree, device=device)
        signature["teacher"] = teacher.provenance
        if study == "controlled" and manifest["teacher"] != teacher.provenance:
            raise ValueError("Teacher checkpoint digest differs from prepared corpus")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(protocol.seed + 1)
            model = Chronos2EquationModel(
                forecaster, skeleton_dimension=teacher.dimension).to(device)
    else:
        teacher = None
        model = forecaster
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.independent_lr,
                                 betas=(.95, .999), eps=1e-8)
    completed = 0
    run_families, run_coverage = {}, {}
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
                raw = source[index]
                item = _batch_tensors(raw)
                if equation_head:
                    item.update({key: torch.as_tensor(value) for key, value in
                                 teacher.targets(raw).items()})
                temporary = folder / f"{index:08d}.tmp"
                with temporary.open("wb") as stream:
                    np.savez(stream, **{key: tensor.numpy() for key, tensor in item.items()})
                os.replace(temporary, folder / f"{index:08d}.npz")
            (folder / "complete").write_text("fixed validation batches\n", encoding="utf-8")
        loader = DataLoader(SavedBatches(folder, protocol.validation_batches), batch_size=None,
                            num_workers=0, pin_memory=device.startswith("cuda"))
        metrics = evaluate(forecaster, loader, device, protocol.microbatch)
        if equation_head:
            skeleton_sum = noise_sum = pieces = 0
            relative_errors = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    for offset in range(0, len(batch["context"]), protocol.microbatch):
                        inputs = _inputs(batch, forecaster, device, offset,
                                         offset + protocol.microbatch)
                        for key in ("teacher_embedding", "equation_target_mask",
                                    "noise_std", "noise_mask"):
                            inputs[key] = batch[key][offset:offset + protocol.microbatch].reshape(
                                -1, *batch[key].shape[2:]).to(device)
                        head_output = model(inputs)
                        _, terms = equation_training_loss(head_output, inputs,
                                                          embedding_weight=embedding_weight,
                                                          noise_weight=noise_weight)
                        # |sigma_hat / sigma - 1| is the same in standardized and
                        # original units, because both share the context scale.
                        mask = inputs["noise_mask"].bool()
                        if mask.any():
                            ratio = (head_output["predicted_log_noise_std"].exp()[mask]
                                     / inputs["noise_std"][mask])
                            relative_errors.append((ratio - 1).abs().cpu())
                        skeleton_sum += terms["skeleton_embedding"].item()
                        noise_sum += terms["innovation_log_std"].item()
                        pieces += 1
            metrics["skeleton_embedding_loss"] = skeleton_sum / pieces
            metrics["innovation_log_std_loss"] = noise_sum / pieces
            if relative_errors:
                metrics["innovation_std_relative_error"] = float(torch.cat(relative_errors).mean())
            model.train()
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
                item = next(iterator)
                batch = _batch_tensors(item)
                if arm == "own" and study == "time":
                    _count_families(_channel_families(item, protocol.channels), run_families)
                if equation_head and study == "time":
                    batch.update({key: torch.as_tensor(value) for key, value in
                                  teacher.targets(item, coverage=run_coverage).items()})
                if equation_head and any(key not in batch for key in
                                         ("teacher_embedding", "equation_target_mask",
                                          "noise_std", "noise_mask")):
                    raise ValueError("Equation-head batch is missing teacher targets")
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
                    inputs = _inputs(batch, forecaster, device, offset,
                                     offset + protocol.microbatch)
                    if equation_head:
                        for key in ("teacher_embedding", "equation_target_mask",
                                    "noise_std", "noise_mask"):
                            inputs[key] = batch[key][offset:offset + protocol.microbatch].reshape(
                                -1, *batch[key].shape[2:]).to(device, non_blocking=True)
                        result = model(inputs)
                        step_loss, _ = equation_training_loss(
                            result, inputs, embedding_weight=embedding_weight,
                            noise_weight=noise_weight)
                    else:
                        result = model(**inputs)
                        step_loss = result.loss
                    if not torch.isfinite(step_loss):
                        raise FloatingPointError("Nonfinite training loss")
                    weight = protocol.microbatch / protocol.batch_size
                    (step_loss * weight).backward()
                    loss += step_loss.detach().item() * weight
                optimizer.step()
                _sync(device)
                completed += 1
                stage_update = index + 1
                row = {"stage": name, "update": completed, "stage_update": stage_update,
                       "elapsed_seconds": elapsed(), "batch_wait_seconds": ready-waiting,
                       "lr": lr, "loss": loss, "windows_seen": completed * protocol.batch_size}
                if equation_head:
                    row["skeleton_targets"] = int(batch["equation_target_mask"].sum())
                    row["noise_targets"] = int(batch["noise_mask"].sum())
                if "channel_mask" in batch:
                    row["active_channels"] = int(batch["channel_mask"].sum())
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
                forecaster.save_pretrained(output / "independent_end")
                if equation_head:
                    torch.save({"skeleton_head": model.skeleton_head.state_dict(),
                                "noise_head": model.noise_head.state_dict()},
                               output / "independent_end" / "auxiliary_heads.pt")
                stage = 1
                stage_update = 0
                save()
        reason = "budget_elapsed" if study == "time" and elapsed() >= limit else "data_exhausted"
    except KeyboardInterrupt:
        interrupted = True
        reason = "interrupted"
    save()
    if not interrupted:
        forecaster.save_pretrained(output / "model_final")
        if equation_head:
            torch.save({"skeleton_head": model.skeleton_head.state_dict(),
                        "noise_head": model.noise_head.state_dict()},
                       output / "model_final" / "auxiliary_heads.pt")
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
               # Counted since the last start; the controlled corpus manifest holds
               # the complete counts for that study.
               "channel_families": run_families if arm == "own" and study == "time" else None,
               "auxiliary_target_coverage": run_coverage if equation_head and study == "time" else None,
               "gpu": _gpu_sample(device),
               "parameters": sum(p.numel() for p in model.parameters())}
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
