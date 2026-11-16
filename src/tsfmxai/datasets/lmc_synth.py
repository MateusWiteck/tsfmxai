"""Seeded, cached windows from the pinned, unmodified TimePFN generators.

One dataset item is a complete effective batch, so CPU workers prepare batches
while the parent process alone owns CUDA. The cache stores raw trajectories,
not noisy windows; augmentation is reproducible independently of worker count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from threadpoolctl import threadpool_limits

TIMEPFN_REVISION = "e11d5c2982b0e49b87873cbddcb7712641861268"


@dataclass(frozen=True)
class LMCSpec:
    # Four mandatory values: the authors did not publish numerical defaults.
    dirichlet_min: float
    dirichlet_max: float
    weibull_shape: float
    weibull_scale: float
    channels: int = 6
    length: int = 1024
    context: int = 96
    horizon: int = 96
    batch_size: int = 16
    independent_bank: int = 20000
    correlated_bank: int = 15000
    max_kernels: int = 5
    lmc_stride: int = 8
    noise_std: float = 0.1
    seed: int = 2021

    def __post_init__(self):
        if not (0 < self.dirichlet_min <= self.dirichlet_max):
            raise ValueError("Invalid Dirichlet interval")
        if self.weibull_shape <= 0 or self.weibull_scale <= 0:
            raise ValueError("Weibull parameters must be positive")
        if self.length != 1024:
            raise ValueError("Upstream kernel bank is normalized to length 1024")
        if (self.channels < 2 or min(self.context, self.horizon, self.batch_size,
                                    self.lmc_stride, self.max_kernels) < 1
                or self.context + self.horizon > self.length or self.noise_std < 0):
            raise ValueError("Invalid batch/window dimensions")
        if self.independent_bank <= self.channels or self.correlated_bank < 1:
            raise ValueError("Insufficient trajectories")

    def groups(self, family):
        if family == "independent":
            # Reproduce upstream's default dataset[:-1] and overlapping grouping.
            return (self.independent_bank - 1 - self.channels) // min(self.channels // 2, 20) + 1
        if family == "lmc":
            return self.correlated_bank
        raise ValueError(f"Unknown family: {family}")

    def windows(self, family):
        stride = 1 if family == "independent" else self.lmc_stride
        return self.groups(family) * ((self.length - self.context - self.horizon) // stride + 1)

    def updates(self, family):
        return self.windows(family) // self.batch_size


def verify_timepfn(path):
    """Reject a different revision or modified generator rather than mislabel it."""
    path = Path(path).resolve()
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != TIMEPFN_REVISION:
        raise ValueError(f"TimePFN must be pinned to {TIMEPFN_REVISION}")
    relative = "synthetic_data_generation"
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain",
                                     "--untracked-files=no", "--", relative], text=True)
    if dirty.strip():
        raise ValueError("TimePFN generator has local modifications")
    return {name: hashlib.sha256((path / relative / name).read_bytes()).hexdigest()
            for name in ("LMC_Synth.py", "kernel-synth.py")}


@lru_cache(maxsize=4)
def _module(root, family):
    name = "LMC_Synth.py" if family == "lmc" else "kernel-synth.py"
    spec = importlib.util.spec_from_file_location("_timepfn_" + family,
                                                 Path(root) / "synthetic_data_generation" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if family == "lmc":
        original = module.sample_from_gp_prior_efficient

        def seeded_sampler(kernel, X, random_seed=None, method="eigh"):
            # default_rng(None) upstream otherwise escapes np.random.seed.
            if random_seed is None:
                random_seed = int(np.random.randint(0, 2**32, dtype=np.uint32))
            return original(kernel, X, random_seed=random_seed, method=method)

        module.sample_from_gp_prior_efficient = seeded_sampler
    return module


def _seed(*values):
    return int(np.random.SeedSequence(values).generate_state(1)[0])


class TimePFNBatches(Dataset):
    """Deterministic bank, window order, and augmentation; split 0/1/2 is disjoint."""

    def __init__(self, spec: LMCSpec, upstream, cache, family="lmc", *, split=0,
                 limit=None, start=0):
        self.spec, self.family, self.split, self.start = spec, family, split, start
        self.upstream = str(Path(upstream).resolve())
        provenance = verify_timepfn(self.upstream)
        key = hashlib.sha256(json.dumps([asdict(spec), provenance], sort_keys=True).encode()).hexdigest()[:20]
        self.cache = Path(cache).resolve() / key / str(split) / family
        self.cache.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(_seed(spec.seed, split, family == "lmc", 99))
        self.order = rng.permutation(spec.windows(family))
        self.stop = spec.updates(family) if limit is None else min(limit, spec.updates(family))
        if not 0 <= start <= self.stop:
            raise ValueError("Invalid resume offset")

    def __len__(self):
        return self.stop - self.start

    def trajectory(self, index):
        path = self.cache / f"{index:06d}.npy"
        if path.exists():
            return np.load(path, allow_pickle=False)
        module = _module(self.upstream, self.family)
        state = np.random.get_state()
        try:
            np.random.seed(_seed(self.spec.seed, self.split, self.family == "lmc", index))
            with threadpool_limits(limits=1):
                if self.family == "lmc":
                    result = module.generate_time_series(
                        self.spec.dirichlet_min, self.spec.dirichlet_max,
                        self.spec.weibull_shape, self.spec.weibull_scale,
                        max_kernels=self.spec.max_kernels, num_channels=self.spec.channels)
                else:
                    result = module.generate_time_series(max_kernels=self.spec.max_kernels)
            array = np.asarray(result["target"], dtype=np.float32)
            expected = (self.spec.channels, 1024) if self.family == "lmc" else (1024,)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"Invalid trajectory {self.family}/{index}: {array.shape}")
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            os.replace(temporary, path)  # Concurrent writers generate identical data.
            return array
        finally:
            np.random.set_state(state)

    def __getitem__(self, index):
        index += self.start
        s = self.spec
        ids = self.order[index * s.batch_size:(index + 1) * s.batch_size]
        contexts, futures = [], []
        for sample in ids:
            group = int(sample) % s.groups(self.family)
            window = int(sample) // s.groups(self.family)
            if self.family == "lmc":
                raw = self.trajectory(group)
                offset = window * s.lmc_stride
            else:
                windows_per_group = s.length - s.context - s.horizon + 1
                group, window = divmod(int(sample), windows_per_group)
                base = group * min(s.channels // 2, 20)
                raw = np.stack([self.trajectory(base + c) for c in range(s.channels)])
                offset = window
            window_data = raw[:, offset:offset + s.context + s.horizon].T.copy()
            rng = np.random.default_rng(_seed(s.seed, self.split, self.family == "lmc", int(sample), 123))
            window_data *= rng.normal(1, s.noise_std, window_data.shape).astype(np.float32)
            contexts.append(window_data[:s.context])
            futures.append(window_data[s.context:])
        return {"context": torch.from_numpy(np.stack(contexts)),
                "future": torch.from_numpy(np.stack(futures)),
                "ids": torch.from_numpy(ids.copy())}


def _worker_init(_):
    torch.set_num_threads(1)


def batch_loader(dataset, workers=0, *, pin_memory=False):
    kwargs = dict(batch_size=None, num_workers=workers, pin_memory=pin_memory,
                  worker_init_fn=_worker_init, generator=torch.Generator().manual_seed(dataset.spec.seed))
    if workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)
