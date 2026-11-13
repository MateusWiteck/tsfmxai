"""Streaming, padded NumPy batches with canonical equation and noise targets.

Simulation is float64; output conversion is checked without clipping. No implicit
normalization, burn-in, or magnitude threshold is applied. Consumers may wrap
arrays with torch.from_numpy without requiring torch in the generator.
"""

from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np

from tsfmxai.generators.initialization import HistorySpec, make_initial_history
from tsfmxai.generators.multiseries import (
    MultiseriesConfig, MultiseriesSystem, sample_multiseries_system,
    positive_integer, stream_seed,
)
from tsfmxai.generators.stochastic import simulate_innovation_path
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.representations.expression import Lag, walk


@dataclass(frozen=True)
class TrajectoryConfig:
    """Controls recurrence outputs after a separately sampled startup history.

    ``generated_steps`` counts only values computed by the sampled equations.
    The startup history is used to seed the recurrence and is not returned by
    ``simulate_innovation_path`` or included in training windows. No additional
    burn-in is applied by this stream.
    """

    generated_steps: int
    initialization: HistorySpec
    max_initialization_retries: int = 3
    on_retry_exhausted: str = "skip"
    resample_innovations: bool = False

    def __post_init__(self):
        positive_integer(self.generated_steps, "generated_steps")
        positive_integer(self.max_initialization_retries, "max_initialization_retries", minimum=0)
        if self.on_retry_exhausted not in ("skip", "raise"):
            raise ValueError("on_retry_exhausted must be skip or raise.")
        if type(self.resample_innovations) is not bool:
            raise ValueError("resample_innovations must be boolean.")
        if self.initialization.kind == "stationary":
            raise ValueError("A mixed system has no retained Heaps joint stationary law. "
                             "Use zero, constant, uniform or cosine initialization.")


@dataclass(frozen=True)
class TrainingFormat:
    context_length: int
    prediction_length: int
    stride: int
    batch_size: int = 32
    dtype: str = "float32"
    layout: str = "batch_time_channel"
    noise_target: str = "scale_and_family"

    def __post_init__(self):
        for name in ("context_length", "prediction_length", "stride", "batch_size"):
            positive_integer(getattr(self, name), name)
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Output dtype must be float32 or float64.")
        if self.layout not in ("batch_time_channel", "batch_channel_time"):
            raise ValueError("Unknown tensor layout.")
        if self.noise_target not in ("scale", "scale_and_family"):
            raise ValueError("Unknown noise target.")


@dataclass
class GenerationStats:
    attempted_systems: int = 0
    attempted_trajectories: int = 0
    initialization_attempts: int = 0
    accepted_trajectories: int = 0
    skipped_trajectories: int = 0
    yielded_windows: int = 0
    failures: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingBatch:
    context: np.ndarray
    future: np.ndarray
    channel_mask: np.ndarray
    equation_targets: np.ndarray
    equation_mask: np.ndarray
    noise_targets: dict[str, np.ndarray]
    metadata: tuple[dict, ...]


def _innovations(system, steps, seed):
    rng = np.random.default_rng(seed)
    noise = np.empty((system.num_channels, steps), dtype=np.float64)
    for channel, spec in enumerate(system.innovations):
        if spec.family == "gaussian":
            noise[channel] = rng.normal(0., spec.std, steps)
        else:
            bound = np.sqrt(3.) * spec.std
            noise[channel] = rng.uniform(-bound, bound, steps)
    return noise


def _collate(windows, training, codec):
    size, channels = len(windows), codec.num_series
    context = np.zeros((size, training.context_length, channels), dtype=training.dtype)
    future = np.zeros((size, training.prediction_length, channels), dtype=training.dtype)
    channel_mask = np.zeros((size, channels), dtype=bool)
    encoded = [w["targets"] for w in windows]
    tokens = max(len(seq) for sample in encoded for seq in sample)
    equations = np.zeros((size, channels, tokens), dtype=np.int64)
    equation_mask = np.zeros_like(equations, dtype=bool)
    std = np.zeros((size, channels), dtype=training.dtype)
    family = np.full((size, channels), -1, dtype=np.int64)
    metadata = []
    for row, (window, sequences) in enumerate(zip(windows, encoded)):
        system = window["system"]
        n = system.num_channels
        context[row, :, :n] = window["context"].T
        future[row, :, :n] = window["future"].T
        channel_mask[row, :n] = True
        for channel, sequence in enumerate(sequences):
            equations[row, channel, :len(sequence)] = sequence
            equation_mask[row, channel, :len(sequence)] = True
            std[row, channel] = system.innovations[channel].std
            family[row, channel] = int(system.innovations[channel].family == "uniform")
        metadata.append(window["metadata"])
    if training.layout == "batch_channel_time":
        context, future = context.transpose(0, 2, 1), future.transpose(0, 2, 1)
    noise = {"std": std}
    if training.noise_target == "scale_and_family":
        noise["family"] = family  # 0 Gaussian, 1 uniform, -1 padded channel
    return TrainingBatch(context, future, channel_mask, equations, equation_mask, noise, tuple(metadata))


class TrainingBatchStream:
    """Single-pass iterator with a fixed token vocabulary and live failure statistics.

Metadata retains the full system and realized initial history. Seeds are indexed
by system/trajectory/attempt, so skips and batch-size changes do not shift RNGs.
"""

    def __init__(self, *, config, systems, num_systems, trajectories_per_system,
                 trajectory, training, seed):
        if (config is None) == (systems is None):
            raise ValueError("Provide exactly one of config or systems.")
        positive_integer(trajectories_per_system, "trajectories_per_system")
        positive_integer(seed, "seed", minimum=0)
        if training.context_length + training.prediction_length > trajectory.generated_steps:
            raise ValueError("Generated steps must cover context plus future; startup is excluded.")
        self.config = config
        self.systems = tuple(systems) if systems is not None else None
        if config is not None:
            positive_integer(num_systems, "num_systems")
            self.num_systems = num_systems
            channels = max(c for c, p in config.channel_count_probs.items() if p > 0)
            minimum_lag, maximum_lag = config.ast_config.minimum_lag, config.ast_config.maximum_lag
            degree = config.ast_config.max_degree
        else:
            if not self.systems or num_systems is not None:
                raise ValueError("Explicit systems must be nonempty; omit num_systems.")
            self.num_systems = len(self.systems)
            channels = max(s.num_channels for s in self.systems)
            # Codec.encode visits the retained AST too, including cancelled terms.
            lags = [e.binding.lag_values[node.slot] for s in self.systems for e in s.equations
                    for node in walk(e.expression) if isinstance(node, Lag)]
            minimum_lag = min(lags, default=1)
            maximum_lag = max(lags, default=1)
            degree = max(p.max_degree for s in self.systems for p in s.canonical)
        self.codec = CanonicalCodec(channels, minimum_lag, maximum_lag, degree)
        self.trajectories_per_system = trajectories_per_system
        self.trajectory, self.training, self.seed = trajectory, training, seed
        self.stats = GenerationStats()
        self._iterator = self._generate()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def _trajectory(self, system, system_index, trajectory_index):
        cfg, dtype = self.trajectory, self.training.dtype
        self.stats.attempted_trajectories += 1
        root = stream_seed(self.seed, 20, system_index, trajectory_index)
        for attempt in range(cfg.max_initialization_retries + 1):
            self.stats.initialization_attempts += 1
            init_seed = stream_seed(root, 0, attempt)
            noise_seed = stream_seed(root, 1, attempt if cfg.resample_innovations else 0)
            unit = np.random.default_rng(init_seed).uniform(
                -1., 1., (system.num_channels, system.history_length))
            history = make_initial_history(cfg.initialization, system, unit, init_seed)
            innovations = _innovations(system, cfg.generated_steps, noise_seed)
            reason = None
            if not np.isfinite(history).all() or not np.isfinite(innovations).all():
                reason = "nonfinite_inputs"
            else:
                try:
                    # This path contains recurrence outputs only. ``history`` is
                    # an initial condition for evaluating the first output, not
                    # a prefix that may be exposed as model context.
                    path = simulate_innovation_path(system.expressions, system.bindings,
                                                    history, innovations, stop_threshold=None)
                    with np.errstate(over="ignore", invalid="ignore"):
                        output = path.astype(dtype)
                        noise_std = np.asarray([s.std for s in system.innovations], dtype=dtype)
                    if not np.isfinite(path).all():
                        reason = "nonfinite_simulation"
                    elif not np.isfinite(output).all() or not np.isfinite(noise_std).all():
                        reason = "nonfinite_output_dtype"
                except OverflowError:
                    reason = "arithmetic_overflow"
            record = dict(system_index=system_index, trajectory_index=trajectory_index,
                          attempt=attempt, initialization_seed=init_seed, innovation_seed=noise_seed)
            if reason is None:
                self.stats.accepted_trajectories += 1
                return output, dict(**record, system=system, initial_history=history,
                                    initialization=cfg.initialization, root_seed=self.seed,
                                    trajectory_config=cfg, training_format=self.training,
                                    simulation_dtype="float64", numpy_version=np.__version__)
            self.stats.failures.append(dict(**record, reason=reason))
        self.stats.skipped_trajectories += 1
        if cfg.on_retry_exhausted == "raise":
            raise RuntimeError(f"Trajectory exhausted initialization attempts: {record}; {reason}")
        return None

    def _generate(self):
        windows = []
        cfg = self.training
        width = cfg.context_length + cfg.prediction_length
        for system_index in range(self.num_systems):
            self.stats.attempted_systems += 1
            system = (self.systems[system_index] if self.systems is not None else
                      sample_multiseries_system(self.config, seed=stream_seed(self.seed, 10, system_index)))
            targets = [self.codec.encode(e)["target"] for e in system.equations]
            # Configuration errors (e.g. a too-short cosine prefix) propagate;
            # retrying a numerical trajectory must not hide invalid settings.
            for trajectory_index in range(self.trajectories_per_system):
                result = self._trajectory(system, system_index, trajectory_index)
                if result is None:
                    continue
                path, metadata = result
                for start in range(0, path.shape[1] - width + 1, cfg.stride):
                    split = start + cfg.context_length
                    windows.append(dict(context=path[:, start:split], future=path[:, split:start+width],
                                        system=system, targets=targets,
                                        metadata=dict(metadata, window_start=start)))
                    self.stats.yielded_windows += 1
                    if len(windows) == cfg.batch_size:
                        yield _collate(windows, cfg, self.codec)
                        windows = []
        if windows:
            yield _collate(windows, cfg, self.codec)


def generate_training_batches(*, config: MultiseriesConfig | None = None,
                              systems: Sequence[MultiseriesSystem] | None = None,
                              num_systems: int | None = None, trajectories_per_system: int = 1,
                              trajectory: TrajectoryConfig, training: TrainingFormat,
                              seed: int = 0) -> TrainingBatchStream:
    """Create a streaming dataset; consume it to execute generation.

Supply either config + num_systems or explicit systems. Retries only change
startup histories by default; replacing failed systems is deliberately not
implemented until the proposed throughput/selection study determines a policy.
"""
    return TrainingBatchStream(config=config, systems=systems, num_systems=num_systems,
                               trajectories_per_system=trajectories_per_system,
                               trajectory=trajectory, training=training, seed=seed)
