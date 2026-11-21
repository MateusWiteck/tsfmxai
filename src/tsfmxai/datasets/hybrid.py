"""Streaming hybrid channel batches through the established batch contract."""

from tsfmxai.datasets.multiseries import TrainingBatchStream
from tsfmxai.generators.hybrid import HybridConfig, sample_hybrid_system
from tsfmxai.generators.multiseries import stream_seed


class HybridBatchStream(TrainingBatchStream):
    def __init__(self, *, config: HybridConfig, num_systems: int,
                 trajectories_per_system: int, trajectory, training, seed: int,
                 system_seeds=None):
        self.hybrid_config = config
        super().__init__(config=config.ast, systems=None, num_systems=num_systems,
                         trajectories_per_system=trajectories_per_system,
                         trajectory=trajectory, training=training, seed=seed,
                         system_seeds=system_seeds)

    def _system(self, system_index, replacement):
        if replacement == 0:
            seed = (self.system_seeds[system_index] if self.system_seeds is not None
                    else stream_seed(self.seed, 10, system_index))
        else:
            seed = stream_seed(self.seed, 11, system_index, replacement)
        return sample_hybrid_system(self.hybrid_config, seed=seed)


def generate_hybrid_training_batches(*, config: HybridConfig, num_systems: int,
                                     trajectories_per_system: int = 1,
                                     trajectory, training, seed: int = 0,
                                     system_seeds=None) -> HybridBatchStream:
    return HybridBatchStream(config=config, num_systems=num_systems,
                             trajectories_per_system=trajectories_per_system,
                             trajectory=trajectory, training=training, seed=seed,
                             system_seeds=system_seeds)
