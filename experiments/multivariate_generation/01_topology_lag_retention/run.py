import argparse
from collections import Counter
from dataclasses import replace
from time import perf_counter
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import RandomHistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig, LagRetentionConfig
from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat, TrainingBatchStream, generate_training_batches

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=314159)
parser.add_argument("--retention", choices=("off", "on"), required=True)
args = parser.parse_args()
base = MultiseriesConfig(ast_config=CombinatorialTreeConfig(maximum_operators=4, maximum_lag=16, max_degree=4), channel_count_probs={6: 1.}, dependency_probs={"independent": 1.}, lag_retention=LagRetentionConfig() if args.retention == "on" else None)
original_system = TrainingBatchStream._system
original_trajectory = TrainingBatchStream._trajectory
for dependency in ("independent", "random", "linked_chain"):
    counts = Counter()
    timings = Counter()
    def measured_system(self, *args):
        start = perf_counter()
        system = original_system(self, *args)
        timings["system"] += perf_counter() - start
        counts.update(t.method for t in system.treatments)
        counts["draws"] += 1
        return system
    def measured_trajectory(self, *args):
        start = perf_counter()
        result = original_trajectory(self, *args)
        timings["trajectory"] += perf_counter() - start
        return result
    TrainingBatchStream._system = measured_system
    TrainingBatchStream._trajectory = measured_trajectory
    cfg = replace(base, dependency_probs={dependency: 1.})
    start = perf_counter()
    stream = generate_training_batches(config=cfg, num_systems=500, trajectories_per_system=2, trajectory=TrajectoryConfig(128, RandomHistorySpec()), training=TrainingFormat(64, 16, 16, batch_size=32), seed=args.seed)
    windows = sum(batch.context.shape[0] for batch in stream)
    elapsed = perf_counter() - start
    print(dependency, "elapsed", round(elapsed, 3), "system_time", round(timings["system"], 3), "trajectory_time", round(timings["trajectory"], 3), "other", round(elapsed-timings["system"]-timings["trajectory"], 3), "draws", counts["draws"], "method_draws", {k: counts[k] for k in ("original", "shao_wu", "jacobian", "heaps")}, "replacements", stream.stats.replaced_systems, "resampled_original", stream.stats.resampled_original_equations, "initialization_attempts", stream.stats.initialization_attempts, "windows", windows, flush=True)
TrainingBatchStream._system = original_system
TrainingBatchStream._trajectory = original_trajectory
