import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import json, platform, subprocess
from pathlib import Path
from time import perf_counter
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import RandomHistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig
from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat
from tsfmxai.datasets.online import CurriculumStage, OnlineBatchPlan, make_dataloader

def main():
    results = []
    for dep in ("independent", "random", "linked_chain"):
        cfg = MultiseriesConfig(ast_config=CombinatorialTreeConfig(maximum_operators=4, maximum_lag=16, max_degree=4), channel_count_probs={6:1.}, dependency_probs={dep:1.})
        plan = OnlineBatchPlan(stages=(CurriculumStage(0, cfg, systems_per_batch=4, trajectories_per_system=2),), total_batches=120, root_seed=314159, trajectory=TrajectoryConfig(128, RandomHistorySpec()), training=TrainingFormat(64,16,16,batch_size=32))
        loader = make_dataloader(plan, num_workers=8, pin_memory=False)
        start = perf_counter()
        iterator = iter(loader)
        batch = next(iterator)
        first = perf_counter()-start
        assert all(t.method != "original" for m in batch['metadata'] for t in m['system'].treatments)
        assert tuple(batch['context'].shape) == (32,64,6)
        for _ in range(19): next(iterator)
        warmup = perf_counter()-start
        timed = perf_counter()
        count = 0
        for batch in iterator:
            assert all(t.method != "original" for m in batch['metadata'] for t in m['system'].treatments)
            assert tuple(batch['context'].shape) == (32,64,6)
            count += 1
        elapsed = perf_counter()-timed
        print(dep, 'first_batch_s',round(first,3),'warmup_20_s',round(warmup,3),'measured_batches',count,'generation_s',round(elapsed,3),'seconds_per_batch',round(elapsed/count,4),'batches_per_second',round(count/elapsed,3),flush=True)
        results.append(dict(topology=dep,first_batch_seconds=first,warmup_seconds=warmup,measured_batches=count,elapsed_seconds=elapsed,seconds_per_batch=elapsed/count))
        del iterator, loader
    root = Path(__file__).resolve().parents[3]
    record = dict(seed=314159, workers=8, prefetch=2, warmup_batches=20, measured_batches=100,
        configuration=dict(channels=6,systems_per_batch=4,trajectories_per_system=2,steps=128,context=64,future=16,stride=16,batch_size=32,maximum_operators=4,maximum_lag=16,max_degree=4,treatments={"original":0,"shao_wu":.5,"jacobian":1/6,"heaps":1/3}),
        results=results,python=platform.python_version(),platform=platform.platform(),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip())
    Path(__file__).with_name("results.json").write_text(json.dumps(record,indent=2))
if __name__ == '__main__': main()
