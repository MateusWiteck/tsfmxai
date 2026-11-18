"""Export one real multiseries system and the generation statistics for the "pipeline" slide.

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_multiseries_demo.py

Everything comes from the production code path: ``sample_multiseries_system`` draws the
system, and ``generate_training_batches`` simulates it, retries, replaces and batches.

* ``demo``: one four-channel system with the default per-channel treatment prior
  (Shao-Wu 1/2, Jacobian 1/6, Heaps 1/3, original 0) under the ``random`` topology, its
  permissions before and after the treatment, the source-assigned trees, the canonical
  equation per channel, the realized startup history and one simulated trajectory.
* ``stats``: two runs of 500 six-channel systems x 2 trajectories (all four topologies
  equally likely, lags 1-16, at most 4 operators, degree 4, 128 steps, seed 314159),
  one with the default treated-only prior and one with the diagnostic mixture of
  report/english/chapters/experimental_design/multivariate_generation.tex
  (original .4, Shao-Wu .3, Jacobian .1, Heaps .2), which contains plain trees.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report" / "slides" / "scripts"))

import numpy as np  # noqa: E402

from export_generator_demo import poly_json, tree_json  # noqa: E402
from tsfmxai.datasets import TrainingFormat, TrajectoryConfig, generate_training_batches  # noqa: E402
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig  # noqa: E402
from tsfmxai.generators.initialization import RandomHistorySpec  # noqa: E402
from tsfmxai.generators.multiseries import (  # noqa: E402
    DEFAULT_TREATMENT_PROBS,
    DEPENDENCIES,
    MultiseriesConfig,
    draw_choice,
    sample_multiseries_system,
    stream_seed,
)
from tsfmxai.generators.shao_wu import global_expression_bounds, saturate_product_lags  # noqa: E402

AST = CombinatorialTreeConfig(minimum_operators=0, maximum_operators=4, minimum_lag=1,
                              maximum_lag=16, max_degree=4)
DEMO_SEED, DEMO_CHANNELS, DEMO_STEPS, DEMO_ROOT = 4, 4, 128, 7
STATS_SEED, STATS_SYSTEMS, STATS_CHANNELS = 314159, 500, 6
DIAGNOSTIC_PROBS = {"original": .4, "shao_wu": .3, "jacobian": .1, "heaps": .2}


def permissions_before_treatment(config: MultiseriesConfig, seed: int):
    """Replay the topology stream of sample_multiseries_system (stream_seed(seed, 0)).

    The system only keeps the permissions after Jacobian/Heaps channels were made
    self-only; the slide shows both, so the first draw is replayed here and checked below.
    """
    rng = np.random.default_rng(stream_seed(seed, 0))
    n = draw_choice(rng, config.channel_count_probs)
    dependency = draw_choice(rng, config.dependency_probs)
    order = tuple(int(i) for i in rng.permutation(n))
    allowed = {}
    for position, target in enumerate(order):
        candidates = (list(order[position-1:position]) if dependency == "linked_chain" else
                      list(order[:position]) if dependency == "previous_chain" else
                      [i for i in range(n) if i != target] if dependency == "random" else [])
        cap = min(len(candidates), config.max_external_sources)
        count = cap if dependency == "linked_chain" else int(rng.integers(cap + 1))
        external = rng.choice(candidates, size=count, replace=False).tolist() if count else []
        allowed[target] = sorted(int(i) for i in [target, *external])
    return dependency, order, [allowed[i] for i in range(n)]


def demo() -> dict:
    config = MultiseriesConfig(ast_config=AST, channel_count_probs={DEMO_CHANNELS: 1.},
                               dependency_probs={"random": 1.})
    system = sample_multiseries_system(config, seed=DEMO_SEED)
    dependency, order, before = permissions_before_treatment(config, DEMO_SEED)
    assert dependency == system.dependency and order == system.channel_order
    for i, t in enumerate(system.treatments):
        expected = [i] if t.method in ("jacobian", "heaps") else before[i]
        assert list(system.allowed_sources[i]) == expected, "topology replay drifted from the generator"

    gains = []
    for eq, t in zip(system.original_equations, system.treatments):
        if t.method != "shao_wu":
            gains.append(None)
            continue
        _, d = global_expression_bounds(saturate_product_lags(eq.expression), eq.binding,
                                        num_series=system.num_channels,
                                        order=max(1, system.history_length))
        gains.append(float(d.sum()))

    width = DEMO_STEPS
    stream = generate_training_batches(
        config=config, num_systems=1, system_seeds=[DEMO_SEED], trajectories_per_system=1,
        seed=DEMO_ROOT,
        trajectory=TrajectoryConfig(generated_steps=DEMO_STEPS, initialization=RandomHistorySpec()),
        training=TrainingFormat(context_length=width - 16, prediction_length=16, stride=width,
                                batch_size=1, dtype="float64"))
    batch = next(stream)
    meta = batch.metadata[0]
    assert meta["system"].seed == DEMO_SEED and meta["system_replacements"] == 0
    path = np.concatenate([batch.context[0], batch.future[0]], axis=0).T  # [channel, time]
    codec = stream.codec
    targets = [codec.encode(e)["target"] for e in system.equations]

    weights = []
    for i, src in enumerate(before):
        w = np.array([config.self_source_weight if j == i else 1. for j in src])
        weights.append({"sources": src, "weights": [float(v) for v in w / w.sum()],
                        "lag_probability": config.lag_sampling.probability(w)})

    return {
        "seed": DEMO_SEED, "channels": system.num_channels, "dependency": dependency,
        "order": list(order), "max_external": config.max_external_sources,
        "self_weight": config.self_source_weight,
        "permissions_before": before,
        "allowed": [list(s) for s in system.allowed_sources],
        "effective": [list(s) for s in system.effective_sources],
        "treatments": [t.method for t in system.treatments],
        "target_gain": system.treatments[0].target_gain,
        "safety_factor": system.treatments[0].safety_factor,
        "domain_radius": system.treatments[0].domain_radius,
        "gains": gains,
        "scales": [float(s) for s in system.treatment_scales],
        "innovations": [{"family": s.family, "std": float(s.std)} for s in system.innovations],
        "history_length": int(system.history_length),
        "original_trees": [tree_json(e.expression, e.binding) for e in system.original_equations],
        "canonical": [poly_json(p) for p in system.canonical],
        "target_tokens": [len(t) for t in targets],
        "target_text": [codec.target_text(t) for t in targets],
        "source_weights": weights,
        "initial_history": [[float(v) for v in row] for row in meta["initial_history"]],
        "initialization_components": meta["initialization_components"],
        "attempt": int(meta["attempt"]),
        "trajectory": [[float(v) for v in row] for row in path],
        "noise_std_target": [float(v) for v in batch.noise_targets["std"][0]],
        "noise_family_target": [int(v) for v in batch.noise_targets["family"][0]],
        "history_spec": {k: getattr(RandomHistorySpec(), k) for k in (
            "constant_probability", "noise_probability", "cosine_probability",
            "constant_magnitude", "noise_radius", "cosine_amplitude", "cosine_period")},
        "noise_std_range": list(config.noise.std_range),
        "treatment_probs": dict(config.treatment_probs),
        "lag_sampling": {"minimum": config.lag_sampling.minimum, "maximum": config.lag_sampling.maximum,
                         "tau": config.lag_sampling.tau,
                         "by_external": [config.lag_sampling.probability(
                             [config.self_source_weight] + [1.] * k) for k in range(3)]},
    }


def stats(treatment_probs: dict) -> dict:
    config = MultiseriesConfig(ast_config=AST, channel_count_probs={STATS_CHANNELS: 1.},
                               dependency_probs={d: 1. / len(DEPENDENCIES) for d in DEPENDENCIES},
                               treatment_probs=treatment_probs)
    training = TrainingFormat(context_length=64, prediction_length=16, stride=16, batch_size=32)
    stream = generate_training_batches(
        config=config, num_systems=STATS_SYSTEMS, trajectories_per_system=2, seed=STATS_SEED,
        trajectory=TrajectoryConfig(generated_steps=128, initialization=RandomHistorySpec()),
        training=training)
    start, shapes = time.perf_counter(), None
    for batch in stream:
        if shapes is None:
            shapes = {"context": list(batch.context.shape), "future": list(batch.future.shape),
                      "channel_mask": list(batch.channel_mask.shape),
                      "equation_targets": list(batch.equation_targets.shape),
                      "noise_std": list(batch.noise_targets["std"].shape)}
    seconds = time.perf_counter() - start
    s = stream.stats
    return {
        "treatment_probs": treatment_probs, "systems": s.attempted_systems,
        "trajectories_attempted": s.attempted_trajectories,
        "initialization_attempts": s.initialization_attempts,
        "accepted_trajectories": s.accepted_trajectories,
        "replaced_systems": s.replaced_systems, "discarded_trajectories": s.discarded_trajectories,
        "resampled_equations": s.resampled_equations,
        "resampled_original_equations": s.resampled_original_equations,
        "failures_by_reason": dict(Counter(f["reason"] for f in s.failures)),
        "failures_by_attempt": {str(k): v for k, v in sorted(Counter(f["attempt"] for f in s.failures).items())},
        "windows": s.yielded_windows, "seconds": round(seconds, 1), "batch_shapes": shapes,
    }


def main() -> None:
    out = {
        "demo": demo(),
        "stats": {"seed": STATS_SEED, "channels": STATS_CHANNELS, "systems": STATS_SYSTEMS,
                  "default": stats(dict(DEFAULT_TREATMENT_PROBS)),
                  "diagnostic": stats(DIAGNOSTIC_PROBS)},
        "python": sys.version.split()[0],
    }
    path = ROOT / "report" / "slides" / "data" / "multiseries_demo.js"
    path.write_text("window.MULTI_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
    d, st = out["demo"], out["stats"]
    print("wrote", path.name, d["treatments"], d["effective"],
          {k: (st[k]["replaced_systems"], st[k]["initialization_attempts"], st[k]["seconds"]) for k in ("default", "diagnostic")})


if __name__ == "__main__":
    main()
