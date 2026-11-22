"""Export one real multiseries system and the generation statistics for the "pipeline" slide.

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_multiseries_demo.py

Everything comes from the production code path: ``sample_multiseries_system`` draws the
system, and ``generate_training_batches`` simulates it, retries, replaces and batches.
Treatments follow the dependency-driven rule: a channel with a permitted external source
uses Shao-Wu and reads at least one external series; a channel without one draws Jacobian
or Heaps (``self_only_treatment_probs``).

* ``demo``: one four-channel system (four for legibility; training uses six) under the
  ``random`` topology, with the AST settings of the forecasting arm except at most four
  operators: its permissions, the source-assigned trees, the canonical equation per
  channel, the realized startup history and one simulated trajectory.
* ``stats``: 500 six-channel systems x 2 trajectories with the AST settings of the
  forecasting arm (random topology, one to six operators, lags 1-16, degree 4, parameter
  magnitudes 0.05-1, 128 steps, seed 314159), including the realized treatment shares.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report" / "slides" / "scripts"))

import numpy as np  # noqa: E402

from export_generator_demo import poly_json, tree_json  # noqa: E402
from tsfmxai.datasets import TrainingFormat, TrajectoryConfig, generate_training_batches  # noqa: E402
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig  # noqa: E402
from tsfmxai.generators.initialization import RandomHistorySpec  # noqa: E402
from tsfmxai.generators.multiseries import MultiseriesConfig, sample_multiseries_system  # noqa: E402
from tsfmxai.generators.shao_wu import global_expression_bounds, saturate_product_lags  # noqa: E402

FORECASTING_AST = dict(minimum_operators=1, maximum_operators=6, minimum_lag=1, maximum_lag=16,
                       max_degree=4, minimum_parameter_magnitude=0.05,
                       maximum_parameter_magnitude=1.0)
DEMO_AST = CombinatorialTreeConfig(**{**FORECASTING_AST, "maximum_operators": 4})
STATS_AST = CombinatorialTreeConfig(**FORECASTING_AST)
DEMO_CHANNELS, DEMO_STEPS, DEMO_ROOT = 4, 128, 7
STATS_SEED, STATS_SYSTEMS, STATS_CHANNELS = 314159, 500, 6


def demo_config() -> MultiseriesConfig:
    return MultiseriesConfig(ast_config=DEMO_AST, channel_count_probs={DEMO_CHANNELS: 1.},
                             dependency_probs={"random": 1.})


def pick_demo_seed(config: MultiseriesConfig) -> int:
    """First seed whose four channels show all three treatments and one cross-series read."""
    for seed in range(1, 500):
        system = sample_multiseries_system(config, seed=seed)
        methods = {t.method for t in system.treatments}
        if (methods == {"shao_wu", "jacobian", "heaps"}
                and max(len(s) for s in system.allowed_sources) >= 3):
            return seed
    raise RuntimeError("no demo seed with all treatments")


def demo() -> dict:
    config = demo_config()
    seed = pick_demo_seed(config)
    system = sample_multiseries_system(config, seed=seed)
    for i, t in enumerate(system.treatments):
        externals = [j for j in system.allowed_sources[i] if j != i]
        assert (t.method == "shao_wu") == bool(externals), "treatment does not follow the rule"
        if t.method == "shao_wu":
            assert any(j != i for j in system.effective_sources[i]), "Shao-Wu reads no external"

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
        config=config, num_systems=1, system_seeds=[seed], trajectories_per_system=1,
        seed=DEMO_ROOT,
        trajectory=TrajectoryConfig(generated_steps=DEMO_STEPS, initialization=RandomHistorySpec()),
        training=TrainingFormat(context_length=width - 16, prediction_length=16, stride=width,
                                batch_size=1, dtype="float64"))
    batch = next(stream)
    meta = batch.metadata[0]
    assert meta["system"].seed == seed and meta["system_replacements"] == 0
    path = np.concatenate([batch.context[0], batch.future[0]], axis=0).T  # [channel, time]
    codec = stream.codec
    targets = [codec.encode(e)["target"] for e in system.equations]

    weights = []
    for i, src in enumerate(system.allowed_sources):
        w = np.array([config.self_source_weight if j == i else 1. for j in src])
        weights.append({"sources": list(src), "weights": [float(v) for v in w / w.sum()]})
    shao_wu = next(t for t in system.treatments if t.method == "shao_wu")
    return {
        "seed": seed, "channels": system.num_channels, "dependency": system.dependency,
        "order": list(system.channel_order), "max_external": config.max_external_sources,
        "self_weight": config.self_source_weight,
        "allowed": [list(s) for s in system.allowed_sources],
        "effective": [list(s) for s in system.effective_sources],
        "treatments": [t.method for t in system.treatments],
        "target_gain": shao_wu.target_gain,
        "safety_factor": shao_wu.safety_factor,
        "domain_radius": next((t.domain_radius for t in system.treatments
                               if t.method == "jacobian"), shao_wu.domain_radius),
        "gains": gains,
        "scales": [float(s) for s in system.treatment_scales],
        "heaps": [None if t.method != "heaps" else {
            "support": t.heaps_support, "order": t.heaps_order, "period": t.heaps_period}
            for t in system.treatments],
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
        "self_only_treatment_probs": dict(config.self_only_treatment_probs),
    }


def stats() -> dict:
    config = MultiseriesConfig(ast_config=STATS_AST, channel_count_probs={STATS_CHANNELS: 1.},
                               dependency_probs={"random": 1.})
    training = TrainingFormat(context_length=64, prediction_length=16, stride=16, batch_size=32)
    stream = generate_training_batches(
        config=config, num_systems=STATS_SYSTEMS, trajectories_per_system=2, seed=STATS_SEED,
        trajectory=TrajectoryConfig(generated_steps=128, initialization=RandomHistorySpec()),
        training=training)
    start, shapes, seen, treatments = time.perf_counter(), None, set(), Counter()
    cross = 0
    for batch in stream:
        if shapes is None:
            shapes = {"context": list(batch.context.shape), "future": list(batch.future.shape),
                      "channel_mask": list(batch.channel_mask.shape),
                      "equation_targets": list(batch.equation_targets.shape),
                      "noise_std": list(batch.noise_targets["std"].shape)}
        for meta in batch.metadata:
            system = meta["system"]
            if system.seed in seen:
                continue
            seen.add(system.seed)
            for i, t in enumerate(system.treatments):
                treatments[t.method] += 1
                cross += any(j != i for j in system.effective_sources[i])
    seconds = time.perf_counter() - start
    s = stream.stats
    total = sum(treatments.values())
    return {
        "seed": STATS_SEED, "channels": STATS_CHANNELS, "systems": s.attempted_systems,
        "accepted_systems": len(seen),
        "treatment_shares": {k: v / total for k, v in treatments.items()},
        "cross_series_share": cross / total,
        "trajectories_attempted": s.attempted_trajectories,
        "initialization_attempts": s.initialization_attempts,
        "accepted_trajectories": s.accepted_trajectories,
        "replaced_systems": s.replaced_systems, "discarded_trajectories": s.discarded_trajectories,
        "resampled_equations": s.resampled_equations,
        "failures_by_reason": dict(Counter(f["reason"] for f in s.failures)),
        "failures_by_attempt": {str(k): v for k, v in sorted(Counter(f["attempt"] for f in s.failures).items())},
        "windows": s.yielded_windows, "seconds": round(seconds, 1), "batch_shapes": shapes,
    }


def main() -> None:
    out = {"demo": demo(), "stats": stats(), "python": sys.version.split()[0]}
    path = ROOT / "report" / "slides" / "data" / "multiseries_demo.js"
    path.write_text("window.MULTI_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n", encoding="utf-8")
    d, st = out["demo"], out["stats"]
    print("wrote", path.name, "seed", d["seed"], d["treatments"], d["allowed"], d["effective"],
          st["treatment_shares"], st["cross_series_share"], st["replaced_systems"], st["seconds"])


if __name__ == "__main__":
    main()
