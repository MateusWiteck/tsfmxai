"""Export two extra real mechanisms for the multiseries step of the "pipeline" slide.

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_pipeline_demo.py

Each lane is what the generator does once: a sampled tree, the same tree after the
product-lag tanh protection, and one trajectory of the certified Shao-Wu mechanism built
from it (the first `history_len` values are the initial history). The first lane of the slide is the example of export_generator_demo.py; these two
show that the same steps, repeated, give more series.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report" / "slides" / "scripts"))

import numpy as np  # noqa: E402

from export_generator_demo import CONFIG, DEMO_INDEX, DEMO_SEED, TRAJ_STEPS, tree_json  # noqa: E402
from tsfmxai.generators import CombinatorialTreeConfig, CombinatorialTreeSampler  # noqa: E402
from tsfmxai.generators.shao_wu import (  # noqa: E402
    ShaoWuConfig,
    build_shao_wu_mechanism,
    sample_shao_wu_trajectories,
    saturate_product_lags,
)
from tsfmxai.generators.trajectory import (  # noqa: E402
    DivergentTrajectoryError,
    required_history_length,
    sample_univariate_trajectories,
)
from tsfmxai.representations.expression import operator_count  # noqa: E402

SIM_STEPS = 120
INNOVATION_STD = 0.05  # as in the four-family stability study
FLAT = 0.15  # 3 sigma: the band of "near zero"
OPERATORS = (6, 5)  # one lane per entry; most sampled trees have 6 operators


def simulate_raw(sampled, root_seed: int = 7):
    """Simulate the raw (unfixed) recurrence; stop where the value stops being finite.

    Returns (trajectory including the initial history, history length, diverged).
    """
    history = required_history_length(sampled.expression, sampled.binding)
    for steps in range(SIM_STEPS, 0, -1):
        try:
            with np.errstate(all="ignore"):
                traj = sample_univariate_trajectories(
                    sampled.expression, sampled.binding, trajectory_count=1, generated_steps=steps,
                    innovation_std=INNOVATION_STD, root_seed=root_seed, maximum_absolute_value=None)[0]
            return traj, history, steps < SIM_STEPS
        except DivergentTrajectoryError:
            continue
    raise RuntimeError("the history itself is not finite")


def classify(traj, history: int, diverged: bool) -> str:
    """exploding: |x| > 100 or not finite; fading: the last 48 values sit in [-0.15, 0.15]."""
    if diverged or np.abs(traj).max() > 100:
        return "exploding"
    gen = traj[history:]
    if np.all(np.abs(gen[-48:]) <= FLAT) and np.abs(gen[:24]).max() > FLAT:
        return "fading"
    return "stable"


def raw_example(kind: str, sampled, traj, history: int) -> dict:
    return {
        "kind": kind, "history_len": int(history), "diverged_at": int(len(traj)) if kind == "exploding" and len(traj) < history + SIM_STEPS else None,
        "raw_tree": tree_json(sampled.expression, sampled.binding),
        # asinh keeps a blown-up trajectory and a calm one on one axis
        "asinh_trajectory": [float(np.arcsinh(v)) for v in traj],
    }


def main() -> None:
    sampler = CombinatorialTreeSampler(CombinatorialTreeConfig(**CONFIG))
    lanes = []
    used = set()
    for wanted in OPERATORS:
        for index in range(5000):
            sampled = sampler.sample(seed=5000 + index, sample_index=index)
            if operator_count(sampled.expression) != wanted or index in used:
                continue
            protected = saturate_product_lags(sampled.expression)
            mech = build_shao_wu_mechanism((protected,), (sampled.binding,), ShaoWuConfig(), seed=5000 + index, sample_index=index)
            traj = sample_shao_wu_trajectories(mech, trajectory_count=1, generated_steps=TRAJ_STEPS, root_seed=7)[0, 0]
            if not all(abs(float(v)) < 50 for v in traj):
                continue
            used.add(index)
            lanes.append({
                "seed": 5000 + index, "index": index, "operators": wanted,
                "raw_tree": tree_json(sampled.expression, sampled.binding),
                "protected_tree": tree_json(protected, sampled.binding),
                "trajectory": [float(v) for v in traj],
                "history_len": int(mech.order),  # the first values of the trajectory are the initial history
            })
            break
        else:
            raise RuntimeError(f"no tree with {wanted} operators found")
    # Raw (unfixed) trees: the slide's own example, then one fading and one stable tree.
    demo = CombinatorialTreeSampler(CombinatorialTreeConfig(**{**CONFIG, "maximum_lag": 12})).sample(
        seed=DEMO_SEED, sample_index=DEMO_INDEX)
    traj, history, diverged = simulate_raw(demo)
    raw = [raw_example(classify(traj, history, diverged), demo, traj, history)]
    assert raw[0]["kind"] == "exploding", "the slide example is expected to blow up when it is not fixed"
    for wanted in ("fading", "stable"):
        for index in range(3000):
            sampled = sampler.sample(seed=9000 + index, sample_index=index)
            if operator_count(sampled.expression) != 6:
                continue
            traj, history, diverged = simulate_raw(sampled)
            if classify(traj, history, diverged) != wanted:
                continue
            if wanted == "stable" and np.abs(traj[history:]).max() < 0.5:
                continue  # a stable tree that is also flat teaches nothing
            raw.append(raw_example(wanted, sampled, traj, history))
            break
        else:
            raise RuntimeError(f"no {wanted} raw tree found")
    out = ROOT / "report" / "slides" / "data" / "pipeline_demo.js"
    out.write_text("window.PIPE_DATA = " + json.dumps({"lanes": lanes, "raw": raw}, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out.name, [(l["operators"], l["seed"]) for l in lanes], [(r["kind"], len(r["asinh_trajectory"])) for r in raw])


if __name__ == "__main__":
    main()
