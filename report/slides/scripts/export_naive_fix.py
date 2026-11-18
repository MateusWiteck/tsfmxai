"""Does shrinking the whole equation make plain random trees stable? (pipeline slide)

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_naive_fix.py

It reuses the protocol of experiments/abstract_syntax_tree/6_stability_methods_tradeoff.ipynb
(the same 5,000 "original" trees, seed, noise, 384 steps, explosion = |x| > 100 or not
finite, near zero = the same tail classifier) and only multiplies each equation by a
constant s: x(t) = s * f(x(t-1), ..., x(t-L)) + e(t). The row s = 1 must reproduce the
notebook's plain-tree shares, which the script asserts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.evaluation import StabilityDiagnosticConfig, TailDiagnosticConfig  # noqa: E402
from tsfmxai.evaluation.generator_tradeoff import benchmark_stability_method  # noqa: E402
from tsfmxai.generators import RandomTreeConfig, sample_random_mechanisms  # noqa: E402
from tsfmxai.representations.expression import MechanismBinding, Multiply, Parameter  # noqa: E402

ROOT_SEED, SAMPLE_COUNT, INNOVATION_STD = 20261001, 5000, 0.05
HORIZONS = (96, 192, 384)
SCALES = (1.0, 0.5, 0.25, 0.1)
BASE_CONFIG = RandomTreeConfig(
    minimum_operators=1, maximum_operators=6, num_series=1, minimum_lag=1, maximum_lag=16,
    parameter_probability=0.55, lag_probability=0.45, add_probability=0.80, require_lag=True,
    minimum_parameter_magnitude=0.05, maximum_parameter_magnitude=0.80,
)


def scaled(mechanism, scale: float):
    if scale == 1.0:
        return mechanism
    values = dict(mechanism.binding.parameter_values)
    values["SCALE"] = scale
    binding = MechanismBinding(values, dict(mechanism.binding.lag_values))
    return replace(mechanism, expression=Multiply(Parameter("SCALE"), mechanism.expression), binding=binding)


def main() -> None:
    diagnostic = StabilityDiagnosticConfig(HORIZONS, 100.0, 3 * INNOVATION_STD, 24)
    tail = TailDiagnosticConfig(window_steps=24, persistence_windows=2,
                                flat_tolerance=3 * INNOVATION_STD, decline_ratio=0.5)
    base = sample_random_mechanisms(BASE_CONFIG, root_seed=ROOT_SEED, count=SAMPLE_COUNT)
    rows = []
    for scale in SCALES:
        result = benchmark_stability_method(
            "original", [scaled(m, scale) for m in base], root_seed=ROOT_SEED,
            diagnostic_config=diagnostic, innovation_std=INNOVATION_STD, tail_config=tail)
        last = [o for o in result["outcomes"] if o["horizon"] == max(HORIZONS)]
        n = len(last)
        explode = sum(bool(o["exploded"]) for o in last) / n
        near = sum(bool(o["near_zero"]) for o in last) / n
        rows.append({"scale": scale, "explode": explode, "near_zero": near, "neither": 1 - explode - near})
        print(f"s={scale}: explode {100 * explode:.2f}%  near zero {100 * near:.2f}%  neither {100 * (1 - explode - near):.2f}%", flush=True)
    assert abs(rows[0]["explode"] - 0.5224) < 1e-4 and abs(rows[0]["near_zero"] - 0.1036) < 1e-4, "s = 1 must reproduce the notebook"
    out = ROOT / "report" / "slides" / "data" / "naive_fix.js"
    out.write_text("window.NAIVE_FIX = " + json.dumps({"trees": SAMPLE_COUNT, "steps": max(HORIZONS), "rows": rows},
                                                      separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
