"""Same cosine start, four stability methods (button on the stability-method overlay).

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_cosine_start.py

It reproduces one condition of
experiments/time_series_initialization/2_initial_magnitude.ipynb, section
"Cosseno completo; A=0.5": the cyclic bank (12 mechanisms per method, lags 12 to 24,
Heaps with fixed order 24), a full cosine cycle of period 96 and amplitude 0.5 as the
starting history, and the notebook's own paired noise streams. The first three
mechanisms and their three trajectories are exported, as in the notebook figure.

For Heaps it also records the slowest companion roots of each exported mechanism and
one run of mechanism 1 from a zero history with the same noise, to show whether the
wave comes from the history or from the mechanism itself.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.evaluation.initialization import (  # noqa: E402
    GROUPS, HistorySpec, InitializationStudyConfig, run_history_case, sample_study_mechanisms,
)
from tsfmxai.generators.heaps import var_companion_matrix  # noqa: E402

EXAMPLES = 3
AMPLITUDE, PERIOD = 0.5, 96


def rounded(a):
    return [round(float(v), 4) for v in np.asarray(a).ravel()]


def slow_roots(mechanism, count=3):
    roots = np.linalg.eigvals(var_companion_matrix(np.asarray(mechanism.coefficients)))
    roots = [z for z in roots if z.imag >= 0]
    out = []
    for z in roots:
        angle = abs(np.angle(z))
        out.append({"modulus": round(float(abs(z)), 4),
                    "period": None if angle < 1e-12 else round(float(2 * np.pi / angle), 1)})
    # longest periods first among the near-unit roots (these dominate long runs)
    near = [r for r in out if r["modulus"] > 0.99]
    near.sort(key=lambda r: -(r["period"] or 1e9))
    return near[:count]


def main() -> None:
    base = InitializationStudyConfig(mechanisms=32, trajectories=3, example_mechanisms=EXAMPLES)
    config = replace(base, mechanisms=12, minimum_lag=12, maximum_lag=24, heaps_fixed_order=True)
    block = f"Cosseno completo; A={AMPLITUDE}"
    spec = HistorySpec(block, "cosine", AMPLITUDE, 0.0, PERIOD, 1.0)
    methods = {}
    for group in GROUPS:
        mechanisms = sample_study_mechanisms(group, config)
        result = run_history_case(group, mechanisms, config, spec)
        examples = {}
        for e in result["examples"]:
            m = examples.setdefault(e["mechanism"], {"order": int(e["actual_order"]), "history": rounded(e["history"][0]),
                                                     "runs": []})
            m["runs"].append(rounded(e["path"][0][-config.horizons[-1]:]))
        final = [r for r in result["rows"] if r["horizon"] == max(config.horizons)]
        methods[group] = {
            "examples": [examples[i] for i in sorted(examples)],
            "explosions": sum(bool(r["exploded"]) for r in final), "trajectories": len(final),
        }
        if group == "Heaps":
            methods[group]["roots"] = [slow_roots(m) for m in mechanisms[:EXAMPLES]]
            zero = run_history_case(group, mechanisms, config, HistorySpec("zero", "zero"))
            methods[group]["zero_start"] = rounded(
                next(e for e in zero["examples"] if e["mechanism"] == 1 and e["trajectory"] == 0)["path"][0][-config.horizons[-1]:])
        print(group, "orders", [m["order"] for m in methods[group]["examples"]],
              "explosions", methods[group]["explosions"], "/", methods[group]["trajectories"], flush=True)
    print("Heaps slow roots", methods["Heaps"]["roots"])
    payload = {"amplitude": AMPLITUDE, "period": PERIOD, "steps": max(config.horizons),
               "innovation_std": config.innovation_std, "mechanisms": config.mechanisms, "methods": methods}
    out = ROOT / "report" / "slides" / "data" / "cosine_start.js"
    out.write_text("window.COSINE_START = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out.name, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
