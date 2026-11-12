"""Time the iterative canonicalizer for the pipeline slides.

Read-only use of the project library: nothing outside report/slides/ is written.
Run from the repository root:

    .venv/Scripts/python -I report/slides/scripts/export_canonicalizer_timing.py

Every number on the "pipeline" slide that concerns the cost of canonicalize_polynomial
comes from this file: wall-clock time of one call (single thread, pure Python, exact
Fraction arithmetic), measured on trees from the same sampler, seeds and protection rule
as the 10-million-AST corpus, and a sweep over the number of operators.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tsfmxai.generators import CombinatorialTreeConfig, CombinatorialTreeSampler  # noqa: E402
from tsfmxai.representations import canonicalize_polynomial  # noqa: E402
from tsfmxai.symbolic_regression.canonical_data import derive_sample_seed, select_canonical_mechanism  # noqa: E402

CONFIG = json.loads((ROOT / "configs/canonicalization_5m_raw_tanh.json").read_text(encoding="utf-8"))
CORPUS_TREES = 100_000
SWEEP_TREES = 3_000
SWEEP_OPERATORS = (2, 4, 6, 8, 10, 12)


def measure(generator: dict, count: int, seed: int, probability: float) -> dict:
    sampler = CombinatorialTreeSampler(CombinatorialTreeConfig(**generator))
    times, terms, rejected = [], [], 0
    for index in range(count):
        sampled = sampler.sample(seed=derive_sample_seed(seed, index), sample_index=index)
        mechanism, _ = select_canonical_mechanism(
            sampled, probability=probability, root_seed=seed, sample_index=index)
        start = time.perf_counter()
        try:
            polynomial = canonicalize_polynomial(
                mechanism.expression, mechanism.binding, max_degree=generator["max_degree"])
            terms.append(len(polynomial.terms))
        except ValueError:
            rejected += 1
        times.append(time.perf_counter() - start)
    ms = np.array(times) * 1e3
    return {
        "trees": count, "rejected": rejected,
        "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
        "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()),
        "trees_per_second": float(1e3 / ms.mean()),
        "terms_mean": float(np.mean(terms)), "terms_max": int(max(terms)),
    }


def main() -> None:
    generator = dict(CONFIG["generator"])
    seed, probability = CONFIG["data_seed"], CONFIG["protection_probability"]
    corpus = measure(generator, CORPUS_TREES, seed, probability)
    sweep = []
    for operators in SWEEP_OPERATORS:
        row = measure(dict(generator, minimum_operators=operators, maximum_operators=operators),
                      SWEEP_TREES, seed, probability)
        sweep.append({"operators": operators, **row})
        print("operators", operators, "median %.3f ms" % row["median_ms"], "terms", round(row["terms_mean"], 1))
    data = {
        "config": CONFIG["experiment"], "generator": generator, "protection_probability": probability,
        "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "corpus": corpus, "sweep": sweep,
    }
    out = ROOT / "report/slides/data/canonicalizer_timing.js"
    out.write_text("window.CANON_TIMING = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("wrote", out.name, "corpus median %.3f ms, max %.1f ms, %d trees/s" % (
        corpus["median_ms"], corpus["max_ms"], corpus["trees_per_second"]))


if __name__ == "__main__":
    main()
