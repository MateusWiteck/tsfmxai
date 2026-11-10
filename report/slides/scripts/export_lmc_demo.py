"""Export a real LMC-Synth example for the slides (TimePFN, Taga et al. 2025).

Follows synthetic_data_generation/LMC_Synth.py of github.com/egetaga/TimePFN, read on 9 Nov 2026:
  * latent functions: KernelSynth draws (1 to J=5 kernels from the 33-entry bank, combined left to
    right by + or x), raw Gaussian-process samples, no normalisation;
  * mixing: one Dirichlet(d * 1_L) row per channel, d drawn uniformly per series, so each channel is
    a convex combination of the latents;
  * output: weights @ latents, no noise term in the code (the paper's appendix mentions one).
The sizes here are illustrative (5 channels, 4 latents; the official defaults are 160 channels and
8 to 160 latents). The latents are drawn with the KernelSynth export's own bank and composition code
(scripts/export_kernelsynth_demo.py); seeds are fixed and the amplitudes are kept within a factor of
four of each other so the picture is readable, and the four demo latents are kept nearly uncorrelated; the official code does neither.

Run from the repository root:
    .venv/Scripts/python report/slides/scripts/export_lmc_demo.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_kernelsynth_demo import (  # noqa: E402
    KERNEL_BANK, LENGTH, compose, draw, expr_with_names, rounded, sample_composition, standardize,
)

OUT = Path(__file__).resolve().parents[1] / "data" / "lmc_demo.js"
THUMB_STEP = 4  # 1,024 points -> 256 for drawing
N_LATENTS, N_CHANNELS = 4, 5
CONCENTRATIONS = {"low": 0.15, "mid": 0.6, "high": 5.0}
SWEEP = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
POOL, TRIALS, SWEEP_L, SWEEP_C = 40, 100, 8, 16


def latent_candidates(first_seed, count, max_kernels=3, wiggle=0.3, max_corr=None):
    """Fixed-seed KernelSynth draws with a readable amplitude and a smooth enough path.

    max_corr: when given, a draw is kept only if its sample correlation with every kept draw is
    at most that, so the demo's latents are close to independent (the official code does not check)."""
    found, seed = [], first_seed
    while len(found) < count:
        seed += 1
        idx, ops, z = sample_composition(seed)
        if len(idx) > max_kernels:
            continue
        partial, _ = compose(idx, ops)
        x = draw(partial[-1], z)
        if not np.isfinite(x).all() or not 0.5 <= x.std() <= 2.0:
            continue
        if np.abs(np.diff(standardize(x))).mean() > wiggle:
            continue
        if max_corr is not None and any(abs(np.corrcoef(x, f["x"])[0, 1]) > max_corr for f in found):
            continue
        found.append({"seed": seed, "idx": idx, "ops": ops, "x": x})
    return found


def mix(latents, d, channels, seed):
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(d * np.ones(len(latents)), size=channels)
    return weights, weights @ latents


def corr_off_diagonal(ts):
    c = np.corrcoef(ts)
    return c, float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())


def main():
    cand = latent_candidates(100, N_LATENTS, max_corr=0.25)
    latents = np.array([c["x"] for c in cand])
    data = {
        "length": LENGTH, "thumb_step": THUMB_STEP,
        "latents": [{"label": expr_with_names(c["idx"], c["ops"]), "std": round(float(c["x"].std()), 3),
                     "series": rounded(c["x"][::THUMB_STEP])} for c in cand],
        "examples": {}, "sweep": [],
        "official": {"series": 15000, "channels": 160, "length": 1024, "max_kernels": 5,
                     "latents_clip": "[max(2, C // 20), C] = [8, 160]", "independent_share": 0.25,
                     "window": 192, "samples": 1.5e6},
    }
    for name, d in CONCENTRATIONS.items():
        weights, ts = mix(latents, d, N_CHANNELS, seed=11)
        corr, mean_abs = corr_off_diagonal(ts)
        data["examples"][name] = {
            "d": d, "weights": rounded(weights.ravel(), 3), "mean_abs_corr": round(mean_abs, 3),
            "corr": rounded(corr.ravel(), 2), "channels": [rounded(c[::THUMB_STEP]) for c in ts],
        }
    # concentration against channel correlation: a pool of latents, random subsets, fresh Dirichlet rows
    pool = np.array([c["x"] for c in latent_candidates(1000, POOL, max_kernels=5, wiggle=1.0)])
    rng = np.random.default_rng(5)
    for d in SWEEP:
        vals = []
        for _ in range(TRIALS):
            pick = pool[rng.choice(POOL, SWEEP_L, replace=False)]
            weights = rng.dirichlet(d * np.ones(SWEEP_L), size=SWEEP_C)
            vals.append(corr_off_diagonal(weights @ pick)[1])
        data["sweep"].append({"d": d, "mean_abs_corr": round(float(np.mean(vals)), 3)})
    OUT.write_text("window.LMC_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    print([l["label"] for l in data["latents"]])
    print({k: v["mean_abs_corr"] for k, v in data["examples"].items()}, data["sweep"])


if __name__ == "__main__":
    main()
