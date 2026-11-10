"""Export KernelSynth samples used by the slide animations.

The kernel bank and the composition procedure (random number of kernels,
left-to-right reduction with a random ``+``/``*``) mirror the official script
https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/kernel-synth.py
(Apache-2.0). Only the GP draw differs: the official script calls
``GaussianProcessRegressor.sample_y`` (SVD); here a draw is ``L z`` with an
eigendecomposition and an explicit noise vector ``z``, so the partial kernels
of one composition can share the same ``z`` and morph coherently.

Run from the repository root:
    .venv/Scripts/python report/slides/scripts/export_kernelsynth_demo.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
)

LENGTH = 1024
THUMB = 256
COV_GRID = 64
DEMO_SEED = 7
OUT = Path(__file__).resolve().parents[1] / "data" / "kernelsynth_demo.js"

# (family, label, kernel) in the order of the official KERNEL_BANK.
PERIODS = [24, 48, 96, 168, 336, 672, 7, 14, 30, 60, 365, 730, 4, 26, 52, 4, 6, 12, 4, 40, 10]
KERNEL_BANK = (
    [("periodic", f"Periodic p={p}", ExpSineSquared(periodicity=p / LENGTH)) for p in PERIODS]
    + [("linear", f"Linear σ₀={s:g}", DotProduct(sigma_0=s)) for s in (0.0, 1.0, 10.0)]
    + [("rbf", f"RBF ℓ={s:g}", RBF(length_scale=s)) for s in (0.1, 1.0, 10.0)]
    + [("rq", f"RatQuad α={a:g}", RationalQuadratic(alpha=a)) for a in (0.1, 1.0, 10.0)]
    + [("white", f"White σ²={n:g}", WhiteKernel(noise_level=n)) for n in (0.1, 1.0)]
    + [("constant", "Constant C=1", ConstantKernel())]
)
assert len(KERNEL_BANK) == 33

X = np.linspace(0, 1, LENGTH)[:, None]


def draw(kernel, z):
    """Sample from GP(0, kernel) at X using the fixed standard-normal vector z."""
    w, v = np.linalg.eigh(kernel(X))
    return v @ (np.sqrt(np.clip(w, 0, None)) * z)


def rounded(a, n=3):
    return [round(float(x), n) for x in a]


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def compose(rng_choice_idx, ops):
    """Return the expression text and the list of partial kernels (left fold)."""
    kernels = [KERNEL_BANK[i][2] for i in rng_choice_idx]
    partial, expr = [kernels[0]], [f"κ{1}"]
    for j, op in enumerate(ops, start=1):
        k = partial[-1] + kernels[j] if op == "+" else partial[-1] * kernels[j]
        partial.append(k)
        expr.append(f"({expr[-1]} {op} κ{j + 1})" if j < len(ops) else f"{expr[-1]} {op} κ{j + 1}")
    return partial, expr


def sample_composition(seed, max_kernels=5):
    """Mimic generate_time_series: R ~ U{1..J}, kernels with replacement, random ops."""
    np.random.seed(seed)
    r = int(np.random.randint(1, max_kernels + 1))
    idx = [int(i) for i in np.random.choice(len(KERNEL_BANK), r, replace=True)]
    ops = ["+" if np.random.randint(2) == 0 else "×" for _ in range(r - 1)]
    z = np.random.default_rng(seed).standard_normal(LENGTH)
    return idx, ops, z


def nice(seed):
    idx, ops, z = sample_composition(seed)
    partial, _ = compose(idx, ops)
    x = draw(partial[-1], z)
    ok = np.isfinite(x).all() and x.std() > 1e-6
    return ok, idx, ops, z, partial, x


def spikiness(x):
    d = np.abs(np.diff(standardize(x)))
    return d.mean()


def find_demo():
    """First seed from DEMO_SEED on with R=4, four kernel families, mixed operators,
    a short-period periodic kernel (visible wiggles) and a readable curve."""
    short = {26, 30, 40, 52, 60}
    for seed in range(DEMO_SEED, DEMO_SEED + 20000):
        idx, ops, z = sample_composition(seed)
        if len(idx) != 4 or len(set(ops)) < 2:
            continue
        fams = [KERNEL_BANK[i][0] for i in idx]
        if len(set(fams)) < 4 or "white" in fams or "constant" in fams:
            continue
        if not any(i < len(PERIODS) and PERIODS[i] in short for i in idx):
            continue
        ok, idx, ops, z, partial, x = nice(seed)
        if ok and spikiness(x) <= 0.25:
            return seed
    raise RuntimeError("no demo seed")


def expr_with_names(idx, ops):
    parts = [KERNEL_BANK[i][1] for i in idx]
    return " ".join(sum(([p, o] for p, o in zip(parts, ops + [""])), [])).strip()


LAG = 80
WIN0 = 2 * LAG  # first index where every lagged term of the sequential example exists
OUT_MV = Path(__file__).resolve().parents[1] / "data" / "multivariate_demo.js"


def fold_names(idx, ops):
    fam = {"periodic": "Periodic", "linear": "Linear", "rbf": "RBF", "rq": "RatQuad"}
    s = fam[KERNEL_BANK[idx[0]][0]]
    for j in range(1, len(idx)):
        s = (f"({s})" if j > 1 else s) + f" {ops[j - 1]} " + fam[KERNEL_BANK[idx[j]][0]]
    return s


def corr_matrix(ys):
    return [rounded(r, 2) for r in np.corrcoef(np.array(ys))]


def xcorr(a, b, max_lag=200, step=2):
    """corr(a[t], b[t + k]) for k in [-max_lag, max_lag]."""
    lags = list(range(-max_lag, max_lag + 1, step))
    out = []
    for k in lags:
        u, v = (a[: len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[: len(b) + k])
        out.append(float(np.corrcoef(u, v)[0, 1]))
    return lags, out


def export_multivariate():
    """Three independent KernelSynth draws plus two *illustrative* multivariatizers.

    Chronos-2 only states that cotemporaneous and sequential multivariatizers exist
    (Ansari et al., 2025, Sec. 4.2); the transformations below are didactic examples.
    """
    # Series 1 and 2 are non-periodic (RBF l=0.1 plus others): the cross-correlation of
    # two periodic series is itself periodic, which would hide a single lead-lag peak.
    smooth, periodic = [], []
    for seed in range(5000, 12000):
        idx, ops, z = sample_composition(seed)
        fams = {KERNEL_BANK[i][0] for i in idx}
        if len(idx) not in (2, 3) or fams & {"white", "constant"}:
            continue
        is_periodic = "periodic" in fams
        if is_periodic:
            wanted = len(periodic) < 15
        else:
            wanted = 24 in idx and fams <= {"rbf", "rq"} and len(smooth) < 25
        if not wanted:
            continue
        ok, idx, ops, z, partial, x = nice(seed)
        if not ok or spikiness(x) > 0.2:
            continue
        (periodic if is_periodic else smooth).append((seed, idx, ops, standardize(x)))
        if len(smooth) >= 25 and len(periodic) >= 15:
            break
    def lagged(v, k):  # v[t - k]
        out = np.zeros_like(v)
        out[k:] = v[:-k]
        return out

    # Prefer a pair whose independent cross-correlation is small and whose sequential
    # cross-correlation peaks exactly at the imposed lag, so the plot reads cleanly.
    best, best_c = None, 9.0
    for a in range(len(smooth)):
        for b in range(len(smooth)):
            if a == b:
                continue
            x1, x2 = smooth[a][3], smooth[b][3]
            lags, xb = xcorr(x1[WIN0:], x2[WIN0:])
            y2 = 0.4 * x2 + 0.9 * lagged(x1, LAG)
            _, xs = xcorr(x1[WIN0:], y2[WIN0:])
            peak = lags[int(np.argmax(xs))]
            pair_score = np.abs(xb).max() + 0.5 * abs(peak - LAG) / 10
            for c in range(len(periodic)):
                w = [t[3][WIN0:] for t in (smooth[a], smooth[b], periodic[c])]
                m = np.abs(np.corrcoef(np.array(w)) - np.eye(3)).max()
                score = pair_score + m
                if score < best_c:
                    best, best_c = (smooth[a], smooth[b], periodic[c]), score
    picks = list(best)
    x = [p[3] for p in picks]

    def shift(v, k):  # v[t - k]
        out = np.zeros_like(v)
        out[k:] = v[:-k]
        return out

    cot = [x[0], 0.8 * x[0] + 0.6 * x[1], np.tanh(x[2] + x[0])]
    y2 = 0.4 * x[1] + 0.9 * shift(x[0], LAG)
    y3 = 0.4 * x[2] + 0.9 * shift(y2, LAG)
    seq = [x[0], y2, y3]
    win = lambda arrs: [a[WIN0:] for a in arrs]
    base_w, cot_w, seq_w = win(x), win(cot), win(seq)
    # Terms that add up to each variate, so the slide can draw "what y is made of".
    cot_parts = win([0.8 * x[0], 0.6 * x[1], x[2] + x[0]])
    seq_parts = win([0.9 * shift(x[0], LAG), 0.4 * x[1], 0.9 * shift(y2, LAG), 0.4 * x[2]])
    lags, xc_base = xcorr(base_w[0], base_w[1])
    _, xc_seq = xcorr(seq_w[0], seq_w[1])
    payload = {
        "n": len(base_w[0]),
        "lag": LAG,
        "seeds": [p[0] for p in picks],
        "names": [fold_names(p[1], p[2]) for p in picks],
        "base": [rounded(a) for a in base_w],
        "cot": [rounded(a) for a in cot_w],
        "cot_parts": [rounded(a) for a in cot_parts],
        "seq_parts": [rounded(a) for a in seq_parts],
        "seq": [rounded(a) for a in seq_w],
        "corr": {"base": corr_matrix(base_w), "cot": corr_matrix(cot_w), "seq": corr_matrix(seq_w)},
        "xc": {"lags": lags, "base": rounded(xc_base, 3), "seq": rounded(xc_seq, 3)},
    }
    OUT_MV.write_text("window.MV_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("multivariate seeds", payload["seeds"], "max |corr| base", round(best_c, 2), "->", OUT_MV.name)


def main():
    thumb_rng = np.random.default_rng(0)
    bank = []
    for i, (fam, label, k) in enumerate(KERNEL_BANK):
        z = thumb_rng.standard_normal(LENGTH)
        s = standardize(draw(k, z)[:THUMB])
        bank.append({"id": i, "family": fam, "label": label, "thumb": rounded(s)})

    seed = find_demo()
    ok, idx, ops, z, partial, x = nice(seed)
    stages = []
    for j, k in enumerate(partial):
        s = standardize(draw(k, z))
        stages.append({"series": rounded(s)})
    K = partial[-1](X)
    step = LENGTH // COV_GRID
    cov = K[::step, ::step]
    cov = cov / np.abs(cov).max()
    demo = {
        "seed": seed,
        "R": len(idx),
        "picks": idx,
        "ops": ops,
        "stages": stages,
        "cov": [rounded(r, 3) for r in cov],
    }

    # Additional independent compositions for the "again" step.
    more = []
    s = seed + 1
    while len(more) < 6:
        s += 1
        ok, idx2, ops2, z2, partial2, x2 = nice(s)
        if not ok or len(idx2) < 2 or spikiness(x2) > 0.35:
            continue
        more.append(
            {
                "seed": s,
                "picks": idx2,
                "ops": ops2,
                "series": rounded(standardize(x2)),
            }
        )

    # Three kernels with their covariance heatmaps and two draws, for the GP slide.
    gp_examples = []
    for name, k in [
        ("RBF  ℓ=0.1", RBF(length_scale=0.1)),
        ("Periodic  p=1/8", ExpSineSquared(periodicity=0.125)),
        ("Linear  σ₀=1", DotProduct(sigma_0=1.0)),
    ]:
        Kf = k(X)
        stp = LENGTH // 64
        Kc = Kf[::stp, ::stp]
        Kc = Kc / np.abs(Kc).max()
        zz = np.random.default_rng(11)
        gp_examples.append(
            {
                "name": name,
                "cov": [rounded(r, 3) for r in Kc],
                "draws": [rounded(standardize(draw(k, zz.standard_normal(LENGTH)))) for _ in range(2)],
            }
        )

    payload = {"length": LENGTH, "bank": bank, "demo": demo, "more": more, "gp": gp_examples}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.KS_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print("seed", seed, "R", demo["R"], "picks", demo["picks"], demo["ops"])
    print("wrote", OUT, OUT.stat().st_size // 1024, "KiB")
    export_multivariate()


if __name__ == "__main__":
    main()
