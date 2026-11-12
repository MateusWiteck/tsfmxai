"""Export the autoregressive-coverage evidence used by the slide "What lagged recurrences can do".

Reproduces, with the experiment's own code, the executed notebooks
  experiments/autoregressive_process_coverage/AR/01_kernelsynth_ar_coverage.ipynb   (53 KernelSynth functions)
  experiments/autoregressive_process_coverage/AR/02_timepfn_lmc_ar_var_coverage.ipynb (33 TimePFN latent series)
by executing the same notebook cells (configuration, generators, AR fit). It follows the corrected
protocol appended to both notebooks ("Corrections after review"): 16 validation trajectories drawn
after the test set, an order grid that contains 16 (the generator's lag cap), and the order selected
on validation as the smallest one whose validation NMSE is <= 0.001. A function counts as reproduced
with at most k lags when its selected order is <= k and its test NMSE is <= 0.001.

Run from the repository root:
    .venv/Scripts/python report/slides/scripts/export_ar_coverage_demo.py
"""

import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data" / "ar_coverage_demo.js"
EXPERIMENT = ROOT / "experiments" / "autoregressive_process_coverage" / "AR"
THRESHOLD = 1e-3
N_VALIDATION = 16
KS_ORDERS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256]
TP_ORDERS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128]
SHOWN = 200  # points drawn per example
EXAMPLES = ["periodic_01_p24", "rbf_scale0.1", "composite_01", "composite_05", "composite_10", "white_level1"]


def load(notebook, name):
    """Execute cells 1 and 3 of a source notebook inside a registered module namespace."""
    nb = json.loads((EXPERIMENT / notebook).read_text(encoding="utf-8"))
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    g = mod.__dict__
    g["__name__"] = name
    g["display"] = lambda *a, **k: None
    for index in (1, 3):
        exec(compile("".join(nb["cells"][index]["source"]), f"{notebook}#cell-{index}", "exec"), g)
    return g


def nmse_at(g, train, test, order):
    fit = g["fit_ar"](train, order)
    _, target = g["lagged_design"](test, order)
    prediction = g["ar_predictions"](fit, test)
    return float(np.mean((target - prediction) ** 2) / (np.var(target) + 1e-12)), fit


def select(scores):
    """Smallest order whose validation NMSE passes, else the lowest validation NMSE."""
    ordered = sorted(scores, key=lambda s: s["order"])
    passing = [s for s in ordered if s["validation"] <= THRESHOLD]
    return passing[0] if passing else min(ordered, key=lambda s: (s["validation"], s["order"]))


def cumulative(chosen, orders):
    return [[o, sum(1 for c in chosen if c["order"] <= o and c["test"] <= THRESHOLD)] for o in orders]


def kernelsynth():
    g = load("01_kernelsynth_ar_coverage.ipynb", "ks_cov")
    chosen, best_of_test, examples = [], 0, {}
    for index, f in enumerate(g["FUNCTIONS"]):
        factor = g["stable_factor"](f["covariance"])
        rng = np.random.default_rng(g["GLOBAL_SEED"] + 100_000 + index)
        raw_train = g["draw_from_factor"](factor, g["N_TRAIN"], rng)
        raw_test = g["draw_from_factor"](factor, g["N_TEST"], rng)
        raw_validation = g["draw_from_factor"](factor, N_VALIDATION, rng)
        mean, scale = float(np.mean(raw_train)), float(np.std(raw_train) + 1e-12)
        train, test, validation = [(x - mean) / scale for x in (raw_train, raw_test, raw_validation)]
        scores = [{"order": o, "validation": nmse_at(g, train, validation, o)[0], "test": nmse_at(g, train, test, o)[0]} for o in KS_ORDERS]
        best_of_test += min(s["test"] for s in scores) <= THRESHOLD
        pick = select(scores)
        chosen.append(pick)
        if f["id"] in EXAMPLES:
            order = pick["order"]
            score, fit = nmse_at(g, train, test, order)
            prediction = g["ar_predictions"](fit, test[:1])
            examples[f["id"]] = {
                "id": f["id"], "family": f["family"], "order": order, "covered": pick["test"] <= THRESHOLD, "nmse": score,
                "intercept": float(fit.intercept), "coefficients": [float(c) for c in fit.coefficients[: min(order, 4)]],
                "actual": [round(float(v), 4) for v in test[0, order:][:SHOWN]],
                "prediction": [round(float(v), 4) for v in prediction[:SHOWN]],
            }
    return {
        "total": len(chosen), "best_of_grid": int(best_of_test), "orders": KS_ORDERS,
        "cumulative": cumulative(chosen, KS_ORDERS), "never": sum(c["test"] > THRESHOLD for c in chosen),
        "examples": [examples[i] for i in EXAMPLES],
    }


def timepfn():
    g = load("02_timepfn_lmc_ar_var_coverage.ipynb", "tp_cov")
    chosen, best_of_test = [], 0
    for index, mechanism in enumerate(g["LMC_MECHANISMS"]):
        rng = np.random.default_rng(g["GLOBAL_SEED"] + 200_000 + index)
        draw = lambda count: np.stack([g["draw_from_factor"](g["latent_factors"][i], count, rng) for i in mechanism["latent_indices"]], axis=1)
        raw_train, raw_test, raw_validation = draw(g["N_TRAIN"]), draw(g["N_TEST"]), draw(N_VALIDATION)
        mean = raw_train.mean(axis=(0, 2))
        scale = raw_train.std(axis=(0, 2)) + 1e-12
        norm = lambda x: (x - mean[None, :, None]) / scale[None, :, None]
        train, test, validation = norm(raw_train), norm(raw_test), norm(raw_validation)
        for latent in range(3):
            scores = [{"order": o, "validation": nmse_at(g, train[:, latent, :], validation[:, latent, :], o)[0],
                       "test": nmse_at(g, train[:, latent, :], test[:, latent, :], o)[0]} for o in TP_ORDERS]
            best_of_test += min(s["test"] for s in scores) <= THRESHOLD
            chosen.append(select(scores))
    return {"total": len(chosen), "best_of_grid": int(best_of_test), "orders": TP_ORDERS,
            "cumulative": cumulative(chosen, TP_ORDERS), "never": sum(c["test"] > THRESHOLD for c in chosen)}


def main():
    data = {"threshold": THRESHOLD, "kernelsynth": kernelsynth(), "timepfn": timepfn()}
    OUT.write_text("window.AR_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    for name in ("kernelsynth", "timepfn"):
        d = data[name]
        print(name, "best-of-grid on test", d["best_of_grid"], "/", d["total"], "cumulative", d["cumulative"], "not reproduced", d["never"])


if __name__ == "__main__":
    main()
