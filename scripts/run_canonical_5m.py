"""Generate a five-million-AST Parquet corpus, count supports, then train models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.symbolic_regression.canonical_large_scale import (  # noqa: E402
    count_corpus, deduplicate_evaluation_in_place, generate_parquet,
    make_unique_evaluation_corpus, train_parquet_variant,
)
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec  # noqa: E402
from tsfmxai.symbolic_regression.canonical_training import atomic_json, utc_now  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonicalization_5m_ast_parquet.json")
    parser.add_argument("--run-dir", default="outputs/canonicalization/ast_5m_20261030")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    run_dir = (ROOT / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = "cuda:0" if args.device == "cuda" else "cpu"
    if config["split"] != {"train": .7, "validation": .1, "test": .2}:
        raise ValueError("The split assignment in the generator must match the recorded proportions.")
    config_sha = sha256(config_path.read_bytes()).hexdigest()
    status_path = run_dir / "status.json"
    status_data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if status_data and status_data.get("config_sha256") != config_sha:
        raise ValueError("Run directory already belongs to a different config.")

    def status(state, **fields):
        status_data.update(fields)
        status_data.update({"state": state, "updated_at_utc": utc_now(),
                            "config_sha256": config_sha})
        atomic_json(status_path, status_data)

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    status("preparing_data", started_at_utc=status_data.get("started_at_utc", utc_now()),
           command=f"{sys.executable} scripts/run_canonical_5m.py --config {args.config} --run-dir {args.run_dir}")
    try:
        if config.get("source_run_dir"):
            make_unique_evaluation_corpus(ROOT / config["source_run_dir"], run_dir, status)
            g = config["generator"]
            codec = CanonicalCodec(g["num_series"], g["minimum_lag"],
                                   g["maximum_lag"], g["max_degree"])
        else:
            codec = generate_parquet(config, run_dir, status)
            if config.get("unique_eval_skeletons"):
                deduplicate_evaluation_in_place(run_dir, status)
        status("counting")
        corpus = count_corpus(config, run_dir, status)
        status("dataset_ready", corpus=corpus)
        print("DATASET", json.dumps(corpus, ensure_ascii=False), flush=True)
        results = status_data.get("results", {})
        for variant in config["variants"]:
            name = variant["name"]
            if name in results:
                continue
            result = train_parquet_variant(config, run_dir, codec, variant, status,
                                           device=device)
            results[name] = result
            status("variant_completed", results=results)
            print("MODEL", name, json.dumps({k: result[k] for k in
                ("count", "skeleton_exact", "term_f1", "joint_exact", "best_epoch")}), flush=True)
        source = ["src/tsfmxai/symbolic_regression/canonical_large_scale.py",
                  "src/tsfmxai/symbolic_regression/canonical_models.py",
                  "src/tsfmxai/generators/combinatorial_expression.py",
                  "src/tsfmxai/generators/shao_wu.py",
                  "src/tsfmxai/representations/canonical_polynomial.py"]
        provenance = {"config": config, "config_sha256": config_sha,
                      "source_sha256": {f: sha256((ROOT / f).read_bytes()).hexdigest() for f in source},
                      "python": platform.python_version(), "torch": torch.__version__,
                      "device": device, "cpu_threads": config["cpu_threads"],
                      "generated_at_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started}
        atomic_json(run_dir / "manifest.json", {"provenance": provenance,
                    "corpus": corpus, "results": results,
                    "scientific_summary": "The split uses SHA-256 of the canonical skeleton; "+
                    "the exact support count refers to sampled ASTs, including duplicate supports."})
        status("completed", results=results, elapsed_seconds=time.perf_counter() - started)
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
