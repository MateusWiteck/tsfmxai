"""Generate the configured multiseries AST Parquet corpus, count supports, then train models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version
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


class ProgressLog:
    """One readable line per state change, and at most one per interval while a state lasts."""

    def __init__(self, epochs, interval=60.0):
        self.epochs, self.interval, self.last, self.state = epochs, interval, 0.0, None
        self.epoch_key, self.epoch_start, self.printed_epoch = None, None, None

    def __call__(self, data, state):
        now = time.time()
        changed = state != self.state
        self.state = state
        line, always = None, changed
        if state == "generating":
            done, total = data.get("generated", 0), data.get("total") or 1
            elapsed = data.get("generation_elapsed_seconds") or 0
            if done and elapsed:
                eta = elapsed * (total - done) / done
                line = (f"generating {done:,}/{total:,} ASTs ({100 * done / total:.1f}%), "
                        f"ETA {eta / 60:.0f} min")
        elif state == "training":
            variant, epoch = data.get("active_variant"), data.get("epoch")
            seen = data.get("training_rows_seen", 0)
            if (variant, epoch) != self.epoch_key or seen == 0:
                self.epoch_key, self.epoch_start = (variant, epoch), now
            rows = ((data.get("corpus") or {}).get("rows_by_split") or {}).get("train")
            last = data.get("last_epoch")
            if last and last.get("epoch") != self.printed_epoch:
                self.printed_epoch, always = last["epoch"], True
                line = (f"{variant} epoch {last['epoch']} done in {last['seconds'] / 60:.1f} min: "
                        f"train loss {last['train_loss']:.4f}, validation loss {last['validation_loss']:.4f}, "
                        f"skeleton exact {last['validation_skeleton_exact']:.3f}")
            elif seen and rows and now > self.epoch_start:
                rate = seen / (now - self.epoch_start)
                line = (f"{variant} epoch {epoch}/{self.epochs}: {seen:,}/{rows:,} rows "
                        f"({100 * seen / rows:.1f}%), {rate:,.0f} rows/s, "
                        f"epoch ETA {(rows - seen) / rate / 60:.0f} min")
            elif seen == 0:
                line, always = f"{variant} epoch {epoch}/{self.epochs} started", True
        elif state == "failed":
            line, always = f"FAILED: {data.get('error')}", True
        elif changed:
            line = state.replace("_", " ")
        if line and (always or now - self.last >= self.interval):
            self.last = now
            print(time.strftime("[%H:%M:%S] ") + line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonicalization_six_series_mixed.json")
    parser.add_argument("--run-dir", default="outputs/canonical_six_series_mixed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--release-dir", default=None,
                        help="after a completed run, also write the Hugging Face release here")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="smoke tests only: run although code or configs differ from HEAD")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    run_dir = (ROOT / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # The release exports source/ from the training commit, so the code that runs
    # must be exactly that commit.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs",
                            "pyproject.toml"], cwd=ROOT, text=True, capture_output=True)
    if dirty.returncode != 0:
        raise RuntimeError("Cannot read the git state of the repository.")
    if dirty.stdout.strip() and not args.allow_dirty:
        raise RuntimeError("Code or configs differ from HEAD; commit before training:\n" + dirty.stdout)
    training_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = sorted({"scripts/run_canonical.py", args.config, "pyproject.toml",
                     *(path.relative_to(ROOT).as_posix()
                       for path in (ROOT / "src" / "tsfmxai").rglob("*.py")
                       if "__pycache__" not in path.parts)})
    source_sha256 = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source}
    if config.get("epochs_from_run"):
        # The production teacher trains on every split for the best epoch count of
        # the run with a skeleton-disjoint split.
        source_manifest = json.loads((ROOT / config["epochs_from_run"] / "manifest.json")
                                     .read_text(encoding="utf-8"))
        best = source_manifest["results"]["transformer_skeleton"]["best_epoch"]
        config = {**config, "epochs": int(best), "patience": int(best) + 1,
                  "epochs_source": {"run": config["epochs_from_run"], "best_epoch": int(best)}}
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

    progress = ProgressLog(config["epochs"])

    def status(state, **fields):
        status_data.update(fields)
        status_data.update({"state": state, "updated_at_utc": utc_now(),
                            "config_sha256": config_sha})
        atomic_json(status_path, status_data)
        progress(status_data, state)

    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    status("preparing_data", started_at_utc=status_data.get("started_at_utc", utc_now()),
           training_commit=training_commit, source_sha256=source_sha256, dirty=bool(dirty.stdout.strip()),
           command=f"{sys.executable} scripts/run_canonical.py --config {args.config} --run-dir {args.run_dir} --device {args.device}")
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
        now_sha256 = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in source}
        if now_sha256 != source_sha256:
            raise RuntimeError("Source files changed while the run was in progress.")
        provenance = {"config": config, "config_sha256": config_sha,
                      "training_commit": training_commit,
                      "working_tree_dirty": bool(dirty.stdout.strip()),
                      "packages": {name: version(name) for name in
                                   ("numpy", "pyarrow", "safetensors", "torch")},
                      "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                      "cuda_version": torch.version.cuda,
                      "source_sha256": source_sha256,
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
    if args.release_dir:  # outside the try: a packaging error must not mark a finished run as failed
        from tsfmxai.symbolic_regression.canonical_release import package_release
        package_release(run_dir, ROOT / args.release_dir, repo_root=ROOT)
        print("RELEASE", args.release_dir, flush=True)


if __name__ == "__main__":
    main()
