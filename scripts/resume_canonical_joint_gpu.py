"""Resume the unfinished joint canonical model on CUDA using the existing corpus."""

from __future__ import annotations

import argparse
import ctypes
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

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_large_scale import train_parquet_variant
from tsfmxai.symbolic_regression.canonical_training import atomic_json, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/canonicalization_5m_ast_parquet_unique_eval.json")
    parser.add_argument("--run-dir", default="outputs/canonicalization/ast_5m_unique_eval_20261030")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    run_dir = (ROOT / args.run_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status_path = run_dir / "status.json"
    status_data = json.loads(status_path.read_text(encoding="utf-8"))
    config_sha256 = sha256(config_path.read_bytes()).hexdigest()
    if status_data.get("config_sha256") != config_sha256:
        raise ValueError("Run configuration differs from the existing dataset")
    if not status_data.get("results", {}).get("transformer_skeleton"):
        raise ValueError("The first model is not recorded as complete")
    if status_data.get("results", {}).get("transformer_joint"):
        raise ValueError("The joint model is already complete")
    if not (run_dir / "parquet" / "train").is_dir():
        raise FileNotFoundError("Training corpus is missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda:0"
    variant = next(v for v in config["variants"] if v["name"] == "transformer_joint")
    generator = config["generator"]
    codec = CanonicalCodec(generator["num_series"], generator["minimum_lag"],
                           generator["maximum_lag"], generator["max_degree"])
    hardware = {"device": device, "name": torch.cuda.get_device_name(0),
                "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "python": platform.python_version()}
    command = f"{sys.executable} scripts/resume_canonical_joint_gpu.py --config {args.config} --run-dir {args.run_dir}"

    def update(state, **fields):
        status_data.update(fields)
        status_data.update({"state": state, "updated_at_utc": utc_now(),
                            "config_sha256": config_sha256})
        atomic_json(status_path, status_data)

    start = time.perf_counter()
    if platform.system() == "Windows":
        # Keep the machine awake while this long-running process is active.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    update("gpu_training_starting", active_variant=variant["name"],
           gpu_resume_command=command, gpu_hardware=hardware, gpu_training_pid=os.getpid())
    print("GPU", json.dumps(hardware), flush=True)
    try:
        result = train_parquet_variant(config, run_dir, codec, variant, update, device=device)
        results = dict(status_data["results"])
        results[variant["name"]] = result
        manifest = {"provenance": {"config": config, "config_sha256": config_sha256,
                    "hardware": hardware, "command": command, "completed_at_utc": utc_now(),
                    "elapsed_seconds": time.perf_counter() - start},
                    "corpus": status_data["corpus"], "results": results}
        atomic_json(run_dir / "manifest.json", manifest)
        update("completed", results=results, active_variant=None,
               gpu_elapsed_seconds=time.perf_counter() - start)
        print("COMPLETE", json.dumps(result), flush=True)
    except Exception as exc:
        update("failed", active_variant=variant["name"], error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if platform.system() == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
