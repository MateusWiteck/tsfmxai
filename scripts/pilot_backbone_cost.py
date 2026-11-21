"""Measure time and memory per update of the Chronos-2 backbone used by the studies.

Runs optimizer updates on random 16 x 96 x 6 batches with 96-step horizons, for
the configured patch size and channel-index embedding, and prints one JSON
record. Use it on the training GPU before freezing the study budgets.

    python scripts/pilot_backbone_cost.py --patch-size 1 --channel-embedding
    python scripts/pilot_backbone_cost.py --patch-size 16            # completed LMC runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.pipelines.chronos2_lmc import _inputs, _sync, scratch_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--channel-embedding", action="store_true")
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tiny", action="store_true", help="Smoke test with a tiny model")
    args = parser.parse_args()
    if 16 % args.microbatch:
        raise ValueError("microbatch must divide the effective batch of 16")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = scratch_model(seed=2021, tiny=args.tiny, patch_size=args.patch_size,
                          channel_index_embedding=6 if args.channel_embedding else None).to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(.95, .999), eps=1e-8)
    generator = torch.Generator().manual_seed(0)
    batch = {"context": torch.randn(16, 96, 6, generator=generator),
             "future": torch.randn(16, 96, 6, generator=generator)}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    times = []
    for step in range(args.warmup + args.measured):
        _sync(args.device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, 16, args.microbatch):
            output = model(**_inputs(batch, model, args.device, offset, offset + args.microbatch))
            (output.loss * args.microbatch / 16).backward()
        optimizer.step()
        _sync(args.device)
        if step >= args.warmup:
            times.append(time.perf_counter() - started)
    seconds = sum(times) / len(times)
    record = {
        "patch_size": args.patch_size, "channel_embedding": args.channel_embedding,
        "microbatch": args.microbatch, "warmup": args.warmup, "measured": args.measured,
        "seconds_per_update": seconds, "updates_per_hour": 3600 / seconds,
        "hours_for_445433_updates": 445433 * seconds / 3600,
        "peak_memory_bytes": (torch.cuda.max_memory_allocated(args.device)
                              if args.device.startswith("cuda") else None),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": (torch.cuda.get_device_name(args.device) if args.device.startswith("cuda")
                   else platform.processor()),
        "torch": torch.__version__,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
