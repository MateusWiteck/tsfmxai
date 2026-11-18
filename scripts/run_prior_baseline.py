"""Run one prospective synthetic-prior arm on this machine.

Examples:
  python scripts/run_prior_baseline.py prepare --config CONFIG --arm lmc --upstream PATH --corpus PATH
  python scripts/run_prior_baseline.py train --config CONFIG --study controlled --arm lmc --upstream PATH --corpus PATH --output PATH
  python scripts/run_prior_baseline.py train --config CONFIG --study time --arm own --output PATH
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.pipelines.prior_baselines import PriorProtocol, prepare_corpus, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("lmc", "own"), required=True)
    parser.add_argument("--study", choices=("controlled", "time"))
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="CPU smoke test only")
    args = parser.parse_args()
    protocol = PriorProtocol.read(args.config)
    if args.arm == "lmc" and args.upstream is None:
        parser.error("LMC requires --upstream pointing to the pinned TimePFN checkout")
    if args.command == "prepare":
        if args.corpus is None or args.study is not None or args.resume:
            parser.error("prepare requires --corpus and no --study/--resume")
        result = prepare_corpus(protocol, args.arm, args.upstream, args.corpus)
    else:
        if args.study is None or args.output is None:
            parser.error("train requires --study and --output")
        result = run(protocol, args.arm, args.study, args.upstream, args.output,
                     corpus=args.corpus, resume=args.resume,
                     device=args.device, tiny=args.tiny)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
