"""Select 25% or 50% KernelSynth using paired real-data development origins.

The 2019 origins have already been inspected in this project and are development
evidence, not an independent confirmation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.evaluation.prior_outcome import seasonal_naive, select_hybrid_probability
from tsfmxai.evaluation.real_world import (forecast_chronos2, load_brazil_energy,
                                            load_ibirapuera, select_windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--energy", type=Path, required=True)
    parser.add_argument("--p25", type=Path, required=True)
    parser.add_argument("--p50", type=Path, required=True)
    parser.add_argument("--p25-corpus", type=Path,
                        help="Prepared corpus of the p=0.25 run; its manifest gives the realized mixture")
    parser.add_argument("--p50-corpus", type=Path,
                        help="Prepared corpus of the p=0.50 run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    windows = {
        "ibirapuera": select_windows(load_ibirapuera(args.air), year=2019),
        "bahia": select_windows(load_brazil_energy(args.energy, state="BA", year=2019), year=2019),
    }
    domains = {name: {"target": item.target, "channels": item.channels,
                      "predictions": {"seasonal": seasonal_naive(item.context, item.target.shape[1])}}
               for name, item in windows.items()}
    from tsfmxai.models.chronos2_indexed import load_backbone

    for label, folder in (("p25", args.p25), ("p50", args.p50)):
        model = load_backbone(folder).to(args.device).eval()
        for name, item in windows.items():
            domains[name]["predictions"][label] = forecast_chronos2(
                model, item, device=args.device, batch_size=args.batch_size)
        del model
    result = select_hybrid_probability(domains)
    result["protocol"] = {"status": "development_2019", "context": 96,
                          "horizon": 96, "origin_stride_hours": 24,
                          "minimum_observed_fraction": 0.8,
                          "models": {"p25": str(args.p25), "p50": str(args.p50)},
                          "domains": {name: {"eligible_origins": item.eligible_count,
                                             "used_origins": len(item.origins),
                                             "channels": item.channels}
                                      for name, item in windows.items()}}
    realized = {}
    for label, corpus in (("p25", args.p25_corpus), ("p50", args.p50_corpus)):
        if corpus is not None:
            manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
            realized[label] = {"channel_families": manifest.get("channel_families"),
                               "kernel_synth_fraction": manifest.get("kernel_synth_fraction")}
    result["realized_mixture"] = realized or None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_probability": result["selected_probability"],
                      "per_domain": {name: item["relative_geomean"]
                                     for name, item in result["per_domain"].items()},
                      "equal_domain_geomean": result["equal_domain_geomean"]}, indent=2))


if __name__ == "__main__":
    main()
