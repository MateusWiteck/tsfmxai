"""Evaluate two trained prior arms on the same exploratory 2019 real windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.evaluation.prior_outcome import (per_dataset_relative_mae,
                                               paired_block_interval, seasonal_naive)
from tsfmxai.evaluation.real_world import (forecast_chronos2, load_brazil_energy,
                                            load_ibirapuera, select_windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", type=Path, required=True, help="PANGAEA 940276 TSV")
    parser.add_argument("--energy", type=Path, required=True, help="EnDAT Zenodo ZIP")
    parser.add_argument("--lmc", type=Path, required=True, help="LMC model_final directory")
    parser.add_argument("--own", type=Path, required=True, help="Own model_final directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--block-windows", type=int, default=7)
    args = parser.parse_args()

    windows = {
        "ibirapuera": select_windows(load_ibirapuera(args.air), year=2019),
        "bahia": select_windows(load_brazil_energy(args.energy, state="BA", year=2019), year=2019),
    }
    domains = {name: {"target": item.target, "channels": item.channels,
                      "predictions": {"seasonal": seasonal_naive(item.context,
                                                                    item.target.shape[1])}}
               for name, item in windows.items()}
    from chronos.chronos2.model import Chronos2Model
    for arm, folder in (("lmc", args.lmc), ("own", args.own)):
        model = Chronos2Model.from_pretrained(str(folder)).to(args.device)
        for name, item in windows.items():
            domains[name]["predictions"][arm] = forecast_chronos2(
                model, item, device=args.device, batch_size=args.batch_size)
        del model

    result = per_dataset_relative_mae(domains)
    result["paired_block_interval"] = paired_block_interval(
        domains, block_windows=args.block_windows,
        replicates=args.bootstrap_replicates)
    result["protocol"] = {"status": "exploratory_2019", "context": 96,
                          "horizon": 96, "origin_stride_hours": 24,
                          "minimum_observed_fraction": 0.8,
                          "seasonal_period_hours": 24,
                          "models": {"lmc": str(args.lmc), "own": str(args.own)},
                          "domains": {name: {"eligible_origins": item.eligible_count,
                                             "used_origins": len(item.origins),
                                             "channels": item.channels}
                                      for name, item in windows.items()}}
    for name, detail in result["per_domain"].items():
        detail["beats_seasonal_per_channel"] = {
            arm: {channel: detail["per_channel"][arm][channel]["mae"] <
                  detail["per_channel"]["seasonal"][channel]["mae"]
                  for channel in windows[name].channels}
            for arm in ("lmc", "own")}
    result["beats_seasonal_by_dataset"] = {
        name: {arm: score < 1 for arm, score in detail["relative_geomean"].items()}
        for name, detail in result["per_domain"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"by_dataset": {
                          name: {"own_over_lmc": detail["own_over_lmc"],
                                 "relative_geomean": detail["relative_geomean"],
                                 "interval": result["paired_block_interval"][name]}
                          for name, detail in result["per_domain"].items()},
                      "beats_seasonal_by_dataset": result["beats_seasonal_by_dataset"]},
                     indent=2))


if __name__ == "__main__":
    main()
