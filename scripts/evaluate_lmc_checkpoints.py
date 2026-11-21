"""Record exploratory 2019 scores for the two completed LMC checkpoints.

This is a diagnostic across two different studies, not a paired comparison
of generator quality. Both models are evaluated on exactly the same origins.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfmxai.evaluation.prior_outcome import seasonal_naive
from tsfmxai.evaluation.real_world import (
    forecast_chronos2,
    load_brazil_energy,
    load_ibirapuera,
    observed_metrics,
    select_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--air", type=Path, required=True)
    parser.add_argument("--energy", type=Path, required=True)
    parser.add_argument("--study-a", type=Path, required=True)
    parser.add_argument("--study-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    windows = {
        "ibirapuera": select_windows(load_ibirapuera(args.air), year=2019),
        "bahia": select_windows(load_brazil_energy(args.energy, state="BA", year=2019), year=2019),
    }
    models = {"study_a": args.study_a, "study_b": args.study_b}
    result = {
        "protocol": {
            "status": "exploratory_2019",
            "context": 96,
            "horizon": 96,
            "origin_stride_hours": 24,
            "minimum_observed_fraction": 0.8,
            "seasonal_period_hours": 24,
            "device": args.device,
            "batch_size": args.batch_size,
            "model_paths": {name: str(path) for name, path in models.items()},
        },
        "domains": {},
    }
    for name, item in windows.items():
        result["domains"][name] = {
            "eligible_origins": item.eligible_count,
            "used_origins": len(item.origins),
            "channels": item.channels,
            "seasonal": observed_metrics(
                seasonal_naive(item.context, item.target.shape[1]),
                item.target,
                item.channels,
            )["per_channel"],
        }

    from tsfmxai.models.chronos2_indexed import load_backbone

    for label, path in models.items():
        model = load_backbone(path).to(args.device).eval()
        for name, item in windows.items():
            predictions = forecast_chronos2(
                model, item, device=args.device, batch_size=args.batch_size
            )
            result["domains"][name][label] = observed_metrics(
                predictions, item.target, item.channels
            )["per_channel"]
        del model

    # Report each domain on its own scale-free summary. Do not pool domains.
    result["relative_mae_geomean_by_dataset"] = {}
    for name, item in result["domains"].items():
        result["relative_mae_geomean_by_dataset"][name] = {}
        for label in models:
            channel_logs = [
                math.log(item[label][channel]["mae"] / item["seasonal"][channel]["mae"])
                for channel in item["channels"]
            ]
            result["relative_mae_geomean_by_dataset"][name][label] = math.exp(
                sum(channel_logs) / len(channel_logs)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"origins": {name: len(item.origins) for name, item in windows.items()},
                      "relative_mae_geomean_by_dataset":
                          result["relative_mae_geomean_by_dataset"]}, indent=2))


if __name__ == "__main__":
    main()
