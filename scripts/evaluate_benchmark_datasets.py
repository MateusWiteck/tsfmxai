"""Zero-shot scores of Chronos-2 checkpoints on ETTh1, ETTh2 and Weather.

Origins lie on a daily grid inside each dataset's usual chronological test
split; context 96 h, horizon 96 h. Each dataset is summarized on its own by the
geometric mean over channels of MAE relative to the 24-hour seasonal naive.
Download the inputs into --data-dir first (see the experiment record).
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
from tsfmxai.evaluation.real_world import (BENCHMARK_FILES, benchmark_windows, forecast_chronos2,
                                            last_observed_forecast, observed_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="checkpoint folder; repeat for several models")
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_FILES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    models = dict(item.split("=", 1) for item in args.model)

    windows = {name: benchmark_windows(name, args.data_dir) for name in args.datasets}
    result = {"protocol": {"status": "zero_shot_benchmark_test_split", "context": 96, "horizon": 96,
                           "origin_stride_hours": 24, "minimum_observed_fraction": 0.8,
                           "seasonal_period_hours": 24, "device": args.device,
                           "models": models},
              "domains": {}}
    for name, item in windows.items():
        result["domains"][name] = {
            "eligible_origins": item.eligible_count, "used_origins": len(item.origins),
            "first_origin": item.origins[0], "last_origin": item.origins[-1],
            "channels": list(item.channels),
            "seasonal": observed_metrics(seasonal_naive(item.context, item.target.shape[1]),
                                         item.target, item.channels)["per_channel"],
            "persistence": observed_metrics(last_observed_forecast(item.context, item.target.shape[1]),
                                            item.target, item.channels)["per_channel"],
        }
    from tsfmxai.models.chronos2_indexed import load_backbone

    for label, path in models.items():
        model = load_backbone(path).to(args.device).eval()
        for name, item in windows.items():
            prediction = forecast_chronos2(model, item, device=args.device, batch_size=args.batch_size)
            result["domains"][name][label] = observed_metrics(prediction, item.target, item.channels)["per_channel"]
        del model

    summary = {}
    for name, item in result["domains"].items():
        summary[name] = {}
        for label in ["persistence", *models]:
            ratios = [item[label][c]["mae"] / item["seasonal"][c]["mae"] for c in item["channels"]]
            summary[name][label] = {"relative_mae_geomean": math.exp(sum(map(math.log, ratios)) / len(ratios)),
                                    "channels_better_than_seasonal": sum(r < 1 for r in ratios),
                                    "channels": len(ratios)}
    result["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
