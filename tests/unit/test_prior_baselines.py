"""Checks for the two prospective baseline contracts."""
import json
from pathlib import Path

import pytest

from tsfmxai.pipelines.prior_baselines import (PriorProtocol, RepeatedBatches,
                                               _cycle, prepare_corpus, run)


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/prior_baseline_example.json"


def small_protocol(tmp_path):
    values = json.loads(CONFIG.read_text(encoding="utf-8"))
    values.update(controlled_updates=2, workers=0, checkpoint_interval=1,
                  validation_batches=1, max_updates_per_stage=2,
                  time_hours=0.0003, own_min_operators=0, own_max_operators=0,
                  own_max_lag=2, own_generated_steps=192)
    return PriorProtocol(**values)


def test_stage_budgets_and_lr_endpoints(tmp_path):
    protocol = small_protocol(tmp_path)
    assert protocol.stage_updates() == (1, 1)
    assert _cycle(0, 1e-4)[0] == pytest.approx(4e-6)
    assert _cycle(0.3, 1e-4)[0] == pytest.approx(1e-4)
    assert _cycle(1, 1e-4)[0] == pytest.approx(4e-10)


def test_corpus_rejects_mismatched_protocol(tmp_path):
    protocol = small_protocol(tmp_path)
    corpus = tmp_path / "corpus"
    prepare_corpus(protocol, "own", None, corpus)
    other = PriorProtocol(**{**protocol.__dict__, "seed": protocol.seed + 1})
    with pytest.raises(ValueError, match="provenance"):
        run(other, "own", "controlled", None, tmp_path / "train", corpus=corpus,
            device="cpu", tiny=True)


def test_tiny_controlled_run_has_two_stage_checkpoints(tmp_path):
    protocol = small_protocol(tmp_path)
    corpus = tmp_path / "corpus"
    prepare_corpus(protocol, "own", None, corpus)
    result = run(protocol, "own", "controlled", None, tmp_path / "train",
                 corpus=corpus, device="cpu", tiny=True)
    assert result["updates"] == 2
    assert (tmp_path / "train/independent_end").is_dir()
    assert (tmp_path / "train/model_final").is_dir()
    rows = [json.loads(line) for line in (tmp_path / "train/validation.jsonl").read_text().splitlines()]
    assert {row["stage"] for row in rows} == {"independent", "correlated"}


def test_repeated_bank_preserves_window_order():
    class Bank:
        spec = object()

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return index

    repeated = RepeatedBatches(Bank(), 2, 8)
    assert [repeated[i] for i in range(len(repeated))] == [2, 0, 1, 2, 0, 1]


def test_tiny_time_run_stops_by_clock_and_reuses_validation(tmp_path):
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "max_updates_per_stage": 100,
                                "time_hours": 0.0002})
    result = run(protocol, "own", "time", None, tmp_path / "time", device="cpu", tiny=True)
    assert result["stop_reason"] == "budget_elapsed"
    assert (tmp_path / "time/validation_corpus/independent/complete").exists()
    assert result["environment"]["packages"]["torch"]


def test_time_run_fails_on_safety_cap(tmp_path):
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "max_updates_per_stage": 1,
                                "time_hours": 1})
    with pytest.raises(RuntimeError, match="Safety update limit"):
        run(protocol, "own", "time", None, tmp_path / "cap", device="cpu", tiny=True)


def test_resume_discards_uncheckpointed_telemetry(tmp_path):
    protocol = small_protocol(tmp_path)
    corpus = tmp_path / "corpus"
    prepare_corpus(protocol, "own", None, corpus)
    output = tmp_path / "train"
    run(protocol, "own", "controlled", None, output, corpus=corpus, device="cpu", tiny=True)
    with (output / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"update": 3, "elapsed_seconds": 100,
                                 "batch_wait_seconds": 2}) + "\n")
    result = run(protocol, "own", "controlled", None, output, corpus=corpus,
                 resume=True, device="cpu", tiny=True)
    assert result["updates"] == 2
    assert result["elapsed_seconds"] >= 100
    assert all(json.loads(line)["update"] <= 2 for line in
               (output / "telemetry.jsonl").read_text().splitlines())
