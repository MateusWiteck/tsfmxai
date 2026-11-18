"""Checks for the two prospective baseline contracts."""
import json
from pathlib import Path

import pytest

from tsfmxai.pipelines.prior_baselines import PriorProtocol, _cycle, prepare_corpus, run


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
