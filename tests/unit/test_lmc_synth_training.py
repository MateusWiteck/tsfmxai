"""Protocol arithmetic, data isolation, grouping, and real optimizer/resume checks."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tsfmxai.datasets.lmc_synth import LMCSpec, TimePFNBatches, batch_loader
from tsfmxai.pipelines.chronos2_lmc import TrainingSpec, scratch_model, train
from tsfmxai.pipelines.chronos2_lmc import _observe_early_stopping

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "experiments_local/TimePFN"


def test_early_stopping_freezes_best_before_trigger():
    model = torch.nn.Linear(1, 1, bias=False)
    settings = TrainingSpec(early_stopping_patience=2)
    state = dict(best=None, weights=None, bad_checks=0, trigger=None)
    for step, score in enumerate([3., 2., 2., 4., 1.], start=1):
        with torch.no_grad():
            model.weight.fill_(step)
        _observe_early_stopping(state, model, score, {"global_update": step}, settings)
    assert state["best"]["global_update"] == 2
    assert state["trigger"]["global_update"] == 4
    assert state["weights"]["weight"].item() == 2


def test_published_loader_budget():
    spec = LMCSpec(.1, 2., 2., 2.)
    assert spec.groups("independent") == 6665
    assert spec.windows("independent") == 5551945
    assert spec.updates("independent") == 346996
    assert spec.windows("lmc") == 1575000
    assert spec.updates("lmc") == 98437
    original = replace(spec, channels=160)
    assert original.windows("independent") == 826336


def _cached_data(tmp_path, monkeypatch):
    monkeypatch.setattr("tsfmxai.datasets.lmc_synth.verify_timepfn", lambda _: {})
    monkeypatch.setattr("tsfmxai.pipelines.chronos2_lmc.verify_timepfn", lambda _: {})
    spec = LMCSpec(.1, 2., 2., 2., batch_size=2, independent_bank=8, correlated_bank=4)
    for split in range(3):
        for family in ("independent", "lmc"):
            dataset = TimePFNBatches(spec, UPSTREAM, tmp_path, family, split=split, limit=2)
            count = spec.independent_bank if family == "independent" else spec.correlated_bank
            for index in range(count):
                rng = np.random.default_rng(index + 100 * split)
                shape = (1024,) if family == "independent" else (6, 1024)
                np.save(dataset.cache / f"{index:06d}.npy", rng.normal(size=shape).astype("float32"))
    return spec


def test_window_resume_and_split_isolation(tmp_path, monkeypatch):
    spec = _cached_data(tmp_path, monkeypatch)
    for family in ("independent", "lmc"):
        first = TimePFNBatches(spec, UPSTREAM, tmp_path, family, limit=2)
        resumed = TimePFNBatches(spec, UPSTREAM, tmp_path, family, limit=2, start=1)
        assert torch.equal(first[1]["context"], resumed[0]["context"])
        assert first[0]["context"].shape == (2, 96, 6)
        heldout = TimePFNBatches(spec, UPSTREAM, tmp_path, family, split=1, limit=2)
        assert first.cache != heldout.cache
        assert not torch.equal(first[0]["context"], heldout[0]["context"])


def test_trigger_saves_distinct_models_and_full_training_continues(tmp_path, monkeypatch):
    pytest.importorskip("chronos.chronos2.model")
    torch.set_num_threads(1)
    spec = _cached_data(tmp_path / "cache", monkeypatch)
    scores = iter([3., 2., 1., 1., 2., .5])
    monkeypatch.setattr("tsfmxai.pipelines.chronos2_lmc.evaluate",
                        lambda *a, **kw: {"loss": next(scores, 9.)})
    settings = TrainingSpec(workers=0, microbatch=1, checkpoint_interval=1,
                            validation_batches=1, steps_per_stage=3, device="cpu",
                            early_stopping_patience=1)
    model = scratch_model(tiny=True)
    result, _ = train(model, spec, settings, UPSTREAM, tmp_path / "cache", tmp_path / "run")
    assert result["early_stopping"]["triggered"]
    assert result["early_stopping"]["selected_checkpoint"]["global_update"] == 4
    assert result["early_stopping"]["stopped_at"]["global_update"] == 5
    assert result["without_early_stopping"]["stopped_at"]["global_update"] == 6
    from safetensors.torch import load_file
    early = load_file(tmp_path / "run/model_early_stopping/model.safetensors")
    full = load_file(tmp_path / "run/model_without_early_stopping/model.safetensors")
    assert any(not torch.equal(early[key], full[key]) for key in early)


def test_real_training_checkpoint_resume(tmp_path, monkeypatch):
    pytest.importorskip("chronos.chronos2.model")
    torch.set_num_threads(1)
    spec = _cached_data(tmp_path / "cache", monkeypatch)
    model = scratch_model(tiny=True)
    before = [p.detach().clone() for p in model.parameters()]
    settings = TrainingSpec(workers=0, microbatch=1, checkpoint_interval=1,
                            validation_batches=1, steps_per_stage=2, device="cpu")
    result, history = train(model, spec, settings, UPSTREAM, tmp_path / "cache", tmp_path / "run")
    assert result["updates"] == 4
    assert result["early_stopping"]["stop_reason"] == "budget_exhausted_without_trigger"
    assert result["without_early_stopping"]["stopped_at"]["global_update"] == 4
    assert (tmp_path / "run/model_early_stopping/config.json").exists()
    assert (tmp_path / "run/model_without_early_stopping/config.json").exists()
    assert all(np.isfinite(row["loss"]) for row in history)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    with pytest.raises(FileExistsError):
        train(model, spec, settings, UPSTREAM, tmp_path / "cache", tmp_path / "run")
    restored = scratch_model(tiny=True)
    resumed, _ = train(restored, spec, settings, UPSTREAM, tmp_path / "cache", tmp_path / "run", resume=True)
    assert resumed["test"] == result["test"]
    assert resumed["early_stopping"] == result["early_stopping"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))

    # Interrupt during the second validation: checkpoint still points to step 1.
    import tsfmxai.pipelines.chronos2_lmc as pipeline
    original_evaluate = pipeline.evaluate
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(scratch_model(tiny=True), spec, settings, UPSTREAM,
              tmp_path / "cache", tmp_path / "interrupted")
    monkeypatch.setattr(pipeline, "evaluate", original_evaluate)
    recovered = scratch_model(tiny=True)
    recovered_result, _ = train(recovered, spec, settings, UPSTREAM,
                                tmp_path / "cache", tmp_path / "interrupted", resume=True)
    assert recovered_result["test"] == result["test"]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), recovered.parameters()))


@pytest.mark.skipif(not UPSTREAM.exists(), reason="Pinned upstream checkout is optional")
def test_actual_generator_worker_invariance(tmp_path):
    # Prewarm only the requested trajectories, then compare spawn and parent.
    spec = LMCSpec(.1, 2., 2., 2., batch_size=1, independent_bank=8, correlated_bank=2)
    dataset = TimePFNBatches(spec, UPSTREAM, tmp_path, "lmc", limit=1)
    parent = dataset[0]
    parallel = next(iter(batch_loader(dataset, workers=1)))
    assert torch.equal(parent["context"], parallel["context"])
    assert torch.equal(parent["future"], parallel["future"])
    # A different cache forces regeneration from seed rather than reading bytes.
    regenerated = TimePFNBatches(spec, UPSTREAM, tmp_path / "regenerate", "lmc", limit=1)[0]
    assert torch.equal(parent["context"], regenerated["context"])
