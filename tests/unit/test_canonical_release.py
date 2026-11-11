"""The packaged release is self-contained, loadable and records the generator parameters."""

import json

import pytest
import torch

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_large_scale import SPLITS, count_corpus, generate_parquet
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoderDecoder
from tsfmxai.symbolic_regression.canonical_release import package_release

safetensors = pytest.importorskip("safetensors.torch")


def _finished_run(tmp_path):
    run = tmp_path / "run"
    generator = {"minimum_operators": 1, "maximum_operators": 3, "num_series": 1,
                 "minimum_lag": 1, "maximum_lag": 4, "max_degree": 3}
    config = {"experiment": "tiny", "ast_count": 120, "shard_size": 60, "generation_workers": 1,
              "data_seed": 3, "split_seed": 4, "protection_probability": 0.5,
              "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
              "generator": generator, "batch_size": 8, "epochs": 1,
              "model": {"dimension": 16, "heads": 2, "encoder_layers": 1, "dropout": 0.0,
                        "max_nodes": 11},
              "variants": [{"name": "transformer_skeleton", "encoder": "transformer",
                            "coefficients": False}]}
    generate_parquet(config, run, lambda *a, **k: None)
    corpus = count_corpus(config, run, lambda *a, **k: None)
    codec = CanonicalCodec(1, 1, 4, 3)
    model = CanonicalEncoderDecoder(codec.vocab_size, "transformer", False, **config["model"])
    checkpoint = run / "checkpoints" / "transformer_skeleton"
    checkpoint.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "epoch": 2,
                "variant": config["variants"][0]}, checkpoint / "best.pt")
    result = {"count": 10, "skeleton_exact": 0.5, "best_epoch": 2, "history": [{"epoch": 1}],
              "checkpoint": "/training/machine/best.pt"}
    (run / "manifest.json").write_text(json.dumps({
        "provenance": {"config": config, "config_sha256": "abc", "source_sha256": {}},
        "corpus": corpus, "results": {"transformer_skeleton": result}}), encoding="utf-8")
    return run, config, model


def test_release_layout_metadata_and_roundtrip(tmp_path):
    run, config, model = _finished_run(tmp_path)
    out = tmp_path / "release"
    release = package_release(run, out, repo_root=tmp_path)
    models = out / "models" / "transformer_skeleton"
    cfg = json.loads((models / "config.json").read_text(encoding="utf-8"))
    assert cfg["generator"]["configured"] == config["generator"]
    assert cfg["generator"]["resolved"]["require_lag"] is True  # defaults are resolved
    assert cfg["generator"]["protection_probability"] == 0.5
    assert (cfg["generator"]["data_seed"], cfg["generator"]["split_seed"]) == (3, 4)
    assert cfg["codec"]["vocab_size"] == CanonicalCodec(1, 1, 4, 3).vocab_size
    assert cfg["model_kwargs"]["max_nodes"] == 11 and cfg["best_epoch"] == 2
    assert "checkpoint" not in cfg["evaluation"]  # no training-machine path leaks

    rebuilt = CanonicalEncoderDecoder(**cfg["model_kwargs"])
    rebuilt.load_state_dict(safetensors.load_file(str(models / "model.safetensors")))
    for key, value in model.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value)

    stored = sum(1 for split in SPLITS for _ in (out / "dataset" / "data" / split).glob("*.parquet"))
    assert stored == 2 * len(SPLITS)  # two shards per split
    metadata = json.loads((out / "dataset" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["generator"]["configured"] == config["generator"]
    assert "split: train" in (out / "dataset" / "README.md").read_text(encoding="utf-8")
    assert set(release["files"]) == {line.split("  ", 1)[1]
                                     for line in (out / "SHA256SUMS").read_text().splitlines()}


def test_release_requires_a_completed_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        package_release(tmp_path / "empty", tmp_path / "release", repo_root=tmp_path)
