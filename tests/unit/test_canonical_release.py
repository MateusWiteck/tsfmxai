"""The packaged release is self-contained, loadable and records the generator parameters."""

import json

import pytest
import torch

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_large_scale import SPLITS, count_corpus, generate_parquet
from tsfmxai.symbolic_regression.canonical_models import (
    CanonicalEncoder, CanonicalEncoderDecoder, collate_canonical)
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


def test_encoder_alone_matches_the_full_model_and_repackaging_keeps_the_dataset(tmp_path):
    from tsfmxai.symbolic_regression.canonical_large_scale import iter_shard_examples

    run, config, model = _finished_run(tmp_path)
    out = tmp_path / "release"
    package_release(run, out, repo_root=tmp_path)
    models = out / "models" / "transformer_skeleton"
    assert (models / "encoder.safetensors").exists()
    assert "encoder.safetensors" in (out / "SHA256SUMS").read_text()
    examples = [e for _, e in zip(range(4), iter_shard_examples(run, "train", seed=1))]
    batch = collate_canonical(examples)
    encoder = CanonicalEncoder.from_pretrained(models)
    model.eval()
    with torch.no_grad():
        memory, mask, root = encoder(batch)
        full_memory, full_mask, hidden = model.encode(batch)
    assert torch.equal(memory, full_memory) and torch.equal(mask, full_mask)
    assert torch.allclose(torch.tanh(model.initial(root)).unsqueeze(0), hidden)

    # The Parquet corpus may already live in the release while the run folder lacks it.
    import shutil
    shutil.rmtree(run / "parquet")
    again = package_release(run, out, repo_root=tmp_path)
    assert any(name.startswith("dataset/data/train/") for name in again["files"])


def test_release_records_commits_and_ships_the_generator_source(tmp_path):
    import hashlib
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not available")
    first, second = "VALUE = 1" + chr(10), "VALUE = 2" + chr(10)
    run, config, _ = _finished_run(tmp_path)
    repo = tmp_path / "repo"
    module = repo / "src" / "pkg" / "generator.py"
    module.parent.mkdir(parents=True)
    module.write_text(first, encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]" + chr(10), encoding="utf-8")
    (repo / "constraints.txt").write_text("", encoding="utf-8")

    def git(*args):
        return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                       cwd=repo, text=True).strip()

    git("init", "-q")
    git("remote", "add", "origin", "git@github.com:someone/tsfmxai.git")
    git("add", ".")
    git("commit", "-q", "-m", "training code")
    training = git("rev-parse", "HEAD")
    module.write_text(second, encoding="utf-8")
    git("commit", "-qam", "later change")

    def record(hashes):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["provenance"]["source_sha256"] = hashes
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    record({"src/pkg/generator.py": hashlib.sha256(first.encode()).hexdigest()})
    out = tmp_path / "release"
    package_release(run, out, repo_root=repo, training_commit=training)
    metadata = json.loads((out / "dataset" / "metadata.json").read_text(encoding="utf-8"))
    provenance = metadata["provenance"]
    assert provenance["training_commit"] == training != provenance["release_commit"]
    assert provenance["repository"] == "https://github.com/someone/tsfmxai"
    assert provenance["source_verified_files"] == 1
    assert (out / "dataset" / "source" / "src" / "pkg" / "generator.py").read_bytes() == first.encode()
    assert (out / "models" / "source" / "src" / "pkg" / "generator.py").read_bytes() == second.encode()
    assert metadata["training_config"] == config
    card = (out / "dataset" / "README.md").read_text(encoding="utf-8")
    assert f"https://github.com/someone/tsfmxai/commit/{training}" in card
    assert f"git checkout {training}" in card

    record({"src/pkg/generator.py": "0" * 64})
    with pytest.raises(ValueError):  # a commit that is not the one the run used is refused
        package_release(run, tmp_path / "other", repo_root=repo, training_commit=training)

    record({})
    (repo / "dirty.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):  # uncommitted changes would make the commit reference false
        package_release(run, tmp_path / "dirty", repo_root=repo, training_commit=training)


def test_release_requires_a_completed_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        package_release(tmp_path / "empty", tmp_path / "release", repo_root=tmp_path)
