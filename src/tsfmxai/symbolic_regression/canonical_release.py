"""Package a finished canonicalization run in a layout ready for Hugging Face.

The training pipeline leaves checkpoints with optimizer state, absolute paths and
the corpus under ``parquet/``. :func:`package_release` turns that run directory
into a self-contained release directory::

    release/
      models/<variant>/model.safetensors   weights only
      models/<variant>/config.json         architecture, codec, generator, metrics
      models/<variant>/training_history.json
      models/<variant>/README.md           model card
      dataset/data/{train,validation,test}/part-*.parquet
      dataset/metadata.json                generator parameters, counts, splits
      dataset/README.md                    dataset card (with the viewer config)
      SHA256SUMS
      release_manifest.json

Nothing here uploads anything; see ``scripts/upload_to_huggingface.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.symbolic_regression.canonical_data import (
    ATOM_OFFSET, BOS, END_TERM, EOS, PAD, TANH_TOKEN, CanonicalCodec,
)
from tsfmxai.symbolic_regression.canonical_large_scale import SCHEMA, SPLITS

PROTECTION_RULE = (
    "With probability protection_probability a sampled tree is protected before "
    "canonicalization: every lag leaf inside a product of two lag-dependent factors is "
    "wrapped in tanh (saturate_product_lags), so products become tanh(u)*tanh(v); "
    "otherwise the tree keeps its raw products. The draw is per tree and deterministic: "
    "SHA-256 of 'protection|data_seed|sample_index'."
)
SPLIT_RULE = (
    "A row goes to train, validation or test by SHA-256 of 'split_seed|skeleton digest' "
    "modulo 10000 (<7000 train, <8000 validation, otherwise test), so a canonical "
    "skeleton belongs to exactly one partition."
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _git(*args: str, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def generator_metadata(config: dict) -> dict:
    """Generator parameters as configured and with every default resolved."""
    configured = dict(config["generator"])
    resolved = asdict(CombinatorialTreeConfig(**configured))
    return {
        "sampler": "tsfmxai.generators.CombinatorialTreeSampler",
        "configured": configured,
        "resolved": resolved,
        "protection_probability": config.get("protection_probability", 0.5),
        "protection_rule": PROTECTION_RULE,
        "data_seed": config["data_seed"],
        "split_seed": config["split_seed"],
        "sample_seed_rule": "derive_sample_seed(data_seed, sample_index)",
        "split_proportions": config["split"],
        "split_rule": SPLIT_RULE,
        "unique_eval_skeletons": bool(config.get("unique_eval_skeletons")),
        "sampled_asts": config["ast_count"],
    }


def codec_metadata(config: dict) -> dict:
    g = config["generator"]
    codec = CanonicalCodec(g["num_series"], g["minimum_lag"], g["maximum_lag"], g["max_degree"])
    return {
        "num_series": codec.num_series, "minimum_lag": codec.minimum_lag,
        "maximum_lag": codec.maximum_lag, "max_degree": codec.max_degree,
        "vocab_size": codec.vocab_size,
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "END_TERM": END_TERM},
        "source_tokens": {"ADD": 1, "MUL": 2, "PARAM": 3, "TANH": TANH_TOKEN},
        "atom_offset": ATOM_OFFSET,
        "atom_id": "ATOM_OFFSET + 2 * (series * lag_count + lag - minimum_lag) + (1 if tanh else 0)",
    }


def _model_card(name: str, config_json: dict) -> str:
    g, ev = config_json["generator"]["resolved"], config_json["evaluation"]
    metrics = "\n".join(f"| {k} | {v} |" for k, v in ev.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool))
    return f"""---
library_name: pytorch
tags:
- time-series
- symbolic-regression
- tsfmxai
---

# {name}

Encoder-decoder that reads the prefix form of a random recurrence tree (an AST over
`x(t-lag)`, coefficients, `+`, `*` and `tanh`) and writes its canonical polynomial
form term by term. Defined in `src/tsfmxai/symbolic_regression/canonical_models.py`
(`CanonicalEncoderDecoder`) of the tsfmxai repository, commit
`{config_json["provenance"]["git_commit"]}`.

- Variant: `{name}` ({"symbols and one coefficient per END_TERM" if config_json["model_kwargs"]["predict_coefficients"] else "symbols only, no coefficients"}).
- Encoder: `{config_json["model_kwargs"]["encoder"]}`; maximum input length (`max_nodes`): {config_json["model_kwargs"]["max_nodes"]} tree nodes.
- Vocabulary: {config_json["codec"]["vocab_size"]} symbols, lags {g["minimum_lag"]} to {g["maximum_lag"]}, RAW and TANH factors.

## Training data

Random trees from `CombinatorialTreeSampler` with
{g["minimum_operators"]}-{g["maximum_operators"]} binary operators, {g["num_series"]} series,
lags {g["minimum_lag"]}-{g["maximum_lag"]}, degree at most {g["max_degree"]}, coefficient
magnitude in [{g["minimum_parameter_magnitude"]}, {g["maximum_parameter_magnitude"]}].
{config_json["generator"]["sampled_asts"]:,} trees were sampled; each one is protected
with probability {config_json["generator"]["protection_probability"]}. All generator
parameters and seeds are in `config.json` (key `generator`). Validation and test keep
one row per canonical skeleton, and no skeleton appears in two partitions.

## Test metrics

| metric | value |
|---|---|
{metrics}

## Loading

```python
import json
from safetensors.torch import load_file
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoderDecoder

cfg = json.load(open("config.json"))
model = CanonicalEncoderDecoder(**cfg["model_kwargs"])
model.load_state_dict(load_file("model.safetensors"))
model.eval()
```

The input and target ids come from `tsfmxai.symbolic_regression.canonical_data.CanonicalCodec`
built with the `codec` block of `config.json`.
"""


def _dataset_card(metadata: dict) -> str:
    g = metadata["generator"]["resolved"]
    rows = metadata["corpus"].get("rows_by_split", {})
    table = "\n".join(f"| {s} | {rows.get(s, 'n/a'):,} |" if isinstance(rows.get(s), int)
                      else f"| {s} | n/a |" for s in SPLITS)
    columns = "\n".join(f"- `{field.name}`: {field.type}" for field in SCHEMA)
    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/*.parquet
  - split: validation
    path: data/validation/*.parquet
  - split: test
    path: data/test/*.parquet
tags:
- time-series
- symbolic-regression
- tsfmxai
---

# Canonical recurrence-tree corpus

Random recurrence trees (ASTs over `x(t-lag)`) paired with their canonical polynomial
form, generated by `tsfmxai.generators.CombinatorialTreeSampler` and encoded by
`CanonicalCodec`. Repository commit: `{metadata["provenance"]["git_commit"]}`.

## Generator parameters

- Operators per tree: {g["minimum_operators"]} to {g["maximum_operators"]}; series: {g["num_series"]};
  lags {g["minimum_lag"]} to {g["maximum_lag"]}; maximum degree {g["max_degree"]}.
- Coefficient magnitude in [{g["minimum_parameter_magnitude"]}, {g["maximum_parameter_magnitude"]}].
- Sampled trees: {metadata["generator"]["sampled_asts"]:,}; protection probability
  {metadata["generator"]["protection_probability"]}.
- Seeds: data {metadata["generator"]["data_seed"]}, split {metadata["generator"]["split_seed"]}.
- Protection: {metadata["generator"]["protection_rule"]}
- Split: {metadata["generator"]["split_rule"]}
  Validation and test keep one row per skeleton.

The complete parameter set, with defaults resolved, is in `metadata.json`.

## Rows

| split | rows |
|---|---|
{table}

## Columns

{columns}
"""


def package_release(run_dir, output_dir, *, repo_root=None) -> dict:
    """Write the release directory for a completed run and return its manifest."""
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("Packaging needs safetensors: pip install -e '.[release]'") from error
    import torch

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is missing: the run did not complete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["provenance"]["config"]
    git_commit = _git("rev-parse", "HEAD", cwd=repo_root)
    provenance = {
        "git_commit": git_commit,
        "git_dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        "config_sha256": manifest["provenance"].get("config_sha256"),
        "source_sha256": manifest["provenance"].get("source_sha256"),
        "python": platform.python_version(), "torch": torch.__version__,
        "experiment": config.get("experiment"),
    }
    generator = generator_metadata(config)
    codec = codec_metadata(config)
    files: list[Path] = []

    for variant in config["variants"]:
        name = variant["name"]
        checkpoint = torch.load(run_dir / "checkpoints" / name / "best.pt",
                                map_location="cpu", weights_only=True)
        target = output_dir / "models" / name
        target.mkdir(parents=True, exist_ok=True)
        state = {key: value.contiguous() for key, value in checkpoint["model"].items()}
        save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})
        result = dict(manifest["results"][name])
        history = result.pop("history", [])
        result.pop("checkpoint", None)  # absolute path of the training machine
        model_config = {
            "architecture": "tsfmxai.symbolic_regression.canonical_models.CanonicalEncoderDecoder",
            "model_kwargs": {"vocab_size": codec["vocab_size"], "encoder": variant["encoder"],
                             "predict_coefficients": variant["coefficients"], **config["model"]},
            "codec": codec, "generator": generator,
            "training": {k: config[k] for k in (
                "batch_size", "epochs", "patience", "learning_rate", "weight_decay",
                "coefficient_weight", "gradient_clip", "min_delta", "max_decode_steps",
                "training_seed", "coefficient_atol", "coefficient_rtol") if k in config},
            "best_epoch": checkpoint["epoch"], "evaluation": result,
            "corpus": manifest.get("corpus"), "provenance": provenance,
        }
        _write_json(target / "config.json", model_config)
        _write_json(target / "training_history.json", history)
        (target / "README.md").write_text(_model_card(name, model_config), encoding="utf-8")
        files += [target / n for n in ("model.safetensors", "config.json",
                                       "training_history.json", "README.md")]

    dataset = output_dir / "dataset"
    for split in SPLITS:
        for source in sorted((run_dir / "parquet" / split).glob("part-*.parquet")):
            destination = dataset / "data" / split / source.name
            _link_or_copy(source, destination)
            files.append(destination)
    dataset_metadata = {"generator": generator, "codec": codec,
                        "corpus": manifest.get("corpus"), "provenance": provenance,
                        "columns": {field.name: str(field.type) for field in SCHEMA}}
    _write_json(dataset / "metadata.json", dataset_metadata)
    (dataset / "README.md").write_text(_dataset_card(dataset_metadata), encoding="utf-8")
    files += [dataset / "metadata.json", dataset / "README.md"]

    checksums = {str(path.relative_to(output_dir)).replace(os.sep, "/"): _sha256_file(path)
                 for path in files}
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())), encoding="utf-8")
    release = {"files": {name: {"sha256": digest, "bytes": (output_dir / name).stat().st_size}
                         for name, digest in sorted(checksums.items())},
               "provenance": provenance}
    _write_json(output_dir / "release_manifest.json", release)
    return release
