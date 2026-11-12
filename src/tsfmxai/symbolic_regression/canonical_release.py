"""Package a finished canonicalization run in a layout ready for Hugging Face.

The training pipeline leaves checkpoints with optimizer state, absolute paths and
the corpus under ``parquet/``. :func:`package_release` turns that run directory
into a self-contained release directory::

    release/
      models/<variant>/model.safetensors   weights only (encoder and decoder)
      models/<variant>/encoder.safetensors encoder weights only (CanonicalEncoder)
      models/<variant>/config.json         architecture, codec, generator, metrics
      models/<variant>/training_history.json
      models/<variant>/README.md           model card
      models/source/                       code that loads the models (release commit)
      dataset/data/{train,validation,test}/part-*.parquet
      dataset/metadata.json                generator parameters, counts, splits
      dataset/README.md                    dataset card (with the viewer config)
      dataset/source/                      generator code and config (training commit)
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
from tsfmxai.symbolic_regression.canonical_models import encoder_state

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


LOADING_PATHS = ("src", "pyproject.toml", "constraints.txt")
GENERATION_PATHS = ("src", "pyproject.toml", "constraints.txt", "scripts/run_canonical_5m.py")


def _git_bytes(*args: str, cwd: Path) -> bytes | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def _repository_url(repo_root: Path) -> str | None:
    url = _git("remote", "get-url", "origin", cwd=repo_root)
    if url is None:
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    return url.removesuffix(".git")


def _verify_training_commit(repo_root: Path, commit: str, source_sha256: dict) -> int:
    """Check that the recorded source hashes are those of ``commit``; return how many matched."""
    for name, expected in (source_sha256 or {}).items():
        blob = _git_bytes("show", f"{commit}:{name}", cwd=repo_root)
        if blob is None or sha256(blob).hexdigest() != expected:
            raise ValueError(f"{name} at commit {commit[:10]} is not the file the run used.")
    return len(source_sha256 or {})


def _export_source(repo_root: Path, commit: str, paths, destination: Path) -> list[Path]:
    """Write the tracked files under ``paths`` at ``commit``, byte-exact (blobs, no EOL conversion)."""
    listing = _git_bytes("ls-tree", "-r", "--name-only", "-z", commit, "--", *paths, cwd=repo_root)
    if listing is None:
        raise FileNotFoundError(f"git ls-tree failed for commit {commit}.")
    written = []
    for name in listing.decode("utf-8").split("\0"):
        if not name or "__pycache__" in name:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git_bytes("show", f"{commit}:{name}", cwd=repo_root))
        written.append(target)
    return sorted(written)


def _provenance_lines(provenance: dict, source_dir: str) -> str:
    repo, train, release = (provenance.get(k) for k in
                            ("repository", "training_commit", "release_commit"))

    def link(commit):
        return f"[`{commit[:10]}`]({repo}/commit/{commit})" if repo else f"`{commit}`"

    rows = []
    if train:
        rows.append(f"- Trained and generated with repository commit {link(train)}"
                    f" ({provenance.get('source_verified_files', 0)} source files verified"
                    " against the hashes recorded during the run).")
    if release and release != train:
        rows.append(f"- Loading code (`CanonicalEncoder`, release tooling) from commit {link(release)}.")
    if repo:
        rows.append(f"- Repository: {repo}")
    rows.append(f"- Config SHA-256: `{provenance.get('config_sha256')}`; "
                f"python {provenance.get('python')}, torch {provenance.get('torch')}, "
                f"device {provenance.get('device')}.")
    rows.append(f"- A byte-exact copy of the code is in `{source_dir}/`.")
    return "\n".join(rows)


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
license: cc-by-4.0
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
(`CanonicalEncoderDecoder`) of the tsfmxai repository.

## Code and data provenance

{_provenance_lines(config_json["provenance"], "models/source")}

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

## Files

| file | content |
|---|---|
| `model.safetensors` | full encoder-decoder weights |
| `encoder.safetensors` | encoder weights only (`CanonicalEncoder`), for feature extraction |
| `config.json` | architecture arguments, codec, generator parameters, seeds, metrics |


## License and credit

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): anyone may use,
share and adapt it, including commercially, as long as they credit the author. Please cite
as: Mateus Witeck, "tsfmxai: canonical recurrence-tree encoder-decoder models", {config_json["provenance"].get("repository")}.

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

### Encoder only

```python
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoder

encoder = CanonicalEncoder.from_pretrained(".")   # reads config.json and encoder.safetensors
memory, mask, root = encoder(batch)
```

`batch` is the dictionary produced by `collate_canonical`. `memory` has shape
(batch, nodes, {config_json["model_kwargs"]["dimension"]}) with one vector per tree node, `mask` is
true on padding, and `root` is the (batch, {config_json["model_kwargs"]["dimension"]}) summary of the whole tree. The encoder accepts trees of at
most {config_json["model_kwargs"]["max_nodes"]} nodes (`max_nodes`, the size of the position table).
To fetch only the encoder, download `encoder.safetensors` and `config.json` of the variant.

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
license: cc-by-4.0
tags:
- time-series
- symbolic-regression
- tsfmxai
---

# Canonical recurrence-tree corpus

Random recurrence trees (ASTs over `x(t-lag)`) paired with their canonical polynomial
form, generated by `tsfmxai.generators.CombinatorialTreeSampler` and encoded by
`CanonicalCodec`.

## Generator code and provenance

{_provenance_lines(metadata["provenance"], "source")}

To regenerate the corpus, check out the training commit and run the training script
with the configuration stored in `metadata.json` (key `training_config`) or
`source/training_config.json`:

```
git clone {metadata["provenance"].get("repository") or "<repository>"} tsfmxai && cd tsfmxai
git checkout {metadata["provenance"].get("training_commit") or "<training commit>"}
pip install -e ".[learning]" -c constraints.txt
python scripts/run_canonical_5m.py --config configs/canonicalization_5m_raw_tanh.json
```

`source/` holds a byte-exact copy of that code, so the generator stays available even if
the repository moves. The generator itself is `tsfmxai.generators.CombinatorialTreeSampler`
configured with `generator.resolved` of `metadata.json`.

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


## License and credit

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): anyone may use,
share and adapt it, including commercially, as long as they credit the author. Please cite
as: Mateus Witeck, "tsfmxai: canonical recurrence-tree corpus", {metadata["provenance"].get("repository")}.

## Rows

| split | rows |
|---|---|
{table}

## Columns

{columns}
"""


def package_release(run_dir, output_dir, *, repo_root=None, training_commit=None,
                    allow_dirty=False) -> dict:
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
    release_commit = _git("rev-parse", "HEAD", cwd=repo_root)
    dirty = bool(_git("status", "--porcelain", cwd=repo_root)) if release_commit else False
    if dirty and not allow_dirty:
        raise RuntimeError("The working tree has uncommitted changes; commit them first so the "
                           "recorded release commit shows the packaged code (or pass allow_dirty).")
    training_commit = (_git("rev-parse", training_commit, cwd=repo_root)
                       if training_commit else release_commit)
    recorded = manifest["provenance"]
    verified = (_verify_training_commit(repo_root, training_commit, recorded.get("source_sha256"))
                if training_commit else 0)
    provenance = {
        "repository": _repository_url(repo_root),
        "training_commit": training_commit, "release_commit": release_commit,
        "source_verified_files": verified,
        "config_sha256": recorded.get("config_sha256"),
        "source_sha256": recorded.get("source_sha256"),
        "python": recorded.get("python"), "torch": recorded.get("torch"),
        "device": recorded.get("device"),
        "training_elapsed_seconds": recorded.get("elapsed_seconds"),
        "training_finished_utc": recorded.get("generated_at_utc"),
        "packaged_with": {"python": platform.python_version(), "torch": torch.__version__},
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
        save_file(encoder_state(state), str(target / "encoder.safetensors"), metadata={"format": "pt"})
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
        files += [target / n for n in ("model.safetensors", "encoder.safetensors", "config.json",
                                       "training_history.json", "README.md")]

    dataset = output_dir / "dataset"
    if (run_dir / "parquet").exists():
        for split in SPLITS:
            for source in sorted((run_dir / "parquet" / split).glob("part-*.parquet")):
                destination = dataset / "data" / split / source.name
                _link_or_copy(source, destination)
                files.append(destination)
    elif (dataset / "data").exists():
        # The Parquet corpus was already moved into the release and the run folder holds
        # only models and logs: repackage the models and keep the dataset files as they are.
        files += sorted(path for path in (dataset / "data").rglob("part-*.parquet"))
    else:
        raise FileNotFoundError(f"{run_dir / 'parquet'} is missing and the release has no dataset.")
    if training_commit:
        files += _export_source(repo_root, training_commit, GENERATION_PATHS, dataset / "source")
        files += _export_source(repo_root, release_commit, LOADING_PATHS, output_dir / "models" / "source")
    _write_json(dataset / "source" / "training_config.json", config)
    files.append(dataset / "source" / "training_config.json")
    dataset_metadata = {"generator": generator, "codec": codec, "training_config": config,
                        "corpus": manifest.get("corpus"), "provenance": provenance,
                        "columns": {field.name: str(field.type) for field in SCHEMA}}
    _write_json(dataset / "metadata.json", dataset_metadata)
    (dataset / "README.md").write_text(_dataset_card(dataset_metadata), encoding="utf-8")
    files += [dataset / "metadata.json", dataset / "README.md"]

    checksums = {str(path.relative_to(output_dir)).replace(os.sep, "/"): _sha256_file(path)
                 for path in files}
    (output_dir / "SHA256SUMS").write_bytes(  # LF endings so `sha256sum -c` works on Linux
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())).encode("utf-8"))
    release = {"files": {name: {"sha256": digest, "bytes": (output_dir / name).stat().st_size}
                         for name, digest in sorted(checksums.items())},
               "provenance": provenance}
    _write_json(output_dir / "release_manifest.json", release)
    return release
