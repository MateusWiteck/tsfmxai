"""Stream a large canonical AST corpus to Parquet and train from its shards."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import math
import random
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig, CombinatorialTreeSampler
from tsfmxai.generators.random_expression import derive_sample_seed
from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec, skeleton_digest
from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoderDecoder, collate_canonical, supervised_loss
from tsfmxai.symbolic_regression.canonical_training import atomic_json, evaluate_greedy, evaluate_teacher, utc_now


SPLITS = ("train", "validation", "test")
SCHEMA = pa.schema([
    ("sample_index", pa.int64()), ("skeleton_hash", pa.binary(32)),
    ("skeleton_json", pa.string()), ("formula", pa.string()),
    ("operator_count", pa.int16()), ("parameter_count", pa.int16()), ("degree", pa.int16()),
    *[(name, pa.list_(pa.int16())) for name in
      ("prefix_ids", "tree_ids", "left_children", "right_children", "target")],
    *[(name, pa.list_(pa.float64())) for name in
      ("prefix_values", "tree_values", "coefficient_targets", "coefficients")],
])


def split_for_skeleton(skeleton, split_seed):
    digest = skeleton_digest(skeleton)
    bucket = int.from_bytes(sha256(f"{split_seed}|{digest}".encode()).digest()[:8], "big") % 10_000
    return "train" if bucket < 7000 else "validation" if bucket < 8000 else "test"


def _row(example):
    return {"sample_index": example["sample_index"],
            "skeleton_hash": bytes.fromhex(skeleton_digest(example["skeleton"])),
            "skeleton_json": json.dumps(example["skeleton"], separators=(",", ":")),
            "formula": example["formula"],
            **{name: example[name] for name in (
                "operator_count", "parameter_count", "degree", "prefix_ids", "tree_ids",
                "left_children", "right_children", "target", "prefix_values", "tree_values",
                "coefficient_targets", "coefficients")}}


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary,
                   compression="zstd", compression_level=3)
    temporary.replace(path)


def _files(run_dir, split):
    return sorted((Path(run_dir) / "parquet" / split).glob("part-*.parquet"))


def make_unique_evaluation_corpus(source_dir, run_dir, status):
    """Retain every training row and one deterministic row per evaluation support."""
    source_dir, run_dir = Path(source_dir), Path(run_dir)
    if not _files(source_dir, "train") or not _files(source_dir, "validation") or not _files(source_dir, "test"):
        raise FileNotFoundError("The original complete Parquet corpus is required.")
    for source in _files(source_dir, "train"):
        target = run_dir / "parquet" / "train" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    status("deduplicating", deduplicated_split="train")
    for split in ("validation", "test"):
        seen = set()
        # Rebuild evaluation partitions from source on each run; output is deterministic.
        for source in _files(source_dir, split):
            table = pq.read_table(source)
            hashes = table["skeleton_hash"].to_pylist()
            selected = []
            for index, digest in enumerate(hashes):
                if digest not in seen:
                    seen.add(digest)
                    selected.append(index)
            target = run_dir / "parquet" / split / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            pq.write_table(table.take(pa.array(selected, type=pa.int64())), temporary,
                           compression="zstd", compression_level=3)
            temporary.replace(target)
        status("deduplicating", deduplicated_split=split, unique_skeletons=len(seen))


def _generate_shard(config, run_dir, shard):
    run_dir = Path(run_dir)
    generator = CombinatorialTreeSampler(CombinatorialTreeConfig(**config["generator"]))
    g = config["generator"]
    codec = CanonicalCodec(g["num_series"], g["minimum_lag"], g["maximum_lag"], g["max_degree"])
    size, shard_size = config["ast_count"], config["shard_size"]
    paths = {split: run_dir / "parquet" / split / f"part-{shard:05d}.parquet"
             for split in SPLITS}
    if all(path.exists() for path in paths.values()):
        return shard, min(size, (shard + 1) * shard_size) - shard * shard_size
    rows = {split: [] for split in SPLITS}
    for index in range(shard * shard_size, min(size, (shard + 1) * shard_size)):
        seed = derive_sample_seed(config["data_seed"], index)
        example = codec.encode(generator.sample(seed=seed, sample_index=index))
        rows[split_for_skeleton(example["skeleton"], config["split_seed"])].append(_row(example))
    for split in SPLITS:
        _write_rows(paths[split], rows[split])
    return shard, min(size, (shard + 1) * shard_size) - shard * shard_size


def generate_parquet(config, run_dir, status):
    """Generate exactly N sampled ASTs with independent, resumable shards."""
    g = config["generator"]
    codec = CanonicalCodec(g["num_series"], g["minimum_lag"], g["maximum_lag"], g["max_degree"])
    size, shard_size = config["ast_count"], config["shard_size"]
    start_time = time.perf_counter()
    total_shards = math.ceil(size / shard_size)
    with ProcessPoolExecutor(max_workers=config["generation_workers"]) as pool:
        futures = [pool.submit(_generate_shard, config, run_dir, shard)
                   for shard in range(total_shards)]
        generated = 0
        for future in as_completed(futures):
            _, count = future.result()
            generated += count
            status("generating", generated=generated, total=size,
                   completed_shards=math.ceil(generated / shard_size),
                   total_shards=total_shards,
                   generation_elapsed_seconds=time.perf_counter() - start_time)
    return codec


def count_corpus(config, run_dir, status):
    """Exact distinct-support count via SHA-256 bytes; also verify split isolation."""
    keys = {split: set() for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    frequencies = {split: defaultdict(int) for split in SPLITS}
    for split in SPLITS:
        for file in _files(run_dir, split):
            table = pq.read_table(file, columns=["skeleton_hash", "operator_count"])
            hashes = table["skeleton_hash"].to_pylist()
            operators = table["operator_count"].to_pylist()
            keys[split].update(hashes)
            counts[split] += len(hashes)
            for n in operators:
                frequencies[split][n] += 1
        status("counting", counted_split=split, counted=sum(counts.values()))
    stored = sum(counts.values())
    if stored != config.get("expected_rows", config["ast_count"]):
        raise RuntimeError("Parquet row count does not equal the expected count.")
    if config.get("unique_eval_skeletons") and any(
            counts[split] != len(keys[split]) for split in ("validation", "test")):
        raise RuntimeError("Validation or test still contains duplicate skeletons.")
    if any(keys[a] & keys[b] for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise RuntimeError("A skeleton occurs in multiple partitions.")
    return {"sampled_asts": config["ast_count"], "stored_examples": stored,
            "rows_by_split": counts,
            "distinct_skeletons_by_split": {k: len(v) for k, v in keys.items()},
            "distinct_skeletons_total": sum(map(len, keys.values())),
            "operator_counts_by_split": {k: dict(v) for k, v in frequencies.items()},
            "skeleton_overlap": 0, "parquet_root": str((Path(run_dir) / "parquet").resolve())}


def _example(row):
    row["skeleton"] = tuple(tuple(tuple(atom) for atom in term)
                            for term in json.loads(row.pop("skeleton_json")))
    row.pop("skeleton_hash")
    return row


def iter_shard_examples(run_dir, split, seed=None):
    files = _files(run_dir, split)
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(files)
    for file in files:
        rows = pq.read_table(file).to_pylist()
        if seed is not None:
            rng.shuffle(rows)
        for row in rows:
            yield _example(row)


def _small_validation(run_dir, limit):
    from itertools import islice
    return list(islice(iter_shard_examples(run_dir, "validation"), limit))


@torch.no_grad()
def evaluate_parquet_test(model, run_dir, codec, config, device="cpu"):
    """Stream all test rows; metric uses free decoding and no target prefix."""
    model.eval()
    totals = defaultdict(int)
    batch = []

    def accumulate(examples):
        collated = {key: tensor.to(device) for key, tensor in collate_canonical(examples).items()}
        tokens, coefficients = model.generate(collated, config["max_decode_steps"])
        for i, truth in enumerate(examples):
            decoded = codec.decode(tokens[i].tolist(),
                                   coefficients[i].tolist() if coefficients is not None else None)
            expected, predicted = set(truth["skeleton"]), set(decoded["skeleton"])
            totals["count"] += 1
            totals["exact"] += int(decoded["valid"] and decoded["skeleton"] == truth["skeleton"])
            totals["valid"] += int(decoded["valid"])
            totals["tp"] += len(expected & predicted)
            totals["fp"] += len(predicted - expected)
            totals["fn"] += len(expected - predicted)
            if model.predict_coefficients:
                actual = dict(zip(truth["skeleton"], truth["coefficients"]))
                forecast = dict(zip(decoded["skeleton"], decoded["coefficients"])) if decoded["valid"] else {}
                joint = decoded["valid"] and decoded["skeleton"] == truth["skeleton"]
                for term in actual.keys() | forecast.keys():
                    error = forecast.get(term, 0.) - actual.get(term, 0.)
                    totals["union_squared"] += error * error
                    totals["union_count"] += 1
                for term in actual.keys() & forecast.keys():
                    totals["matched_absolute"] += abs(forecast[term] - actual[term])
                    totals["matched_count"] += 1
                totals["joint"] += int(joint and all(math.isclose(forecast[t], v,
                    abs_tol=config["coefficient_atol"], rel_tol=config["coefficient_rtol"])
                    for t, v in actual.items()))

    for row in iter_shard_examples(run_dir, "test"):
        batch.append(row)
        if len(batch) == config["batch_size"]:
            accumulate(batch)
            batch.clear()
    if batch:
        accumulate(batch)
    n = totals["count"]
    return {"count": n, "skeleton_exact": totals["exact"] / n,
            "valid_rate": totals["valid"] / n,
            "term_f1": 2 * totals["tp"] / max(1, 2 * totals["tp"] + totals["fp"] + totals["fn"]),
            "joint_exact": totals["joint"] / n if model.predict_coefficients else None,
            "coefficient_union_rmse": math.sqrt(totals["union_squared"] / max(1, totals["union_count"]))
               if model.predict_coefficients else None,
            "coefficient_matched_mae": totals["matched_absolute"] / max(1, totals["matched_count"])
               if model.predict_coefficients else None}


def train_parquet_variant(config, run_dir, codec, variant, status, device="cpu"):
    """Train over every shard on the selected device; checkpoint each epoch."""
    run_dir = Path(run_dir)
    name = variant["name"]
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    torch.manual_seed(config["training_seed"] + int(variant["coefficients"]))
    model = CanonicalEncoderDecoder(codec.vocab_size, encoder=variant["encoder"],
        predict_coefficients=variant["coefficients"], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"])
    last = run_dir / "checkpoints" / name / "last.pt"
    best = last.with_name("best.pt")
    last.parent.mkdir(parents=True, exist_ok=True)
    history, best_loss, bad_epochs, start_epoch = [], math.inf, 0, 1
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        torch.set_rng_state(checkpoint["rng"])
        history, best_loss = checkpoint["history"], checkpoint["best_loss"]
        bad_epochs, start_epoch = checkpoint["bad_epochs"], checkpoint["epoch"] + 1
    validation = _small_validation(run_dir, config["validation_examples_per_epoch"])
    for epoch in range(start_epoch, config["epochs"] + 1):
        if bad_epochs >= config["patience"]:
            break
        model.train()
        running, steps, rows_seen = 0., 0, 0
        batch = []
        started = time.perf_counter()
        status("training", active_variant=name, epoch=epoch, training_rows_seen=0)
        def train_batch(examples):
            nonlocal running, steps, rows_seen
            optimizer.zero_grad(set_to_none=True)
            collated = {key: tensor.to(device) for key, tensor in collate_canonical(examples).items()}
            loss, _ = supervised_loss(model, collated, config["coefficient_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            rows_seen += len(examples)
            if steps % 500 == 0:
                status("training", active_variant=name, epoch=epoch,
                       training_rows_seen=rows_seen, train_batches=steps)
        for example in iter_shard_examples(run_dir, "train", seed=config["training_seed"] + epoch):
            batch.append(example)
            if len(batch) == config["batch_size"]:
                train_batch(batch)
                batch.clear()
        if batch:
            train_batch(batch)
        teacher = evaluate_teacher(model, validation, config, device=device)
        greedy, _, _ = evaluate_greedy(model, validation[:config["validation_greedy_size"]], codec, config, device=device)
        row = {"epoch": epoch, "train_loss": running / steps,
               "validation_loss": teacher["loss"], "validation_skeleton_exact": greedy["skeleton_exact"],
               "rows_seen": rows_seen, "seconds": time.perf_counter() - started}
        history.append(row)
        if teacher["loss"] < best_loss - config["min_delta"]:
            best_loss, bad_epochs = teacher["loss"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": variant}, best)
        else:
            bad_epochs += 1
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(), "history": history, "epoch": epoch,
                    "best_loss": best_loss, "bad_epochs": bad_epochs}, tmp)
        tmp.replace(last)
        status("training", active_variant=name, epoch=epoch, last_epoch=row,
               training_rows_seen=rows_seen)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    status("testing", active_variant=name, best_epoch=checkpoint["epoch"])
    result = evaluate_parquet_test(model, run_dir, codec, config, device=device)
    result.update({"best_epoch": checkpoint["epoch"], "history": history,
                   "checkpoint": str(best.resolve())})
    return result
