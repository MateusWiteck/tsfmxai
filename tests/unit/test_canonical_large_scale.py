"""Invariants of the streamed Parquet corpus used for large training runs."""

import pyarrow as pa
import pyarrow.parquet as pq

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_large_scale import (
    count_corpus, generate_parquet, iter_shard_examples, make_unique_evaluation_corpus,
    split_for_skeleton,
)


def test_large_corpus_count_split_and_roundtrip(tmp_path):
    config = {
        "ast_count": 240, "shard_size": 60, "generation_workers": 1,
        "data_seed": 17, "split_seed": 19,
        "generator": {"minimum_operators": 1, "maximum_operators": 4,
                      "num_series": 1, "minimum_lag": 1, "maximum_lag": 3,
                      "max_degree": 3},
    }
    states = []
    codec = generate_parquet(config, tmp_path, lambda state, **kwargs: states.append(state))
    assert isinstance(codec, CanonicalCodec)
    summary = count_corpus(config, tmp_path, lambda state, **kwargs: states.append(state))
    assert summary["sampled_asts"] == 240
    assert summary["skeleton_overlap"] == 0
    assert sum(summary["rows_by_split"].values()) == 240
    for split in ("train", "validation", "test"):
        for example in iter_shard_examples(tmp_path, split):
            assert split_for_skeleton(example["skeleton"], config["split_seed"]) == split
            assert example["target"][0] == 1
            assert example["target"][-1] == 2
    assert "generating" in states and "counting" in states


def test_evaluation_deduplication_keeps_every_training_row(tmp_path):
    source, target = tmp_path / "source", tmp_path / "unique"
    config = {"ast_count": 240, "shard_size": 60, "generation_workers": 1,
              "data_seed": 17, "split_seed": 19,
              "generator": {"minimum_operators": 1, "maximum_operators": 4,
                            "num_series": 1, "minimum_lag": 1, "maximum_lag": 3,
                            "max_degree": 3}}
    generate_parquet(config, source, lambda *args, **kwargs: None)
    validation_file = sorted((source / "parquet" / "validation").glob("*.parquet"))[0]
    table = pq.read_table(validation_file)
    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), validation_file)
    make_unique_evaluation_corpus(source, target, lambda *args, **kwargs: None)
    stored = sum(pq.ParquetFile(file).metadata.num_rows for file in
                 (target / "parquet").rglob("*.parquet"))
    summary = count_corpus({**config, "expected_rows": stored,
                            "unique_eval_skeletons": True}, target,
                           lambda *args, **kwargs: None)
    assert summary["rows_by_split"]["train"] == sum(
        1 for _ in iter_shard_examples(source, "train"))
    assert summary["rows_by_split"]["validation"] == summary["distinct_skeletons_by_split"]["validation"]
    assert summary["rows_by_split"]["test"] == summary["distinct_skeletons_by_split"]["test"]
