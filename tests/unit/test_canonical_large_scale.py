"""Invariants of the streamed Parquet corpus used for large training runs."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
from tsfmxai.symbolic_regression.canonical_large_scale import (
    count_corpus, deduplicate_evaluation_in_place, generate_parquet,
    iter_shard_examples, make_unique_evaluation_corpus,
    split_for_skeleton,
)


@pytest.mark.parametrize('num_series', [1, 6])
def test_large_corpus_count_split_and_roundtrip(tmp_path, num_series):
    config = {
        "ast_count": 240, "shard_size": 60, "generation_workers": 1,
        "data_seed": 17, "split_seed": 19,
        "generator": {"minimum_operators": 1, "maximum_operators": 4,
                      "num_series": num_series, "minimum_lag": 1, "maximum_lag": 3,
                      "max_degree": 3},
    }
    states = []
    codec = generate_parquet(config, tmp_path, lambda state, **kwargs: states.append(state))
    assert isinstance(codec, CanonicalCodec)
    assert codec.num_series == num_series
    assert len({codec.atom_token(i, 1) for i in range(num_series)}) == num_series
    summary = count_corpus(config, tmp_path, lambda state, **kwargs: states.append(state))
    assert summary["sampled_asts"] == 240
    assert summary["skeleton_overlap"] == 0
    assert sum(summary["rows_by_split"].values()) == 240
    source_series = set()
    has_cross_series_equation = False
    for split in ("train", "validation", "test"):
        for example in iter_shard_examples(tmp_path, split):
            assert split_for_skeleton(example["skeleton"], config["split_seed"]) == split
            assert example["target"][0] == 1
            assert example["target"][-1] == 2
            sources = {codec.token_atom(token)[0] for token in example["target"] if token >= 5}
            source_series.update(sources)
            has_cross_series_equation |= len(sources) > 1
    assert source_series == set(range(num_series))
    if num_series > 1:
        assert has_cross_series_equation
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


def test_in_place_evaluation_deduplication(tmp_path):
    config = {"ast_count": 240, "shard_size": 60, "generation_workers": 1,
              "data_seed": 17, "split_seed": 19, "unique_eval_skeletons": True,
              "generator": {"minimum_operators": 1, "maximum_operators": 4,
                            "num_series": 1, "minimum_lag": 1, "maximum_lag": 3,
                            "max_degree": 3}}
    generate_parquet(config, tmp_path, lambda *args, **kwargs: None)
    before = sum(1 for _ in iter_shard_examples(tmp_path, "train"))
    deduplicate_evaluation_in_place(tmp_path, lambda *args, **kwargs: None)
    summary = count_corpus(config, tmp_path, lambda *args, **kwargs: None)
    assert summary["rows_by_split"]["train"] == before
    assert summary["rows_by_split"]["validation"] == summary["distinct_skeletons_by_split"]["validation"]
    assert summary["rows_by_split"]["test"] == summary["distinct_skeletons_by_split"]["test"]


def test_parquet_corpus_contains_both_ast_variants(tmp_path):
    config = {"ast_count": 240, "shard_size": 60, "generation_workers": 1,
              "data_seed": 17, "split_seed": 19, "protection_probability": 0.5,
              "generator": {"minimum_operators": 1, "maximum_operators": 4,
                            "num_series": 1, "minimum_lag": 1, "maximum_lag": 3,
                            "max_degree": 3}}
    codec = generate_parquet(config, tmp_path, lambda *args, **kwargs: None)
    examples = [example for split in ("train", "validation", "test")
                for example in iter_shard_examples(tmp_path, split)]
    assert len(examples) == 240
    assert 0 < sum(e["protection_selected"] for e in examples) < 240
    assert all(e["tanh_count"] == 0 for e in examples if not e["protection_selected"])
    for e in examples:
        decoded = codec.decode(e["target"][1:], e["coefficient_targets"][1:])
        assert decoded["valid"] and decoded["skeleton"] == e["skeleton"]

    second = tmp_path / "parallel"
    generate_parquet({**config, "generation_workers": 2}, second,
                     lambda *args, **kwargs: None)
    other = [example for split in ("train", "validation", "test")
             for example in iter_shard_examples(second, split)]
    identity = lambda e: (e["sample_index"], e["protection_selected"],
                          e["skeleton"], e["target"])
    assert sorted(map(identity, examples)) == sorted(map(identity, other))


def test_multiseries_source_encodes_executed_equations_of_the_forecasting_generator(tmp_path):
    from tsfmxai.generators.multiseries import sample_multiseries_system
    from tsfmxai.generators.random_expression import derive_sample_seed
    from tsfmxai.symbolic_regression.canonical_data import (
        multiseries_teacher_config, sample_multiseries_example)
    generator = {"source": "multiseries", "num_series": 6, "minimum_operators": 1,
                 "maximum_operators": 3, "minimum_lag": 1, "maximum_lag": 6, "max_degree": 4,
                 "minimum_parameter_magnitude": .05, "maximum_parameter_magnitude": 1.,
                 "dependency": "random", "max_external_sources": 2, "self_source_weight": 3.}
    config = {"ast_count": 120, "shard_size": 40, "generation_workers": 1,
              "data_seed": 23, "split_seed": 29, "max_decode_steps": 64,
              "model": {"max_nodes": 64}, "generator": generator}
    codec = generate_parquet(config, tmp_path, lambda *args, **kwargs: None)
    summary = count_corpus(config, tmp_path, lambda *args, **kwargs: None)
    assert sum(summary["rows_by_split"].values()) == 120
    treatments = {}
    for split_counts in summary["treatments_by_split"].values():
        for name, count in split_counts.items():
            treatments[name] = treatments.get(name, 0) + count
    assert set(treatments) <= {"shao_wu", "jacobian", "heaps"} and "shao_wu" in treatments
    example = sample_multiseries_example(config, codec, 5)
    system = sample_multiseries_system(multiseries_teacher_config(generator),
                                       seed=derive_sample_seed(23, 5))
    assert example["skeleton"] in [codec.encode(e)["skeleton"] for e in system.equations]
    with pytest.raises(ValueError, match="max_nodes"):
        generate_parquet({**config, "model": {"max_nodes": 4}}, tmp_path / "small",
                         lambda *args, **kwargs: None)


def test_all_channels_per_system_encode_every_channel_and_match_uncached(tmp_path):
    from tsfmxai.generators.multiseries import sample_multiseries_system
    from tsfmxai.generators.random_expression import derive_sample_seed
    from tsfmxai.symbolic_regression.canonical_data import (
        multiseries_teacher_config, sample_multiseries_example)
    generator = {"source": "multiseries", "num_series": 6, "minimum_operators": 1,
                 "maximum_operators": 3, "minimum_lag": 1, "maximum_lag": 6, "max_degree": 4,
                 "minimum_parameter_magnitude": .05, "maximum_parameter_magnitude": 1.,
                 "dependency": "random", "max_external_sources": 2, "self_source_weight": 3.}
    # 40 is not a multiple of 6: a system that straddles two shards is sampled in both.
    config = {"ast_count": 120, "shard_size": 40, "generation_workers": 2,
              "channels_per_system": 6, "data_seed": 23, "split_seed": 29,
              "max_decode_steps": 64, "model": {"max_nodes": 64}, "generator": generator}
    codec = generate_parquet(config, tmp_path, lambda *args, **kwargs: None)
    rows = sorted((e for split in ("train", "validation", "test")
                   for e in iter_shard_examples(tmp_path, split)),
                  key=lambda e: e["sample_index"])
    assert [e["sample_index"] for e in rows] == list(range(120))
    multiseries = multiseries_teacher_config(generator)
    for system_index in range(20):
        system = sample_multiseries_system(multiseries, seed=derive_sample_seed(23, system_index))
        expected = [codec.encode(e)["skeleton"] for e in system.equations]
        assert [rows[6 * system_index + c]["skeleton"] for c in range(6)] == expected
    cache = {}
    cached = [sample_multiseries_example(config, codec, i, multiseries, cache) for i in range(12)]
    fresh = [sample_multiseries_example(config, codec, i, multiseries) for i in range(12)]
    assert [e["target"] for e in cached] == [e["target"] for e in fresh]
    with pytest.raises(ValueError, match="channels_per_system"):
        sample_multiseries_example({**config, "channels_per_system": 3}, codec, 0)


def test_production_training_can_iterate_over_every_split(tmp_path):
    config = {"ast_count": 120, "shard_size": 40, "generation_workers": 1,
              "data_seed": 17, "split_seed": 19,
              "generator": {"minimum_operators": 1, "maximum_operators": 3, "num_series": 2,
                            "minimum_lag": 1, "maximum_lag": 3, "max_degree": 3}}
    generate_parquet(config, tmp_path, lambda *args, **kwargs: None)
    every = list(iter_shard_examples(tmp_path, ["train", "validation", "test"], seed=3))
    assert len(every) == 120
    assert len({example["sample_index"] for example in every}) == 120
