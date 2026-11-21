"""Checks for the two prospective baseline contracts."""
import json
from pathlib import Path

import pytest

from tsfmxai.pipelines.prior_baselines import (PriorProtocol, RepeatedBatches, own_plan,
                                               _compare_resume_environment, _cycle,
                                               prepare_corpus, run)


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/prior_baseline_example.json"
HISTORICAL = CONFIG.with_name("prior_baseline_lmc_20261118.json")


def small_protocol(tmp_path):
    values = json.loads(CONFIG.read_text(encoding="utf-8"))
    values.update(controlled_updates=2, workers=0, checkpoint_interval=1,
                  validation_batches=1, max_updates_per_stage=2,
                  time_hours=0.0003, own_generated_steps=192)
    return PriorProtocol(**values)


def test_forecasting_generator_matches_declared_distribution():
    protocol = PriorProtocol.read(CONFIG)
    plan = own_plan(protocol, "independent", 1)
    config = plan.stages[0].config
    assert config.self_only_treatment_probs == {"jacobian": .5, "heaps": .5}
    assert config.ast_config.minimum_parameter_magnitude == 0.05
    assert config.ast_config.maximum_parameter_magnitude == 1.0
    assert config.ast_config.maximum_operators == 6
    assert config.ast_config.max_degree == 4
    assert plan.trajectory.generated_steps == 192
    assert plan.trajectory.initialization.kind == "random"
    assert (plan.training.context_length, plan.training.prediction_length) == (96, 96)
    assert config.dependency_probs == {"random": 1.0}
    assert config.max_external_sources == 2
    later = own_plan(protocol, "correlated", 1).stages[0].config
    assert later.dependency_probs == config.dependency_probs
    assert later.max_external_sources == config.max_external_sources


def test_historical_lmc_configuration_keeps_its_original_hash():
    import hashlib
    assert hashlib.sha256(HISTORICAL.read_bytes()).hexdigest() == (
        "b18e5a493466307a33c516a9dd75fe00f7685b7572878b292ea9eda1912f60e4")
    historical = PriorProtocol.read(HISTORICAL)
    assert historical.own_self_only_treatment_probs == {"jacobian": .5, "heaps": .5}


@pytest.mark.parametrize("probability", [.25, .5])
def test_preliminary_hybrid_config_generates_training_batch(probability):
    from tsfmxai.datasets.online import OnlineBatchDataset
    from tsfmxai.generators.hybrid import HybridConfig
    path = CONFIG.with_name(f"preliminary_hybrid_p{int(probability * 100):02d}.json")
    protocol = PriorProtocol.read(path)
    plan = own_plan(protocol, "independent", 1)
    assert isinstance(plan.stages[0].config, HybridConfig)
    assert plan.stages[0].config.family_probs["kernel_synth"] == probability
    batch = OnlineBatchDataset(plan)[0]
    assert batch["context"].shape == batch["future"].shape == (16, 96, 6)
    assert batch["equation_mask"].shape[:2] == (16, 6)
    assert any("kernel_synth" in item["channel_families"] for item in batch["metadata"])


def tiny_teacher(tmp_path, max_nodes=20):
    from safetensors.torch import save_file
    from tsfmxai.symbolic_regression.canonical_data import CanonicalCodec
    from tsfmxai.symbolic_regression.canonical_models import CanonicalEncoder
    source = tmp_path / "teacher" / "transformer_skeleton"
    source.mkdir(parents=True)
    kwargs = dict(vocab_size=CanonicalCodec(6, 1, 16, 4).vocab_size,
                  dimension=8, heads=2, encoder_layers=1, max_nodes=max_nodes)
    source.joinpath("config.json").write_text(json.dumps({"model_kwargs": kwargs}))
    save_file({key: value.contiguous() for key, value in
               CanonicalEncoder(**kwargs).state_dict().items()},
              str(source / "encoder.safetensors"))
    return source


def test_equation_head_corpus_masks_kernel_channels_and_trains(tmp_path):
    source = tiny_teacher(tmp_path)
    values = small_protocol(tmp_path).__dict__
    protocol = PriorProtocol(**{**values, "own_kernel_probability": .5,
                                "backbone_channel_embedding": True})
    corpus = tmp_path / "corpus"
    prepare_corpus(protocol, "own", None, corpus, equation_head=True,
                   teacher_source=str(source.parent))
    with __import__("numpy").load(corpus / "independent" / "00000000.npz") as batch:
        assert batch["teacher_embedding"].shape == (16, 6, 8)
        assert batch["equation_target_mask"].shape == (16, 6)
        assert 0 < batch["equation_target_mask"].sum() < 96
        assert (batch["noise_mask"] >= batch["equation_target_mask"]).all()
        assert (batch["noise_std"][~batch["noise_mask"]] == 0).all()
        kernel = batch["kernel_mask"]
        assert kernel.shape == (16, 6) and kernel.any()
        assert not (kernel & batch["noise_mask"]).any()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert 0 < manifest["kernel_synth_fraction"] < 1
    assert set(manifest["channel_families"]) == {"ast", "kernel_synth"}
    coverage = manifest["auxiliary_target_coverage"]
    assert set(coverage) <= {"shao_wu", "jacobian", "heaps"}
    assert sum(c["covered"] for c in coverage.values()) == manifest["auxiliary_target_counts"]["skeleton"]
    result = run(protocol, "own", "controlled", None, tmp_path / "train",
                 corpus=corpus, device="cpu", tiny=True, equation_head=True,
                 teacher_source=str(source.parent))
    assert result["updates"] == 2
    assert (tmp_path / "train" / "model_final" / "auxiliary_heads.pt").exists()


def test_curriculum_and_mixed_control_reorder_same_examples():
    import hashlib
    from tsfmxai.datasets.online import OnlineBatchDataset
    base = small_protocol(None).__dict__
    signatures = []
    counts = []
    for policy in ("mixed", "curriculum"):
        protocol = PriorProtocol(**{**base, "own_difficulty_policy": policy})
        plan = own_plan(protocol, "independent", 6)
        assert [cfg.ast_config.maximum_operators for cfg in plan.difficulty_configs] == [2, 4, 6]
        assert all(cfg.channel_count_probs == {6: 1.0} for cfg in plan.difficulty_configs)
        batches = [OnlineBatchDataset(plan)[i]
                   for i in range(6)]
        counts.append([plan.difficulty_configs[min(2, plan._rank(i) * 3 // 6)]
                       .ast_config.maximum_operators for i in range(6)])
        signatures.append(sorted(hashlib.sha256(batch["context"].tobytes()).hexdigest()
                                 for batch in batches))
        assert all(batch["context"].shape == (16, 96, 6) for batch in batches)
        assert all(batch["channel_mask"].all() for batch in batches)
    assert signatures[0] == signatures[1]
    assert sorted(counts[0]) == sorted(counts[1]) == [2, 2, 4, 4, 6, 6]
    assert counts[1] == [2, 2, 4, 4, 6, 6]
    assert counts[0] != counts[1]


def test_curriculum_tiny_training_keeps_six_channel_tensors(tmp_path):
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "own_difficulty_policy": "curriculum"})
    corpus = tmp_path / "corpus"
    prepare_corpus(protocol, "own", None, corpus)
    with __import__("numpy").load(corpus / "independent" / "00000000.npz") as batch:
        assert batch["context"].shape == (16, 96, 6)
        assert batch["channel_mask"].sum(axis=1).tolist() == [6] * 16
    result = run(protocol, "own", "controlled", None, tmp_path / "train",
                 corpus=corpus, device="cpu", tiny=True)
    assert result["updates"] == 2


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
    assert result["environment"]["hostname"]
    assert result["environment"]["logical_cores"] > 0
    assert result["environment"]["disk_total_bytes"] > 0


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


def test_resume_before_first_update_without_telemetry(tmp_path):
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "time_hours": 1e-9})
    output = tmp_path / "pre_update"
    first = run(protocol, "own", "time", None, output, device="cpu", tiny=True)
    assert first["updates"] == 0
    assert not (output / "telemetry.jsonl").exists()
    resumed = run(protocol, "own", "time", None, output,
                  resume=True, device="cpu", tiny=True)
    assert resumed["updates"] == 0
    assert resumed["stop_reason"] == "budget_elapsed"


def test_resume_environment_ignores_nvidia_smi_failure(tmp_path, monkeypatch):
    import tsfmxai.pipelines.prior_baselines as baseline

    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "time_hours": 1e-9})
    output = tmp_path / "transient_environment"
    first = run(protocol, "own", "time", None, output, device="cpu", tiny=True)
    assert first["updates"] == 0
    original = baseline._environment

    def without_nvidia_smi(device, folder):
        environment = original(device, folder)
        environment["nvidia_driver"] = None
        environment["gpu_power_limit"] = None
        return environment

    # Simulate a successful hardware query in the checkpoint and a timeout
    # on resume without requiring an NVIDIA GPU in the unit test.
    saved = first["environment"]
    saved["nvidia_driver"] = "example-driver"
    saved["gpu_power_limit"] = "200 W"
    checkpoint = output / "latest.pt"
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["environment"] = saved
    torch.save(payload, checkpoint)
    monkeypatch.setattr(baseline, "_environment", without_nvidia_smi)
    resumed = run(protocol, "own", "time", None, output,
                  resume=True, device="cpu", tiny=True)
    assert resumed["updates"] == 0
    events = [json.loads(line) for line in (output / "resume_events.jsonl").read_text().splitlines()]
    assert events[0]["environment_differences"]["nvidia_driver"] == {
        "checkpoint": "example-driver", "resume": None}


def test_resume_environment_rejects_identity_change():
    saved = {"hostname": "vm02", "nvidia_driver": "old"}
    current = {"hostname": "vm03", "nvidia_driver": None}
    with pytest.raises(ValueError, match="identity"):
        _compare_resume_environment(saved, current)


def test_equation_head_time_study_records_mixture_and_coverage(tmp_path):
    source = tiny_teacher(tmp_path)
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__, "own_kernel_probability": .5,
                                "backbone_channel_embedding": True})
    result = run(protocol, "own", "time", None, tmp_path / "train", device="cpu", tiny=True,
                 equation_head=True, teacher_source=str(source.parent))
    assert result["updates"] >= 1
    assert result["channel_families"]["kernel_synth"] > 0
    assert result["auxiliary_target_coverage"]


def test_teacher_targets_scale_noise_and_follow_example_channel_order(tmp_path):
    import numpy as np
    from tsfmxai.datasets.online import OnlineBatchDataset
    from tsfmxai.pipelines.equation_targets import EquationTargetTeacher
    source = tiny_teacher(tmp_path)
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__, "own_kernel_probability": .5})
    item = OnlineBatchDataset(own_plan(protocol, "independent", 1))[0]
    teacher = EquationTargetTeacher(str(source.parent), "")
    coverage = {}
    targets = teacher.targets(item, coverage=coverage)
    context = np.asarray(item["context"])
    for example, metadata in enumerate(item["metadata"]):
        system = metadata["system"]
        for channel, family in enumerate(metadata["channel_families"]):
            if family == "kernel_synth":
                assert not targets["noise_mask"][example, channel]
                assert not targets["equation_target_mask"][example, channel]
            else:
                expected = system.innovations[channel].std / context[example, :, channel].std()
                assert targets["noise_std"][example, channel] == pytest.approx(expected, rel=1e-5)
    assert sum(c["covered"] + c["masked"] for c in coverage.values()) == targets["noise_mask"].sum()
    strict = EquationTargetTeacher(str(tiny_teacher(tmp_path / "small", max_nodes=3).parent), "",
                                   on_overflow="raise")
    with pytest.raises(ValueError, match="teacher accepts 3"):
        strict.targets(item)


def test_head_arm_shares_the_corpus_of_the_arm_without_head():
    from tsfmxai.pipelines.prior_baselines import corpus_fields
    plain = PriorProtocol.read(CONFIG.with_name("prior_baseline_frozen.json"))
    head = PriorProtocol.read(CONFIG.with_name("prior_baseline_frozen_head.json"))
    assert (plain.backbone_patch_size, plain.backbone_channel_embedding) == (16, False)
    assert (head.backbone_patch_size, head.backbone_channel_embedding) == (16, True)
    assert corpus_fields(plain, "own") == corpus_fields(head, "own")
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    assert corpus_fields(historical, "lmc") == corpus_fields(plain, "lmc")


def test_equation_head_requires_the_channel_index(tmp_path):
    source = tiny_teacher(tmp_path)
    protocol = PriorProtocol(**{**small_protocol(tmp_path).__dict__,
                                "backbone_channel_embedding": False})
    with pytest.raises(ValueError, match="backbone_channel_embedding"):
        run(protocol, "own", "time", None, tmp_path / "train", device="cpu", tiny=True,
            equation_head=True, teacher_source=str(source.parent))
