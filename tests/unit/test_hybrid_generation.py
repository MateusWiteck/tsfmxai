"""KernelSynth channels remain forecast targets without invented equations."""

import numpy as np

from tsfmxai.datasets.hybrid import generate_hybrid_training_batches
from tsfmxai.datasets.multiseries import TrainingFormat, TrajectoryConfig
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.hybrid import HybridConfig, sample_hybrid_system
from tsfmxai.generators.initialization import HistorySpec
from tsfmxai.generators.multiseries import MultiseriesConfig


def test_hybrid_targets_and_replay():
    base = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=2, maximum_lag=2),
        channel_count_probs={3: 1.}, dependency_probs={"linked_chain": 1.},
        treatment_probs={"shao_wu": 1.},
    )
    config = HybridConfig(base, {"ast": .5, "kernel_synth": .5})
    def batch():
        return next(generate_hybrid_training_batches(
            config=config, num_systems=1,
            trajectory=TrajectoryConfig(12, HistorySpec("zero", "zero")),
            training=TrainingFormat(5, 3, 4, batch_size=1), seed=3))
    first, second = batch(), batch()
    np.testing.assert_array_equal(first.context, second.context)
    np.testing.assert_array_equal(first.future, second.future)
    np.testing.assert_array_equal(first.equation_mask, second.equation_mask)
    assert first.channel_mask.all()
    for channel, family in enumerate(first.metadata[0]["channel_families"]):
        assert first.equation_mask[0, channel].any() == (family == "ast")
        assert first.noise_targets["mask"][0, channel] == (family == "ast")
    from tsfmxai.datasets.chronos2_adapter import adapt_tsfmxai_batch
    adapted = adapt_tsfmxai_batch(first, output_patch_size=4)
    assert adapted["context"].shape[0] == 3
    np.testing.assert_array_equal(adapted["noise_mask"].numpy(),
                                  first.noise_targets["mask"].reshape(-1))
    assert sample_hybrid_system(config, seed=3).channel_families == sample_hybrid_system(config, seed=3).channel_families


def test_auxiliary_losses_ignore_kernel_channels():
    import torch
    from tsfmxai.models.chronos2_equation import equation_training_loss

    output = {
        "forecast_loss": torch.tensor(2.0),
        "skeleton_embedding": torch.tensor([[1.0, 0.0], [100.0, 100.0]]),
        "predicted_log_noise_std": torch.tensor([0.0, 100.0]),
    }
    batch = {
        "teacher_embedding": torch.tensor([[1.0, 0.0], [-100.0, -100.0]]),
        "noise_std": torch.tensor([1.0, 100.0]),
        "equation_target_mask": torch.tensor([True, False]),
        "noise_mask": torch.tensor([True, False]),
    }
    total, terms = equation_training_loss(output, batch)
    assert total.item() == 2.0
    assert terms["skeleton_embedding"].item() == 0.0
    assert terms["innovation_log_std"].item() == 0.0
    batch["equation_target_mask"] = torch.tensor([False, False])
    batch["noise_mask"] = torch.tensor([False, False])
    total, _ = equation_training_loss(output, batch)
    assert torch.isfinite(total) and total.item() == 2.0


def test_ast_channel_uses_lagged_kernel_source(monkeypatch):
    from tsfmxai.generators.hybrid import HybridSystem, simulate_hybrid_path
    from tsfmxai.generators.multiseries import (
        InnovationSpec, TreatmentSpec, build_multiseries_system,
    )
    from tsfmxai.generators.random_expression import SampledMechanism
    from tsfmxai.representations.expression import Lag, MechanismBinding, Parameter

    equations = [
        SampledMechanism(Parameter("A"), MechanismBinding({"A": 0.}, {}), 0, 0),
        SampledMechanism(Lag(0, "L"), MechanismBinding({}, {"L": 1}), 0, 1),
    ]
    base = build_multiseries_system(
        equations=equations, allowed_sources={0: [0], 1: [0, 1]},
        treatments=[TreatmentSpec("original")] * 2,
        innovations=[InnovationSpec("gaussian", 0.)] * 2,
    )
    hybrid = HybridSystem(base, ("kernel_synth", "ast"), (((0,), ()), None))
    monkeypatch.setattr("tsfmxai.generators.hybrid._draw_kernel_synth",
                        lambda spec, length, seed: np.arange(length, dtype=float) + 7.)
    values, history = simulate_hybrid_path(hybrid, np.zeros((2, 1)),
                                           np.zeros((2, 3)), seed=4)
    np.testing.assert_array_equal(history[0], [7.])
    np.testing.assert_array_equal(values[0], [8., 9., 10.])
    np.testing.assert_array_equal(values[1], [7., 8., 9.])
