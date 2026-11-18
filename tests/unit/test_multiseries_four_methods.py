"""Four-family integration contracts (not executed as part of implementation)."""
from dataclasses import replace

import numpy as np
import pytest

from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.multiseries import MultiseriesConfig, sample_multiseries_system
from tsfmxai.generators.initialization import RandomHistorySpec
from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat, generate_training_batches


def config():
    return MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=4),
        channel_count_probs={3: 1.}, dependency_probs={"random": 1.})


def test_requested_prior():
    assert config().treatment_probs == {
        "original": 0., "shao_wu": .5, "jacobian": 1./6., "heaps": 1./3.}


def test_effective_source_retention_scales_without_channel_count_limit():
    from tsfmxai.generators.multiseries import LagSamplingConfig
    rule = LagSamplingConfig()
    assert rule.probability([1.]) == .25
    assert rule.probability([3., 1.]) == pytest.approx(.6875 - .25)
    assert rule.probability([1.] * 100) == pytest.approx(.995 - .25)
    assert rule.probability([3., 1.]) < rule.probability([1., 1.])


def test_retention_preserves_required_memory_and_replay():
    cfg = replace(config(), treatment_probs={"original": 1.}, allow_no_lags=False)
    for seed in range(20):
        system = sample_multiseries_system(cfg, seed=seed)
        assert system == sample_multiseries_system(cfg, seed=seed)
        assert all(e.binding.lag_values for e in system.equations)


@pytest.mark.parametrize("method,family", [("jacobian", "uniform"), ("heaps", "gaussian")])
def test_certified_channels_and_random_history(method, family):
    cfg = replace(config(), treatment_probs={method: 1.})
    system = sample_multiseries_system(cfg, seed=31)
    assert system.allowed_sources == ((0,), (1,), (2,))
    assert all(c.holds for c in system.channel_certificates)
    assert all(n.family == family for n in system.innovations)
    stream = generate_training_batches(
        systems=[system], trajectory=TrajectoryConfig(16, RandomHistorySpec()),
        training=TrainingFormat(8, 8, 16, batch_size=1), seed=42)
    batch = next(stream)
    assert np.isfinite(batch.context).all()
    assert np.isfinite(batch.future).all()
    if method == "jacobian":
        assert np.max(np.abs(batch.future)) <= 3.
    assert stream.stats.resampled_equations == 0


def test_exhausted_histories_count_discarded_trees(monkeypatch):
    import tsfmxai.datasets.multiseries as module

    calls = []
    actual = module.simulate_innovation_path

    def fail_first_system(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 4:
            return np.full_like(args[3], np.nan)
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "simulate_innovation_path", fail_first_system)
    cfg = replace(config(), treatment_probs={"original": 1.})
    stream = generate_training_batches(
        config=cfg, num_systems=1,
        trajectory=TrajectoryConfig(2, RandomHistorySpec()),
        training=TrainingFormat(1, 1, 2, batch_size=1), seed=42)
    next(stream)
    assert stream.stats.initialization_attempts == 5
    assert stream.stats.replaced_systems == 1
    assert stream.stats.resampled_equations == 3
    assert stream.stats.resampled_original_equations == 3
    assert len(stream.stats.replacement_records) == 1


def test_single_terminal_probability_is_shared_by_lag_labels():
    from tsfmxai.generators.combinatorial_expression import CombinatorialTreeSampler
    from tsfmxai.representations.expression import Lag
    cfg = CombinatorialTreeConfig(minimum_operators=0, maximum_operators=0,
                                  maximum_lag=4, require_lag=False)
    sampler = CombinatorialTreeSampler(cfg, lag_probability=.75)
    draws = [sampler.unrank(i) for i in range(sampler.total_structures)]
    lag_draws = [m for m in draws if isinstance(m.expression, Lag)]
    assert len(lag_draws) == .75 * len(draws)
    for lag in range(1, 5):
        assert sum(m.binding.lag_values["L0"] == lag for m in lag_draws) == len(lag_draws) / 4
