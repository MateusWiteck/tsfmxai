"""Four-family integration contracts (not executed as part of implementation)."""
from dataclasses import replace

import numpy as np
import pytest

from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.multiseries import (
    MultiseriesConfig, draw_heaps_lag_support, heaps_lag_supports,
    sample_multiseries_system,
)
from tsfmxai.generators.initialization import RandomHistorySpec
from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat, generate_training_batches
from tsfmxai.representations.expression import Lag, walk


def config():
    return MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=4),
        channel_count_probs={3: 1.}, dependency_probs={"random": 1.})


def test_requested_prior():
    assert config().self_only_treatment_probs == {"jacobian": .5, "heaps": .5}


def test_dependency_policy_matches_topology_and_uses_external_source():
    cfg = replace(config(), channel_count_probs={6: 1.})
    for seed in range(30):
        system = sample_multiseries_system(cfg, seed=seed)
        for target, (sources, treatment, equation) in enumerate(zip(
                system.allowed_sources, system.treatments, system.original_equations)):
            if len(sources) > 1:
                assert treatment.method == "shao_wu"
                assert any(isinstance(node, Lag) and node.variable != target
                           for node in walk(equation.expression))
                assert set(system.effective_sources[target]) - {target}
            else:
                assert treatment.method in ("jacobian", "heaps")


def test_independent_topology_has_no_shao_wu():
    cfg = replace(config(), channel_count_probs={6: 1.},
                  dependency_probs={"independent": 1.})
    for seed in range(12):
        system = sample_multiseries_system(cfg, seed=seed)
        assert all(t.method in ("jacobian", "heaps") for t in system.treatments)


def test_external_kernel_channel_counts_as_dependency():
    from tsfmxai.generators.hybrid import HybridConfig, sample_hybrid_system

    cfg = HybridConfig(replace(config(), channel_count_probs={6: 1.}),
                       {"ast": .5, "kernel_synth": .5})
    checked = 0
    for seed in range(30):
        system = sample_hybrid_system(cfg, seed=seed)
        for target, sources in enumerate(system.allowed_sources):
            if system.channel_families[target] != "ast":
                continue
            if any(source != target and system.channel_families[source] == "kernel_synth"
                   for source in sources):
                assert system.treatments[target].method == "shao_wu"
                assert set(system.effective_sources[target]) - {target}
                checked += 1
    assert checked > 0


def test_retention_preserves_required_memory_and_replay():
    cfg = config()
    for seed in range(20):
        system = sample_multiseries_system(cfg, seed=seed)
        assert system == sample_multiseries_system(cfg, seed=seed)
        assert all(e.binding.lag_values for e in system.equations)


@pytest.mark.parametrize("method,family", [("jacobian", "uniform"), ("heaps", "gaussian")])
def test_certified_channels_and_random_history(method, family):
    cfg = replace(config(), dependency_probs={"independent": 1.},
                  self_only_treatment_probs={method: 1.})
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


def test_heaps_support_combinatorics_and_declared_probabilities():
    supports = heaps_lag_supports(16)
    assert {name: len(pairs) for name, pairs in supports.items()} == {
        "consecutive": 16, "isolated": 15, "periodic": 19}
    assert len(set().union(*(set(pairs) for pairs in supports.values()))) == 50
    rng = np.random.default_rng(234)
    draws = [draw_heaps_lag_support(rng, 16)[0] for _ in range(20_000)]
    for name, expected in (("consecutive", .4), ("isolated", .2), ("periodic", .4)):
        assert draws.count(name) / len(draws) == pytest.approx(expected, abs=.015)


def test_heaps_direct_support_sampling_and_replay():
    cfg = replace(config(), channel_count_probs={1: 1.},
                  self_only_treatment_probs={"heaps": 1.})
    seen = set()
    for seed in range(35):
        system = sample_multiseries_system(cfg, seed=seed)
        assert system == sample_multiseries_system(cfg, seed=seed)
        assert not system.original_equations[0].binding.lag_values
        treatment = system.treatments[0]
        expected, period = treatment.heaps_order, treatment.heaps_period
        assert (expected, period) in heaps_lag_supports(4)[treatment.heaps_support]
        assert system.equations[0].binding.lag_values
        assert max(system.equations[0].binding.lag_values.values()) == expected * period
        assert set(system.equations[0].binding.lag_values.values()) == {
            period * index for index in range(1, expected + 1)}
        seen.add(treatment.heaps_support)
    assert seen == {"consecutive", "isolated", "periodic"}


def test_heaps_support_stability_sweep_with_full_training_windows():
    cfg = replace(config(),
                  ast_config=CombinatorialTreeConfig(maximum_operators=6,
                                                     maximum_lag=16),
                  channel_count_probs={6: 1.}, dependency_probs={"independent": 1.},
                  self_only_treatment_probs={"heaps": 1.})
    for seed in range(100):
        system = sample_multiseries_system(cfg, seed=seed)
        assert all(certificate.holds for certificate in system.channel_certificates)
        stream = generate_training_batches(
            systems=[system], trajectory=TrajectoryConfig(192, RandomHistorySpec()),
            training=TrainingFormat(96, 96, 192, batch_size=1), seed=seed)
        batch = next(stream)
        assert np.isfinite(batch.context).all()
        assert np.isfinite(batch.future).all()


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
    cfg = config()
    stream = generate_training_batches(
        config=cfg, num_systems=1,
        trajectory=TrajectoryConfig(2, RandomHistorySpec()),
        training=TrainingFormat(1, 1, 2, batch_size=1), seed=42)
    next(stream)
    assert stream.stats.initialization_attempts == 5
    assert stream.stats.replaced_systems == 1
    assert stream.stats.resampled_equations == 3
    assert stream.stats.resampled_original_equations == 0
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
