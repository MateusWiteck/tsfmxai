"""Contracts for topology, initialization, execution, labels and numerical retries."""

from dataclasses import replace

import numpy as np
import pytest

from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import HistorySpec, make_initial_history
from tsfmxai.generators.multiseries import (
    MultiseriesConfig, NoiseConfig, TreatmentSpec, InnovationSpec,
    build_multiseries_system, sample_multiseries_system,
)
from tsfmxai.generators.random_expression import SampledMechanism
from tsfmxai.representations import Add, Multiply, Parameter, Lag, MechanismBinding
from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat, generate_training_batches


def equation(expression, parameters=None, lags=None):
    return SampledMechanism(expression, MechanismBinding(parameters or {}, lags or {}), 0, 0)


def explicit(equations, *, noise=0., family="gaussian", methods=None):
    n = len(equations)
    return build_multiseries_system(
        equations=equations, allowed_sources={i: list(range(n)) for i in range(n)},
        treatments=[TreatmentSpec(m) for m in (methods or ["original"] * n)],
        innovations=[InnovationSpec(family, noise)] * n)


def config(dependency="random"):
    return MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=4, max_degree=2),
        channel_count_probs={4: 1.}, dependency_probs={dependency: 1.},
        treatment_probs={"original": .5, "shao_wu": .5})


def stream(system, *, steps=12, dtype="float64", history=None, retries=0, batch_size=2):
    return generate_training_batches(systems=[system], trajectory=TrajectoryConfig(
        steps, history or HistorySpec("zero", "zero"), retries),
        training=TrainingFormat(3, 2, 2, batch_size, dtype=dtype), seed=42)


@pytest.mark.parametrize("dependency", ["independent", "linked_chain", "previous_chain", "random"])
def test_topology_permissions_and_reproducible_sampling(dependency):
    cfg = config(dependency)
    first = sample_multiseries_system(cfg, seed=31)
    np.random.seed(987)
    assert first == sample_multiseries_system(cfg, seed=31)
    positions = {channel: i for i, channel in enumerate(first.channel_order)}
    for target, sources in enumerate(first.allowed_sources):
        assert target in sources
        assert len(sources) - 1 <= cfg.max_external_sources
        assert set(first.effective_sources[target]) <= set(sources)
        for source in set(sources) - {target}:
            if dependency == "independent":
                pytest.fail("Independent channels cannot reference external channels")
            if dependency == "linked_chain":
                assert positions[source] == positions[target] - 1
            if dependency == "previous_chain":
                assert positions[source] < positions[target]
    assert first.joint_certificate is None


def test_self_preference_is_not_a_mandatory_lag_and_no_lags_allowed():
    # Source assignment statistics use single-lag trees to isolate the policy.
    cfg = replace(config("linked_chain"), channel_count_probs={2: 1.},
                  ast_config=CombinatorialTreeConfig(minimum_operators=0, maximum_operators=0,
                                                    maximum_lag=1, max_degree=1),
                  allow_no_lags=False, treatment_probs={"original": 1.}, self_source_weight=3.)
    own = 0
    for seed in range(200):
        system = sample_multiseries_system(cfg, seed=seed)
        target = system.channel_order[1]
        own += system.effective_sources[target] == (target,)
    assert 120 < own < 180
    cfg = replace(cfg, allow_no_lags=True, ast_config=replace(cfg.ast_config, max_degree=0, require_lag=False))
    assert all(not s for s in sample_multiseries_system(cfg, seed=1).effective_sources)


def test_synchronous_execution_matches_manual_recurrence_and_keeps_noise_targets():
    system = explicit([equation(Lag(1, "L"), lags={"L": 1}),
                       equation(Add(Lag(0, "L"), Parameter("b")), {"b": 1.}, {"L": 1})])
    batches = list(stream(system))
    # Simultaneous from [0,0]: [0,1], [1,1], [1,2], [2,2], [2,3].
    np.testing.assert_array_equal(batches[0].context[0], [[0, 1], [1, 1], [1, 2]])
    np.testing.assert_array_equal(batches[0].future[0], [[2, 2], [2, 3]])
    assert batches[0].metadata[0]["window_start"] == 0
    assert batches[0].noise_targets["std"].shape == (2, 2)


def test_history_api_is_shared_with_initialization_experiments():
    from tsfmxai.evaluation.initialization import HistorySpec as StudySpec
    assert StudySpec is HistorySpec
    system = explicit([equation(Lag(0, "L"), lags={"L": 2})])
    spec = HistorySpec("half cosine", "cosine", amplitude=2., period=8, fraction=.5)
    batch = next(stream(system, history=spec))
    expected = 2 * np.cos(2 * np.pi * np.arange(4) / 8)
    np.testing.assert_allclose(batch.metadata[0]["initial_history"][0], expected)
    np.testing.assert_allclose(batch.context[0, :2, 0], expected[-2:])
    assert batch.metadata[0]["initial_history"].shape == (1, 4)
    with pytest.raises(ValueError, match="shorter"):
        make_initial_history(HistorySpec("short", "cosine", period=4, fraction=.25),
                             system, np.zeros((1, 2)), 0)


def test_training_windows_exclude_sampled_startup_history():
    # The lag makes a two-value startup history necessary, but its zero
    # coefficient means every recurrence output is exactly seven. If the
    # startup prefix leaked into the batch, the context would contain 999s.
    system = explicit([equation(
        Add(Parameter("offset"), Multiply(Parameter("zero"), Lag(0, "L"))),
        {"offset": 7., "zero": 0.}, {"L": 2})])
    data = generate_training_batches(
        systems=[system], seed=19,
        trajectory=TrajectoryConfig(5, HistorySpec("sentinel", "constant", offset=999.)),
        training=TrainingFormat(3, 2, 1, batch_size=1, dtype="float64"),
    )
    batch = next(data)
    np.testing.assert_array_equal(batch.metadata[0]["initial_history"], [[999., 999.]])
    np.testing.assert_array_equal(batch.context, np.full((1, 3, 1), 7.))
    np.testing.assert_array_equal(batch.future, np.full((1, 2, 1), 7.))


def test_protection_targets_match_executed_expression():
    raw = equation(Multiply(Lag(0, "a"), Add(Lag(1, "b"), Parameter("c"))),
                   {"c": 1.}, {"a": 1, "b": 2})
    system = explicit([raw, equation(Parameter("c"), {"c": .2})], methods=["shao_wu", "original"])
    assert any(len(atom) == 3 for term in system.canonical[0].skeleton for atom in term)
    data = stream(system)
    batch = next(data)
    ids = batch.equation_targets[0, 0][batch.equation_mask[0, 0]]
    decoded = data.codec.decode(ids[1:])
    assert decoded["valid"]
    assert decoded["skeleton"] == system.canonical[0].skeleton


def test_finite_large_values_accepted_but_output_overflow_recorded():
    system = explicit([equation(Parameter("c"), {"c": 1e50})])
    large = next(stream(system))
    assert np.all(large.context == 1e50)
    data = stream(system, dtype="float32", retries=2)
    assert list(data) == []
    assert data.stats.initialization_attempts == 3
    assert data.stats.skipped_trajectories == 1
    assert {f["reason"] for f in data.stats.failures} == {"nonfinite_output_dtype"}
    assert len({f["innovation_seed"] for f in data.stats.failures}) == 1
    assert len({f["initialization_seed"] for f in data.stats.failures}) == 3


def test_nonfinite_recurrence_stops_and_raise_policy():
    system = explicit([equation(Multiply(Parameter("a"), Lag(0, "L")),
                                {"a": 1e200}, {"L": 1})])
    data = stream(system, history=HistorySpec("one", "constant", offset=1.))
    assert list(data) == []
    assert data.stats.failures[0]["reason"] == "nonfinite_simulation"
    data = generate_training_batches(systems=[system], trajectory=TrajectoryConfig(
        6, HistorySpec("one", "constant", offset=1.), 0, "raise"),
        training=TrainingFormat(3, 2, 1))
    with pytest.raises(RuntimeError, match="exhausted"):
        list(data)


def test_padding_noise_laws_and_batch_size_do_not_change_examples():
    systems = [explicit([equation(Parameter("c"), {"c": 2.})], noise=.2, family="uniform"),
               explicit([equation(Parameter("c"), {"c": 1.})] * 3, noise=.1)]
    def collect(batch_size):
        return list(generate_training_batches(systems=systems, seed=77,
                    trajectory=TrajectoryConfig(10, HistorySpec("uniform", "uniform")),
                    training=TrainingFormat(3, 2, 2, batch_size)))
    a, b = collect(2), collect(5)
    np.testing.assert_array_equal(np.concatenate([x.context for x in a]),
                                  np.concatenate([x.context for x in b]))
    assert a[0].channel_mask.tolist() == [[True, False, False]] * 2
    assert not a[0].equation_mask[:, 1:].any()
    assert (a[0].noise_targets["family"][:, 0] == 1).all()
    assert np.max(np.abs(a[0].context[:, :, 0] - 2)) <= np.sqrt(3) * .2 + 1e-6


def test_sampling_stream_vocabulary_and_scale_only_layout():
    cfg = replace(config(), channel_count_probs={1: .5, 3: .5}, treatment_probs={"shao_wu": 1.})
    data = generate_training_batches(config=cfg, num_systems=4, seed=9,
        trajectory=TrajectoryConfig(8, HistorySpec("uniform", "uniform")),
        training=TrainingFormat(3, 2, 2, 3, layout="batch_channel_time", noise_target="scale"))
    batches = list(data)
    assert data.codec.num_series == 3
    assert all(b.context.shape[1:] == (3, 3) for b in batches)
    assert set(batches[0].noise_targets) == {"std"}
    assert data.stats.accepted_trajectories == 4


@pytest.mark.parametrize("kwargs", [dict(channel_count_probs={0: 1.}),
    dict(dependency_probs={"random": -.5}), dict(treatment_probs={"heaps": 1.}),
    dict(self_source_weight=0.), dict(max_external_sources=-1)])
def test_bad_configs_fail_early(kwargs):
    with pytest.raises(ValueError):
        replace(config(), **kwargs)


def test_stationary_history_not_claimed_for_mixed_system():
    with pytest.raises(ValueError, match="stationary"):
        TrajectoryConfig(10, HistorySpec("stationary", "stationary"))


def test_explicit_forbidden_sources_rejected():
    with pytest.raises(ValueError, match="forbidden"):
        build_multiseries_system(equations=[equation(Lag(1, "L"), lags={"L": 1}),
                                           equation(Parameter("c"), {"c": 1.})],
            allowed_sources={0: [0], 1: [1]}, treatments=[TreatmentSpec()] * 2,
            innovations=[InnovationSpec()] * 2)


def test_metadata_replays_gaussian_noise_and_initialization_exactly():
    system = explicit([equation(Add(Multiply(Parameter("a"), Lag(0, "L")), Parameter("b")),
                                {"a": .4, "b": 2.}, {"L": 1})], noise=.15)
    data = stream(system, history=HistorySpec("offset uniform", "uniform", amplitude=2., offset=1.))
    batch = next(data)
    meta = batch.metadata[0]
    unit = np.random.default_rng(meta["initialization_seed"]).uniform(-1., 1., (1, 1))
    np.testing.assert_array_equal(meta["initial_history"], 1. + 2. * unit)
    noise = np.random.default_rng(meta["innovation_seed"]).normal(0., .15, 12)
    expected, previous = [], meta["initial_history"][0, -1]
    for shock in noise:
        previous = .4 * previous + 2. + shock
        expected.append(previous)
    np.testing.assert_allclose(batch.context[0, :, 0], expected[:3])
    np.testing.assert_allclose(batch.future[0, :, 0], expected[3:5])
    assert batch.noise_targets["std"][0, 0] == .15


def test_failed_system_does_not_suppress_following_system_or_partial_batch():
    bad = explicit([equation(Parameter("c"), {"c": 1e50})])
    good = explicit([equation(Parameter("c"), {"c": 7.})])
    data = generate_training_batches(systems=[bad, good],
        trajectory=TrajectoryConfig(6, HistorySpec("zero", "zero"), 0),
        training=TrainingFormat(3, 2, 2, 8))
    batches = list(data)
    assert len(batches) == 1 and batches[0].context.shape == (1, 3, 1)
    assert batches[0].metadata[0]["system_index"] == 1
    assert data.stats.skipped_trajectories == 1
    assert data.stats.accepted_trajectories == 1


def test_codec_handles_cancelled_lags_and_zero_equation():
    system = explicit([equation(Add(Lag(0, "L"), Multiply(Parameter("a"), Lag(0, "L"))),
                                {"a": -1.}, {"L": 20})])
    data = stream(system)
    batch = next(data)
    assert data.codec.maximum_lag == 20
    assert system.canonical[0].skeleton == ()
    ids = batch.equation_targets[0, 0][batch.equation_mask[0, 0]]
    assert data.codec.decode(ids[1:])["skeleton"] == ()
