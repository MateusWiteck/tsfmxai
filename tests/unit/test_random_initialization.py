"""Random startup families: presence draws, summed components and stream integration."""

import numpy as np
import pytest

from tsfmxai.datasets.multiseries import TrajectoryConfig, TrainingFormat, generate_training_batches
from tsfmxai.generators.combinatorial_expression import CombinatorialTreeConfig
from tsfmxai.generators.initialization import (
    HistorySpec, RandomHistorySpec, make_initial_history, sample_random_history,
)
from tsfmxai.generators.multiseries import MultiseriesConfig


def draw(spec, channels=4, length=16, seed=0):
    return sample_random_history(spec, channels, length, np.random.default_rng(seed))


def test_components_sum_to_the_history_and_respect_their_intervals():
    spec = RandomHistorySpec(constant_probability=1., noise_probability=1., cosine_probability=1.,
                             constant_magnitude=(.5, .7), noise_radius=(.2, .3),
                             cosine_amplitude=(.4, .6), cosine_period=(5, 9))
    history, components = draw(spec, channels=6, length=20)
    k = np.arange(20)
    for row, parts in zip(history, components):
        assert [p["family"] for p in parts] == ["constant", "noise", "cosine"]
        constant, noise, cosine = parts
        assert .5 <= abs(constant["level"]) <= .7
        assert .2 <= noise["radius"] <= .3 and .4 <= cosine["amplitude"] <= .6
        assert 5 <= cosine["period"] <= 9 and 0 <= cosine["phase"] < 2 * np.pi
        wave = cosine["amplitude"] * np.cos(2 * np.pi * k / cosine["period"] + cosine["phase"])
        residual = row - constant["level"] - wave
        assert np.all(np.abs(residual) <= noise["radius"] + 1e-12)
        assert np.std(residual) > 0


def test_single_families_and_the_empty_draw():
    only_constant, parts = draw(RandomHistorySpec(constant_probability=1., constant_magnitude=(.3, .3),
                                                    noise_probability=0., cosine_probability=0.))
    assert all(len(set(np.round(np.abs(row), 12))) == 1 for row in only_constant)
    assert all(p == [{"family": "constant", "level": pytest.approx(row[0])}]
               for p, row in zip(parts, only_constant))
    nothing, parts = draw(RandomHistorySpec(constant_probability=0., noise_probability=0.,
                                               cosine_probability=0.))
    assert np.all(nothing == 0) and parts == [[]] * 4


def test_presence_frequencies_follow_probabilities_and_channels_differ():
    spec = RandomHistorySpec(constant_probability=.2, noise_probability=.5, cosine_probability=.8)
    _, components = draw(spec, channels=4000, length=8, seed=3)
    for family, p in (("constant", .2), ("noise", .5), ("cosine", .8)):
        rate = np.mean([any(c["family"] == family for c in parts) for parts in components])
        assert abs(rate - p) < .03
    assert len({tuple(c["family"] for c in parts) for parts in components}) == 8


def test_draws_are_reproducible_and_family_parameters_do_not_shift():
    a = draw(RandomHistorySpec(), seed=11)
    b = draw(RandomHistorySpec(), seed=11)
    np.testing.assert_array_equal(a[0], b[0])
    with_cosine = draw(RandomHistorySpec(constant_probability=1., noise_probability=0.,
                                     cosine_probability=1.), seed=5)[1]
    without = draw(RandomHistorySpec(constant_probability=1., noise_probability=0.,
                                 cosine_probability=0.), seed=5)[1]
    assert [p[0] for p in with_cosine] == [p[0] for p in without]


@pytest.mark.parametrize("bad", [dict(constant_probability=1.5), dict(noise_probability=-.1),
                                 dict(constant_magnitude=(1., .5)), dict(noise_radius=(-1., 1.)),
                                 dict(cosine_period=(1, 4)), dict(cosine_period=(4., 8.)),
                                 dict(cosine_amplitude=(0., np.inf)), dict(name=1.)])
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValueError):
        RandomHistorySpec(**bad)


def test_make_initial_history_accepts_the_random_spec():
    history = make_initial_history(RandomHistorySpec(), None, np.zeros((3, 7)), 9)
    assert history.shape == (3, 7) and np.isfinite(history).all()


def _stream(initialization, seed=8):
    cfg = MultiseriesConfig(
        ast_config=CombinatorialTreeConfig(maximum_operators=3, maximum_lag=6, max_degree=2),
        channel_count_probs={3: 1.}, dependency_probs={"random": 1.})
    return generate_training_batches(
        config=cfg, num_systems=3, trajectories_per_system=2, seed=seed,
        trajectory=TrajectoryConfig(30, initialization),
        training=TrainingFormat(20, 10, 30, 64))


def test_training_stream_draws_families_per_channel_and_records_them():
    batch = next(_stream(RandomHistorySpec()))
    for metadata in batch.metadata:
        history, parts = metadata["initial_history"], metadata["initialization_components"]
        assert len(parts) == history.shape[0] == metadata["system"].num_channels
        for row, channel_parts in zip(history, parts):
            if not channel_parts:
                assert np.all(row == 0)
    again = next(_stream(RandomHistorySpec()))
    np.testing.assert_array_equal(batch.context, again.context)
    fixed = next(_stream(HistorySpec("uniform", "uniform")))
    assert all(m["initialization_components"] is None for m in fixed.metadata)


def test_stationary_and_unknown_initializations_are_rejected():
    with pytest.raises(ValueError, match="stationary"):
        TrajectoryConfig(10, HistorySpec("s", "stationary"))
    with pytest.raises(ValueError):
        TrajectoryConfig(10, "random")


def test_study_bound_lets_the_jacobian_run_only_inside_its_box():
    from tsfmxai.evaluation.initialization import (
        InitializationStudyConfig, history_spec_unavailable, sample_study_mechanisms,
    )
    from tsfmxai.generators.initialization import random_history_bound
    wide = RandomHistorySpec()
    unit = RandomHistorySpec(constant_magnitude=(0., 1 / 3), noise_radius=(.05, 1 / 3),
                             cosine_amplitude=(.05, 1 / 3))
    only_noise = RandomHistorySpec(constant_probability=0., cosine_probability=0.)
    assert random_history_bound(wide) == 3. and random_history_bound(only_noise) == 1.
    jacobian = sample_study_mechanisms("Jacobian", InitializationStudyConfig(mechanisms=1))[0]
    assert history_spec_unavailable(wide, jacobian) == "Requested startup support exceeds certified box"
    assert history_spec_unavailable(unit, jacobian) is None
