from dataclasses import replace

import numpy as np
import pytest

from tsfmxai.evaluation import (
    TAIL_LABELS, StabilityDiagnosticConfig, TailDiagnosticConfig,
    classify_tail_windows, measure_tail_prefixes,
)
from tsfmxai.evaluation.generator_tradeoff import benchmark_stability_method
from tsfmxai.generators import SampledMechanism
from tsfmxai.representations import Add, Lag, MechanismBinding, Multiply, Parameter


STABILITY = StabilityDiagnosticConfig((12,), 100., .15, 4)
TAIL = TailDiagnosticConfig(4, 2, .15)


def diagnose(values, config=TAIL, stability=STABILITY):
    stats = measure_tail_prefixes(values, stability, config)[0]
    return stats, classify_tail_windows(stats, config, near_zero_radius=stability.near_zero_radius)


def test_flat_nonzero_level_and_initially_quiet_are_distinct_from_near_zero():
    stats, flags = diagnose(np.full(12, 5.))
    assert stats.tail_mean == (5.,)
    assert stats.tail_variation == (0.,)
    assert flags['flat_tail'] and flags['persistent_flat_tail']
    assert flags['flat_nonzero'] and flags['quiet_initial_and_tail']
    assert not flags['flattening'] and not flags['persistent_near_zero']


def test_flattening_uses_initial_variation_and_can_finish_away_from_zero():
    stats, flags = diagnose(np.r_[[-1., 1., -1., 1.], np.full(8, 5.)])
    assert stats.initial_variation == (1.,)
    assert flags['flattening'] and flags['flat_nonzero']
    assert not flags['quiet_initial_and_tail']
    assert not flags['amplitude_decline_near_zero']


def test_amplitude_decline_and_near_zero_from_start_are_distinguished():
    _, declining = diagnose(np.r_[[-1., 1., -1., 1.], np.full(8, .1)])
    _, always_small = diagnose(np.full(12, .1))
    assert declining['persistent_near_zero'] and always_small['persistent_near_zero']
    assert declining['amplitude_decline_near_zero']
    assert not always_small['amplitude_decline_near_zero']


def test_sign_changes_are_measured_before_any_magnitude_envelope():
    _, flags = diagnose(np.tile([-1., 1.], 6))
    assert not flags['flat_tail']
    _, low = diagnose(np.tile([-.14, .14], 6), replace(TAIL, flat_tolerance=.05))
    assert low['persistent_near_zero'] and not low['flat_tail']


def test_adjacent_flat_plateaus_with_changing_level_fail_persistence():
    stats, flags = diagnose(np.r_[np.zeros(8), np.ones(4)])
    assert stats.recent_window_variations == ((0.,), (0.,))
    assert stats.persistent_variation == (.5,)
    assert flags['flat_tail'] and not flags['persistent_flat_tail']


def test_persistence_also_requires_each_individual_window():
    # Pooled maximum is .15, but the last window has a deviation of .225.
    path = np.r_[np.zeros(4), [.15, .15, -.15, .15], [-.15, -.15, -.15, .15]]
    stats, flags = diagnose(path)
    assert max(stats.persistent_variation) <= .15
    assert not flags['persistent_flat_tail']


def test_multivariate_centers_are_channel_specific_and_all_channels_must_be_flat():
    _, flags = diagnose(np.vstack([np.full(12, 5.), np.full(12, -5.)]))
    assert flags['persistent_flat_tail'] and flags['flat_nonzero']
    _, flags = diagnose(np.vstack([np.full(12, 5.), np.tile([-1., 1.], 6)]))
    assert not flags['flat_tail']


def test_decline_must_hold_for_each_initially_active_channel():
    cfg = replace(TAIL, flat_tolerance=.1)
    values = np.array([
        [-2., 2., -2., 2.] + [-.02, .02]*4,
        [-.12, .12, -.12, .12] + [-.09, .09]*4,
    ])
    _, flags = diagnose(values, cfg)
    assert flags['persistent_flat_tail'] and not flags['flattening']


@pytest.mark.parametrize('failure', [101., -101., np.inf, np.nan])
def test_terminal_failures_are_ineligible_for_every_tail_label(failure):
    stats, flags = diagnose(np.r_[np.zeros(11), failure])
    assert not stats.tail_eligible and stats.tail_mean is None
    assert all(not flag for flag in flags.values())


def test_early_stop_is_ineligible_but_an_earlier_successful_prefix_is_measured():
    cfg = replace(STABILITY, horizons=(12, 24))
    stats = measure_tail_prefixes(np.r_[np.zeros(12), 101.], cfg, TAIL)
    assert stats[0].tail_eligible and not stats[1].tail_eligible
    assert stats[1].tail_variation is None


def test_exact_tolerance_boundary_and_sensitivity_reuse_statistics():
    stats, flags = diagnose(np.tile([-.125, .125], 6), replace(TAIL, flat_tolerance=.125))
    assert flags['persistent_flat_tail']
    stricter = classify_tail_windows(stats, replace(TAIL, flat_tolerance=.124), near_zero_radius=.15)
    assert not stricter['flat_tail']


def test_strict_zero_tolerance_detects_exact_constant():
    _, flags = diagnose(np.full(12, 5.), replace(TAIL, flat_tolerance=0.))
    assert flags['persistent_flat_tail']


@pytest.mark.parametrize('changes', [
    {'window_steps': 1}, {'window_steps': 2.5}, {'window_steps': True},
    {'persistence_windows': 1}, {'flat_tolerance': -1.}, {'flat_tolerance': np.nan},
    {'flat_tolerance': np.inf}, {'decline_ratio': 0.}, {'decline_ratio': 1.},
])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        replace(TAIL, **changes)


def test_windows_must_be_disjoint_and_successful_trajectory_complete():
    with pytest.raises(ValueError, match='overlap'):
        measure_tail_prefixes(np.zeros(12), STABILITY, replace(TAIL, window_steps=5))
    with pytest.raises(ValueError, match='largest horizon'):
        diagnose(np.zeros(11))


def test_early_quiet_label_only_compares_observed_endpoint_windows():
    values = np.zeros(24)
    values[10] = 10.
    _, flags = diagnose(values, stability=replace(STABILITY, horizons=(24,)))
    assert flags['quiet_initial_and_tail']  # Intermediate behavior is explicitly unconstrained.


def test_all_window_statistics_match_independent_vectorized_calculation():
    rng = np.random.default_rng(19)
    for _ in range(100):
        values = rng.normal(size=(3, 24))
        stats, _ = diagnose(values, stability=replace(STABILITY, horizons=(24,)))
        for block, means, deviations in [
            (values[:, :4], stats.initial_mean, stats.initial_variation),
            (values[:, -4:], stats.tail_mean, stats.tail_variation),
            (values[:, -8:], stats.persistent_mean, stats.persistent_variation),
        ]:
            np.testing.assert_allclose(means, np.mean(block, axis=1))
            np.testing.assert_allclose(deviations, np.max(abs(block-block.mean(axis=1, keepdims=True)), axis=1))


def test_optional_benchmark_extension_preserves_generated_paths_and_old_metrics():
    mechanisms = [
        SampledMechanism(Add(Multiply(Parameter('a'), Lag(0, 'l')), Parameter('b')),
                         MechanismBinding({'a': .3, 'b': 5.}, {'l': 1}), 0, 0),
        SampledMechanism(Parameter('p'), MechanismBinding({'p':101.}, {}), 1, 1),
    ]
    kwargs = dict(root_seed=33, diagnostic_config=STABILITY, keep_examples=2)
    previous = benchmark_stability_method('audit', mechanisms, **kwargs)
    extended = benchmark_stability_method('audit', mechanisms, **kwargs,
        tail_config=TAIL, tail_tolerances=(.05,.15))
    for old, new in zip(previous['outcomes'], extended['outcomes']):
        assert all(new[key] == value for key, value in old.items())
    for old, new in zip(previous['examples'], extended['examples']):
        np.testing.assert_array_equal(old['generated'], new['generated'])
        np.testing.assert_array_equal(old['history'], new['history'])
    assert len(extended['outcomes']) == 2 and len(extended['tail_sensitivity']) == 4
    assert all(not extended['outcomes'][1][name] for name in TAIL_LABELS)
    assert extended['outcomes'][0]['tail_eligible']
    assert not extended['outcomes'][1]['tail_eligible']
    assert extended['performance']['full_horizon_success_rate'] == .5


def test_sensitivity_configuration_validation():
    m = SampledMechanism(Parameter('p'), MechanismBinding({'p':0.}, {}), 0, 0)
    with pytest.raises(ValueError, match='requires'):
        benchmark_stability_method('audit', [m], root_seed=1, diagnostic_config=STABILITY,
                                   tail_tolerances=(.1,))
