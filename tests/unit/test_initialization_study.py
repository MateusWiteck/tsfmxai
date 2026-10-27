from dataclasses import replace

import numpy as np
import pytest

from tsfmxai.evaluation.initialization import (
    GROUPS, InitializationStudyConfig, amplitude_bound_for_history, history_admissible,
    mechanism_parts, paired_inputs, pair_distances, run_forgetting_cases,
    run_initialization_cases, sample_study_mechanisms, simulate_from_history,
    HistorySpec, make_initial_history, run_history_case, run_history_pair, generator_configs,
)
from tsfmxai.generators.heaps import build_heaps_var_mechanism


@pytest.mark.parametrize("value", [0., .5])
@pytest.mark.parametrize("view", ["history", "trajectories"])
def test_single_sample_history_is_visible(value, view):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tsfmxai.visualization.initialization_study import (
        plot_history_comparison, plot_startup_histories,
    )

    results = {"Original AST": {"examples": [{
        "mechanism": 0, "trajectory": 0, "actual_order": 1,
        "history": np.array([[value]]), "path": np.array([[.2, .1]]),
    }], "skipped": []}}
    if view == "history":
        fig = plot_startup_histories(results, "Single sample")
    else:
        cfg = InitializationStudyConfig(mechanisms=1, example_mechanisms=1)
        fig = plot_history_comparison(results, "Single sample", cfg)
    try:
        fig.canvas.draw()
        axis = fig.axes[0]
        point = next(line for line in axis.lines if len(line.get_xdata()) == 1)
        np.testing.assert_array_equal(point.get_xdata(), [0])
        np.testing.assert_array_equal(point.get_ydata(), [value])
        assert point.get_marker() == "o" and point.get_markersize() > 0
        assert axis.get_xlim()[0] < 0 < axis.get_xlim()[1]
        assert axis.get_ylim()[0] < value < axis.get_ylim()[1]
    finally:
        plt.close(fig)


@pytest.fixture
def config():
    return InitializationStudyConfig(mechanisms=2, trajectories=2, horizons=(12, 24),
                                     tail_steps=4, window_steps=4, forgetting_tail=4)


@pytest.mark.parametrize("group", GROUPS)
def test_all_groups_execute_all_three_studies(group, config):
    mechanisms = sample_study_mechanisms(group, config)
    for study in ("starting_values", "magnitude"):
        result = run_initialization_cases(group, mechanisms, config, study=study)
        assert result["rows"]
        for case in {row["case"] for row in result["rows"]}:
            rows = [r for r in result["rows"] if r["case"] == case]
            assert len(rows) == config.mechanisms * config.trajectories * len(config.horizons)
        if group == "Jacobian" and study == "magnitude":
            assert {r["case"] for r in result["skipped"]} == {2., 4.}
        if group != "Heaps" and study == "starting_values":
            assert result["skipped"][0]["case"] == "stationary joint"
    forgetting = run_forgetting_cases(group, mechanisms, config)
    assert len(forgetting["rows"]) == 3 * config.mechanisms * config.trajectories * len(config.horizons)
    if group == "Jacobian":
        assert all(r["weighted_bound_passed"] for r in forgetting["rows"])


def test_streams_ignore_global_rng_and_cases(config):
    m = sample_study_mechanisms("Original AST", config)[0]
    a = paired_inputs(m, config, 0, 1)
    np.random.seed(45)
    np.random.normal(size=1000)
    b = paired_inputs(m, config, 0, 1)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(a[1], paired_inputs(m, config, 0, 0)[1])


def test_jacobian_outside_box_rejected(config):
    m = sample_study_mechanisms("Jacobian", config)[0]
    _, _, order = mechanism_parts(m)
    history = np.full((1, order), 2.)
    assert not history_admissible(m, history)
    with pytest.raises(ValueError, match="outside"):
        simulate_from_history(m, history, np.zeros((1, 24)), config)


def test_output_bound_updates_without_changing_equation(config):
    m = sample_study_mechanisms("Shao-Wu: output", config)[0]
    _, _, order = mechanism_parts(m)
    assert amplitude_bound_for_history(m, np.full((1, order), 100.)) >= 100.
    assert m == sample_study_mechanisms("Shao-Wu: output", config)[0]


def test_ar1_known_forgetting_and_initial_prefix_exclusion(config):
    # u/sqrt(1+u**2) = 0.5.
    m = build_heaps_var_mechanism(np.array([[[1/np.sqrt(3)]]]), np.array([[.01]]))
    a, b = np.array([[1.]]), np.array([[-1.]])
    noise = np.random.default_rng(8).normal(0, .1, (1, 24))
    pa = simulate_from_history(m, a, noise, config)
    pb = simulate_from_history(m, b, noise, config)
    assert pa.shape == (1, 24)
    assert pa[0, 0] == pytest.approx(.5 + noise[0, 0])
    output, state = pair_distances(a, b, pa, pb, 24)
    np.testing.assert_allclose(output, 2 * .5**np.arange(1, 25), atol=1e-14)
    np.testing.assert_allclose(state, 2 * .5**np.arange(25), atol=1e-14)


def test_missing_path_distances_are_nan():
    a, b = np.zeros((1, 2)), np.ones((1, 2))
    output, state = pair_distances(a, b, np.ones((1, 1)), np.ones((1, 3)), 4)
    assert np.isnan(output[1:]).all() and np.isnan(state[2:]).all()


def test_exploded_pairs_stay_in_denominator(config):
    config = replace(config, explosion_limit=1e-12)
    result = run_forgetting_cases("Original AST", sample_study_mechanisms("Original AST", config), config)
    assert all(r["pair_exploded"] and not r["persistent_forgetting"] for r in result["rows"])
    assert all(np.isnan(r["state_gap"]) for r in result["rows"])


def test_cosine_truncation_keeps_same_period_and_adds_scalar(config):
    mechanism = sample_study_mechanisms("Original AST", config)[0]
    unit, _, seed = paired_inputs(mechanism, config, 0, 0)
    full = make_initial_history(HistorySpec("full", "cosine", .5, .3, 48, 1.), mechanism, unit, seed)
    for fraction in (.5, .25):
        part = make_initial_history(HistorySpec("part", "cosine", .5, .3, 48, fraction), mechanism, unit, seed)
        np.testing.assert_array_equal(part, full[:, :int(48*fraction)])
    np.testing.assert_allclose(full[0], .3 + .5*np.cos(2*np.pi*np.arange(48)/48))


def test_cosine_insufficient_memory_is_reported(config):
    extended = replace(config, minimum_lag=12, maximum_lag=12)
    mechanisms = sample_study_mechanisms("Original AST", extended)
    result = run_history_case("Original AST", mechanisms, extended,
                              HistorySpec("too short", "cosine", period=16, fraction=.25))
    assert not result["rows"]
    assert sum(r["skipped_trajectories"] for r in result["skipped"]) == 4


def test_support_exclusions_do_not_select_favorable_draws(config):
    mechanisms = sample_study_mechanisms("Jacobian", config)
    result = run_history_case("Jacobian", mechanisms, config, HistorySpec("wide", "uniform", 2.))
    assert not result["rows"] and len(result["skipped"]) == config.mechanisms
    cosine = run_history_case("Jacobian", mechanisms, config,
                               HistorySpec("shifted", "cosine", .5, .5, 48, .25))
    assert len(cosine["rows"]) == 8
    assert not cosine["skipped"]


def test_long_memory_settings_apply_to_each_family(config):
    extended = replace(config, minimum_lag=4, maximum_lag=8, heaps_fixed_order=True)
    configs = generator_configs(extended)
    assert configs["Heaps"].minimum_order == configs["Heaps"].maximum_order == 8
    for group in GROUPS:
        for mechanism in sample_study_mechanisms(group, extended):
            assert 4 <= mechanism_parts(mechanism)[2] <= 8


def test_cosine_pair_distances_use_only_actual_lag_state(config):
    mechanism = build_heaps_var_mechanism(np.array([[[1/np.sqrt(3)]]]), np.array([[.01]]))
    left = HistorySpec("full", "cosine", .5, 0., 48, 1.)
    right = HistorySpec("half", "cosine", .5, 0., 48, .5)
    result = run_history_pair("Heaps", [mechanism], config, left, right)
    example = result["examples"][0]
    initial_gap = abs(example["history"][0, -1] - example["history_b"][0, -1])
    np.testing.assert_allclose(result["curves"][0]["state_gap"], initial_gap * .5**np.arange(25), atol=1e-14)
    assert example["history"].shape[1] == 48 and example["history_b"].shape[1] == 24
    assert example["path"].shape == (1, 24)


def test_failed_new_cases_stay_in_metrics(config):
    failed = replace(config, explosion_limit=1e-12)
    mechanisms = sample_study_mechanisms("Original AST", failed)
    result = run_history_case("Original AST", mechanisms, failed, HistorySpec("zero", "zero"))
    assert all(r["exploded"] for r in result["rows"])
    assert len(result["examples"]) == config.mechanisms * config.trajectories
