import json

import numpy as np
import pytest

from frontend.app import STORE, FrontendHandler
from tsfmxai.visualization.studio_multiseries import (
    DEPENDENCIES, build_system_config, sample_studio_multiseries,
    sample_studio_trajectories, system_payload, trajectory_payload,
)


class Capture:
    _write_json = lambda self, payload: setattr(self, "payload", payload)  # noqa: E731


def small_system(**overrides):
    request = {"seed": 7, "channels": 3, "maximum_operators": 3, "maximum_lag": 4,
               "max_degree": 2, "shao_wu_probability": 1.0,
               "jacobian_probability": 0., "heaps_probability": 0., **overrides}
    return request, sample_studio_multiseries(request)


def test_trajectory_payload_preserves_multivariate_axes() -> None:
    trajectories = np.asarray([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                               [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]]], dtype=np.float32)
    payload = trajectory_payload(trajectories, seed=17, history_length=1)
    assert payload["seed"] == 17
    assert payload["times"] == [0, 1, 2]
    assert payload["generated_start"] == 1
    assert np.asarray(payload["trajectories"]).shape == (2, 2, 3)
    assert np.asarray(payload["summary"]["mean"]).shape == (2, 3)
    assert payload["summary"]["minimum"] == [1.0, 10.0]
    assert payload["summary"]["maximum"] == [6.0, 60.0]


@pytest.mark.parametrize("dependency", DEPENDENCIES)
@pytest.mark.parametrize("channels", [1, 4])
def test_system_payload_describes_every_channel(dependency, channels) -> None:
    _, system = small_system(channels=channels, dependency=dependency)
    payload = system_payload(system, "abc")
    json.dumps(payload, allow_nan=False)
    assert payload["num_channels"] == channels == len(payload["equations"])
    assert np.asarray(payload["dependency_lags"], dtype=object).shape[:2] == (channels, channels)
    for equation in payload["equations"]:
        assert equation["treatment"]["method"] == "shao_wu"
        assert set(equation["effective_sources"]) <= set(equation["allowed_sources"])
        assert equation["tree"]["type"]
    if dependency == "independent":
        assert payload["summary"]["allowed_edges"] == 0


def test_studio_uses_the_training_generator_exactly() -> None:
    from tsfmxai.generators.multiseries import sample_multiseries_system

    request, system = small_system()
    config, seed = build_system_config(request)
    assert system == sample_multiseries_system(config, seed=seed)


def test_trajectories_are_reproducible_and_have_context_plus_horizon_steps() -> None:
    _, system = small_system(channels=2)
    request = {"trajectory_count": 3, "context_length": 10, "prediction_length": 4, "seed": 5}
    first, _ = sample_studio_trajectories(system, request)
    second, _ = sample_studio_trajectories(system, request)
    assert first == second
    shape = np.asarray(first["trajectories"]).shape
    assert shape[1:] == (2, 14)
    assert first["stats"]["accepted"] + first["stats"]["skipped"] == 3
    assert len(first["initial_history"]) == shape[0] == len(first["trajectory_indices"])
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("bad", [
    {"channels": 0}, {"channels": 99}, {"dependency": "nope"}, {"shao_wu_probability": 2},
    {"channels": 2.5}, {"allow_no_lags": 1}, {"noise_minimum": 1, "noise_maximum": 0.5},
])
def test_invalid_system_requests_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        sample_studio_multiseries(bad)


@pytest.mark.parametrize("bad", [
    {"trajectory_count": 0}, {"context_length": 0}, {"initialization": "nope"},
    {"context_length": 1000, "prediction_length": 1000},
])
def test_invalid_trajectory_requests_are_rejected(bad) -> None:
    _, system = small_system(channels=1)
    with pytest.raises(ValueError):
        sample_studio_trajectories(system, bad)


def test_http_handlers_round_trip_through_the_store() -> None:
    handler = Capture()
    FrontendHandler._sample_system(handler, {"seed": 3, "channels": 2, "maximum_lag": 3,
                                             "shao_wu_probability": 1.0, "jacobian_probability": 0., "heaps_probability": 0.})
    system_id = handler.payload["id"]
    assert len(handler.payload["equations"]) == 2
    FrontendHandler._sample_trajectories(handler, {"system_id": system_id, "trajectory_count": 2,
                                                   "context_length": 8, "prediction_length": 4})
    assert np.asarray(handler.payload["trajectories"]).shape[1:] == (2, 12)
    with pytest.raises(ValueError, match="não existe mais"):
        STORE.get("missing")


def test_studio_replaces_an_exploding_system_and_reports_the_accepted_one(monkeypatch) -> None:
    import tsfmxai.datasets.multiseries as generation
    actual = generation.simulate_innovation_path
    calls = [0]
    def fail_start(*args, **kwargs):
        calls[0] += 1
        if calls[0] <= 4:
            return np.full_like(args[3], np.nan)
        return actual(*args, **kwargs)
    monkeypatch.setattr(generation, "simulate_innovation_path", fail_start)
    request = {"seed": 5, "channels": 3, "maximum_operators": 4, "maximum_lag": 4, "max_degree": 3,
               "shao_wu_probability": 0.0}
    system = sample_studio_multiseries(request)
    trajectories = {"trajectory_count": 3, "context_length": 20, "prediction_length": 10, "seed": 5,
                    "initialization": "uniform"}
    payload, final = sample_studio_trajectories(system, trajectories)
    assert final is not system and payload["stats"]["replaced_systems"] > 0
    calls[0] = 0
    handler = Capture()
    FrontendHandler._sample_system(handler, {**request, "channels": 3})
    FrontendHandler._sample_trajectories(handler, {**trajectories, "system_id": handler.payload["id"]})
    assert handler.payload["replacement_system"]["id"] != ""
    assert handler.payload["replacement_system"]["seed"] != request["seed"]


def test_studio_defaults_to_random_families_and_reports_components() -> None:
    system = sample_studio_multiseries({"channels": 3, "maximum_lag": 6, "shao_wu_probability": 1.0, "jacobian_probability": 0., "heaps_probability": 0.})
    payload, _ = sample_studio_trajectories(system, {"trajectory_count": 4, "context_length": 12,
                                                     "prediction_length": 4})
    assert payload["initialization"] == "random"
    components = payload["initialization_components"]
    assert len(components) == 4 and all(len(parts) == 3 for parts in components)
    json.dumps(payload)
    fixed, _ = sample_studio_trajectories(system, {"trajectory_count": 2, "context_length": 12,
                                                   "prediction_length": 4, "initialization": "uniform"})
    assert fixed["initialization_components"] == [None, None]
    with pytest.raises(ValueError, match="famílias"):
        sample_studio_trajectories(system, {"constant_minimum": 2.0, "constant_maximum": 1.0})
