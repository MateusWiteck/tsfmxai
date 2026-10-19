import json

import numpy as np
import pytest

from tsfmxai.generators import RandomTreeConfig, initialize_shao_wu_history
from tsfmxai.representations import Lag, walk
from tsfmxai.visualization.studio_generation import METHODS, sample_studio_system, sample_studio_trajectories
from frontend.app import FrontendHandler


@pytest.mark.parametrize("method", METHODS)
def test_studio_methods_generate_reproducible_two_channel_batches(method):
    system = sample_studio_system(RandomTreeConfig(num_series=2, maximum_operators=2,
                                                   maximum_lag=3), method=method, seed=19,
                                  innovation=.05)
    kwargs = dict(trajectory_count=3, generated_steps=16, root_seed=47, innovation_std=.05)
    a = sample_studio_trajectories(system, **kwargs)
    np.testing.assert_array_equal(a, sample_studio_trajectories(system, **kwargs))
    assert a.shape == (3, 2, system.history_length + 16)
    assert np.isfinite(a).all()
    assert bool(system.certificates) == (method not in ("none", "coefficient_gate"))
    assert all(c["holds"] for c in system.certificates)
    if method == "shao_wu":
        initial_seed, _ = np.random.SeedSequence([47, 0]).spawn(2)
        expected_history = initialize_shao_wu_history(
            system.retained,
            seed=int(initial_seed.generate_state(1, dtype=np.uint64)[0]),
        )
        np.testing.assert_array_equal(a[0, :, :system.history_length], expected_history)
        assert "burn_in_steps" not in system.certificates[0]
    if method == "contraction":
        assert np.abs(a).max() <= 1 + 1e-12
        for i, equation in enumerate(system.equations):
            assert all(n.variable == i for n in walk(equation.expression) if isinstance(n, Lag))


@pytest.mark.parametrize("method", METHODS)
def test_frontend_routes_serialize_all_method_trees_and_certificates(method):
    class Handler:
        _write_json = lambda self, payload: setattr(self, "payload", payload)
    handler = Handler()
    FrontendHandler._sample_system(handler, {"stability_method": method, "seed": 19,
                                            "maximum_operators": 2, "maximum_lag": 3,
                                            "innovation_std": .05})
    mechanism = handler.payload
    assert mechanism["stability_method"] == method
    assert len(mechanism["equations"]) == 2
    json.dumps(mechanism, allow_nan=False)
    FrontendHandler._sample_trajectories(handler, {"mechanism_id": mechanism["id"],
                                                 "trajectory_count": 2, "generated_steps": 8,
                                                 "innovation_std": .05})
    assert len(handler.payload["trajectories"]) == 2
    json.dumps(handler.payload, allow_nan=False)


def test_unknown_method_and_changed_certified_noise_are_rejected():
    config = RandomTreeConfig(num_series=2, maximum_lag=2)
    with pytest.raises(ValueError, match="desconhecido"):
        sample_studio_system(config, method="invalid", seed=7)
    system = sample_studio_system(config, method="shao_wu", seed=7, innovation=.05)
    with pytest.raises(ValueError, match="inovação mudou"):
        sample_studio_trajectories(system, trajectory_count=1, generated_steps=3,
                                  root_seed=7, innovation_std=.1)
