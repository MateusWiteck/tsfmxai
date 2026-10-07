import numpy as np

from frontend.app import _trajectory_payload


def test_frontend_payload_preserves_multivariate_axes() -> None:
    trajectories = np.asarray(
        [
            [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
            [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
        ]
    )

    payload = _trajectory_payload(trajectories, seed=17, history_length=1)

    assert payload["seed"] == 17
    assert payload["times"] == [0, 1, 2]
    assert payload["generated_start"] == 1
    assert np.asarray(payload["trajectories"]).shape == (2, 2, 3)
    assert np.asarray(payload["summary"]["mean"]).shape == (2, 3)
    assert payload["summary"]["minimum"] == [1.0, 10.0]
    assert payload["summary"]["maximum"] == [6.0, 60.0]
