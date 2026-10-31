"""Check the important difference between recursive and direct prediction."""

import numpy as np

from main import forecast


def test_recursive_uses_predictions_and_keeps_history_size():
    history = np.array([[0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0]], dtype=float)
    seen_positions = []

    def moving_model(observed, steps_ahead, dt_seconds):
        assert observed.shape == (2, 7)
        assert steps_ahead == 1
        assert dt_seconds == 5.0
        seen_positions.append(observed[-1, 0])
        pose = observed[-1].copy()
        pose[0] += 1
        return pose

    predictions = forecast(history, moving_model, 3, 5.0, "recursive")
    np.testing.assert_allclose(predictions[:, 0], [2, 3, 4])
    assert seen_positions == [1, 2, 3]
    np.testing.assert_allclose(history[:, 0], [0, 1])


def test_direct_calls_model_once_with_full_horizon():
    calls = []
    history = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=float)

    def direct_model(observed, steps_ahead, dt_seconds):
        calls.append(steps_ahead)
        pose = observed[-1].copy()
        pose[0] += steps_ahead
        return pose

    predictions = forecast(history, direct_model, 10, 5.0, "direct")
    assert calls == [10]
    assert predictions.shape == (1, 7)
    assert predictions[0, 0] == 10
