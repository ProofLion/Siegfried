"""Unscented Kalman Filter: teammate implementation goes here."""


def predict(history, steps_ahead, dt_seconds):
    """Input: (N, 7) pose history. Return one (7,) pose at the requested horizon."""
    # TODO: Define the UKF state/noise, process the history, and forecast.
    # Poses: [x, y, z, qw, qx, qy, qz]; meters and a unit quaternion.
    raise NotImplementedError("TODO: implement the Unscented Kalman Filter.")
