"""Simple baseline for checking the pipeline."""


def predict(history, steps_ahead, dt_seconds):
    """Repeat the last pose; this model assumes no movement."""
    return history[-1].copy()
