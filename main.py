"""Run a small SHIRT forecast. Change the settings below to experiment."""

from pathlib import Path

import numpy as np

import baseline
import grey_dynamic_filter
import unscented_kalman_filter
from data_loader import load_trajectory

DATA_DIR = Path(__file__).resolve().parent / "shirtv1"
TRAJECTORY = "roe1"       # "roe1" or "roe2"
DOMAIN = "synthetic"     # "synthetic" or "lightbox"
MODEL = baseline         # baseline, grey_dynamic_filter, or unscented_kalman_filter
MODE = "recursive"       # "direct" or "recursive"
START_INDEX = 0
HISTORY_LENGTH = 10
HORIZON = 10             # Frames ahead of the last observation


def forecast(history, predict, horizon, dt_seconds, mode):
    """Return one final pose (direct) or every future pose (recursive)."""
    if horizon < 1 or mode not in ("direct", "recursive"):
        raise ValueError("Use a positive horizon and 'direct' or 'recursive' mode.")
    if mode == "direct":
        return np.array([predict(history.copy(), horizon, dt_seconds)])

    predictions = []
    history = history.copy()
    for _ in range(horizon):
        pose = predict(history, 1, dt_seconds)
        predictions.append(pose)
        # Feed predictions back in; never use future true poses.
        history = np.vstack((history[1:], pose))
    return np.array(predictions)


def main():
    frames, dt_seconds = load_trajectory(DATA_DIR, TRAJECTORY, DOMAIN)
    end = START_INDEX + HISTORY_LENGTH
    if START_INDEX < 0 or HISTORY_LENGTH < 1 or end + HORIZON > len(frames):
        raise ValueError("Choose a history and horizon that fit within the data.")

    # TODO: Use pose_estimator.estimate_pose(frame["image_path"]) when ready.
    history = np.array([frame["pose"] for frame in frames[START_INDEX:end]])
    predictions = forecast(history, MODEL.predict, HORIZON, dt_seconds, MODE)

    print(f"{TRAJECTORY} / {DOMAIN}: {MODEL.__name__}, {MODE}")
    print(f"Horizon: {HORIZON} frames ({HORIZON * dt_seconds:g} seconds)")
    print("Demo uses true pose labels as observations.")
    print("Pose: [x, y, z, qw, qx, qy, qz] (meters, scalar-first quaternion)")
    offsets = [HORIZON] if MODE == "direct" else range(1, HORIZON + 1)
    for offset, prediction in zip(offsets, predictions):
        print(f"\n+{offset} frames")
        print("Predicted:", np.round(prediction, 5))
        print("True:     ", np.round(frames[end - 1 + offset]["pose"], 5))
    # TODO: Add pose error measurements. Image similarity is a separate function.


if __name__ == "__main__":
    main()
