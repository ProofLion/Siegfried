"""Small wrappers around the teammates' existing models; no training here."""

from pathlib import Path
import warnings

import numpy as np

from grey_dynamic_filter import GDF
from models.gm_mn import GM_MN


def predict_gdf(history, steps_ahead, dt_seconds):
    """Forecast x/y/z with three independent scalar GDFs, using frame steps."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or history.shape[1] != 3 or len(history) < 3:
        raise ValueError("GDF needs at least three position observations, shape (N, 3).")
    if not np.isfinite(history).all():
        raise ValueError("GDF observations must be finite.")

    # The current grey model accepts scalar sequences, not whole pose arrays.
    # Fresh instances prevent state from leaking between axes or windows.
    position = []
    with warnings.catch_warnings(), np.errstate(divide="raise", invalid="raise", over="raise"):
        warnings.simplefilter("error", RuntimeWarning)
        for axis in range(3):
            model = GDF(GM_MN(1, 1), GM_MN(1, 1), window_size=len(history))
            for value in history[:, axis]:
                model.update(float(value))
            position.append(float(model.predict(steps_ahead)))
    position = np.array(position)
    if not np.isfinite(position).all():
        raise ValueError("GDF returned a nonfinite position.")
    return position


def make_cnn_estimator(checkpoint):
    """Use the existing inference function; importing GDF does not require torch."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {checkpoint}")
    try:
        from pose_estimator import estimate_pose
    except ImportError as error:
        raise RuntimeError("CNN dependencies are unavailable. Run 'uv sync --locked'.") from error

    def estimate(image_path):
        return estimate_pose(image_path, model_path=checkpoint)

    return estimate
