"""Evaluate current-image poses and future positions as separate tasks."""

from collections import defaultdict

import numpy as np

import baseline
from model_adapters import predict_gdf

POSE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz")


def position_error_m(prediction, truth):
    """Euclidean position distance for one pose/position or a batch."""
    return np.linalg.norm(np.asarray(prediction)[..., :3] - np.asarray(truth)[..., :3], axis=-1)


def rotation_error_deg(prediction, truth):
    """Shortest rotation angle; normalize both quaternions and ignore their sign."""
    guess_q = np.asarray(prediction, dtype=float)[..., 3:]
    true_q = np.asarray(truth, dtype=float)[..., 3:]
    guess_norm = np.linalg.norm(guess_q, axis=-1, keepdims=True)
    true_norm = np.linalg.norm(true_q, axis=-1, keepdims=True)
    if np.any(guess_norm == 0) or np.any(true_norm == 0):
        raise ValueError("A zero quaternion has no rotation angle.")
    dot = np.sum((guess_q / guess_norm) * (true_q / true_norm), axis=-1)
    return np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def forecast(history, predict, horizon, dt_seconds, mode):
    """Return one final prediction (direct) or every future step (recursive)."""
    history = np.asarray(history, dtype=float).copy()
    if horizon < 1 or mode not in ("direct", "recursive"):
        raise ValueError("Use a positive horizon and 'direct' or 'recursive' mode.")
    if mode == "direct":
        return np.array([predict(history, horizon, dt_seconds)])

    predictions = []
    for _ in range(horizon):
        pose = np.asarray(predict(history.copy(), 1, dt_seconds), dtype=float)
        predictions.append(pose.copy())
        history = np.vstack((history[1:], pose))
    return np.array(predictions)


def select_origins(frame_count, settings):
    """Validate all requested windows before calling either teammate's model."""
    for name in ("history", "samples", "stride"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if settings["start_index"] < 0:
        raise ValueError("start-index must be nonnegative.")
    if not settings["horizons"] or min(settings["horizons"]) < 1:
        raise ValueError("horizons must be positive frame counts.")
    if settings["run"] not in ("cnn", "gdf", "combined", "compare"):
        raise ValueError("run must be cnn, gdf, combined, or compare.")
    if settings["mode"] not in ("direct", "recursive"):
        raise ValueError("mode must be direct or recursive.")
    if settings["run"] != "cnn" and settings["history"] < 3:
        raise ValueError("GDF requires a history of at least three frames.")
    first = settings["start_index"] + settings["history"] - 1
    origins = [first + sample * settings["stride"] for sample in range(settings["samples"])]
    future = 0 if settings["run"] == "cnn" else max(settings["horizons"])
    if origins[-1] + future >= frame_count:
        raise ValueError("Requested windows exceed the trajectory; reduce samples, stride, or horizon.")
    return origins


def prediction_row(frames, dt, task, method, source, origin, horizon, mode, prediction=None, error=None):
    """Use the same CSV columns for successes and failures; missing values stay blank."""
    target = origin + horizon
    truth = np.asarray(frames[target]["pose"], dtype=float)
    row = {
        "task": task, "method": method, "input_source": source, "mode": mode,
        "origin_index": origin, "target_index": target,
        "target_image": str(frames[target]["image_path"]),
        "horizon_frames": horizon, "horizon_seconds": horizon * dt,
        "position_error_m": None, "rotation_error_deg": None,
        "status": "failed", "error": "", "comparison_included": False,
    }
    row.update({f"true_{name}": float(value) for name, value in zip(POSE_NAMES, truth)})
    row.update({f"pred_{name}": None for name in POSE_NAMES})
    if error is not None:
        row["error"] = f"{type(error).__name__}: {error}"
        return row
    try:
        prediction = np.asarray(prediction, dtype=float)
        expected = (7,) if task == "pose_estimation" else (3,)
        if prediction.shape != expected or not np.isfinite(prediction).all():
            raise ValueError(f"Expected a finite prediction with shape {expected}.")
        row.update({f"pred_{name}": float(value) for name, value in zip(POSE_NAMES, prediction)})
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            row["position_error_m"] = float(position_error_m(prediction, truth))
            if task == "pose_estimation":
                row["rotation_error_deg"] = float(rotation_error_deg(prediction, truth))
        row["status"] = "ok"
    except Exception as problem:
        row["position_error_m"] = row["rotation_error_deg"] = None
        row["error"] = f"{type(problem).__name__}: {problem}"
    return row


def mark_common_cases(rows):
    """Forecast comparisons use exactly the same successful windows for every method."""
    groups = defaultdict(list)
    for row in rows:
        if row["task"] == "pose_estimation":
            row["comparison_included"] = row["status"] == "ok"
        else:
            key = (row["origin_index"], row["target_index"], row["mode"])
            groups[key].append(row)
    for group in groups.values():
        included = all(row["status"] == "ok" for row in group)
        for row in group:
            row["comparison_included"] = included


def evaluate(frames, dt_seconds, settings, gdf_predict=predict_gdf, cnn_estimate=None, progress=None):
    """Run selected tasks. Each model failure becomes a row, not a substituted prediction."""
    origins = select_origins(len(frames), settings)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("The dataset timestep must be positive and finite.")
    run = settings["run"]
    if run != "gdf" and cnn_estimate is None:
        raise ValueError("This run needs a CNN estimator.")
    horizons = sorted(set(settings["horizons"]))
    rows, cache = [], {}

    def cnn_pose(index):
        # Cache failures too, so an unreadable image is not retried for each window.
        if index not in cache:
            try:
                pose = np.asarray(cnn_estimate(frames[index]["image_path"]), dtype=float)
                if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) == 0:
                    raise ValueError("CNN must return a finite seven-value pose with a nonzero quaternion.")
                cache[index] = pose.copy()
            except Exception as error:
                cache[index] = error
        if isinstance(cache[index], Exception):
            raise cache[index]
        return cache[index].copy()

    jobs = []
    if run in ("gdf", "compare"):
        jobs.extend((("gdf", "labels", gdf_predict), ("persistence_labels", "labels", baseline.predict)))
    if run in ("combined", "compare"):
        jobs.extend((("cnn_gdf", "cnn", gdf_predict), ("persistence_cnn", "cnn", baseline.predict)))

    for sample, origin in enumerate(origins):
        if progress:
            progress(f"Window {sample + 1}/{len(origins)}: last observed frame index {origin}")
        if run in ("cnn", "compare"):
            try:
                prediction, error = cnn_pose(origin), None
            except Exception as problem:
                prediction, error = None, problem
            rows.append(prediction_row(frames, dt_seconds, "pose_estimation", "cnn", "images",
                                       origin, 0, "current", prediction, error))
        for method, source, predict in jobs:
            first = origin if method.startswith("persistence") else origin - settings["history"] + 1
            try:
                if source == "labels":
                    history = np.array([frame["pose"][:3] for frame in frames[first:origin + 1]])
                else:
                    history = np.array([cnn_pose(index)[:3] for index in range(first, origin + 1)])
                history_error = None
            except Exception as error:
                history, history_error = None, error
            for horizon in horizons:
                prediction, error = None, history_error
                if error is None:
                    try:
                        prediction = forecast(history, predict, horizon, dt_seconds, settings["mode"])[-1]
                    except Exception as problem:
                        error = problem
                rows.append(prediction_row(frames, dt_seconds, "forecast", method, source, origin,
                                           horizon, settings["mode"], prediction, error))
    mark_common_cases(rows)
    return rows
