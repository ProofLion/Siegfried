"""Small integration checks; no SHIRT images, CNN weights, or training needed."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from evaluation import evaluate, position_error_m, rotation_error_deg, select_origins
import main
import model_adapters
from results import save_results, summarize


@pytest.fixture
def frames():
    return [
        {"image_path": Path(f"frame{index}.png"),
         "pose": np.array([index, 2 * index, 3 * index, 1, 0, 0, 0], dtype=float)}
        for index in range(20)
    ]


@pytest.fixture
def settings(tmp_path):
    return {
        "run": "compare", "history": 3, "horizons": [1, 2], "mode": "direct",
        "start_index": 0, "samples": 3, "stride": 1,
        "trajectory": "roe1", "domain": "synthetic", "checkpoint": Path("unused.pt"),
        "output_dir": tmp_path / "outputs",
    }


def moving_gdf(history, steps_ahead, dt_seconds):
    assert history.shape == (3, 3)
    return history[-1] + np.array([1, 2, 3]) * steps_ahead


def fake_cnn(path):
    index = int(path.stem.removeprefix("frame"))
    return np.array([index + 0.5, 2 * index, 3 * index, 1, 0, 0, 0])


def test_metrics_use_meters_and_sign_invariant_normalized_rotations():
    truth = np.array([0, 0, 0, 2, 0, 0, 0], dtype=float)
    guess = np.array([3, 4, 0, -3, 0, 0, 0], dtype=float)
    assert position_error_m(guess, truth) == pytest.approx(5)
    assert rotation_error_deg(guess, truth) == pytest.approx(0)
    guess[3:] = [0, 4, 0, 0]
    assert rotation_error_deg(guess, truth) == pytest.approx(180)
    guess[3:] = [1, 1, 0, 0]
    assert rotation_error_deg(guess, truth) == pytest.approx(90)
    guess[3:] = 0
    with pytest.raises(ValueError, match="zero quaternion"):
        rotation_error_deg(guess, truth)


def test_adapter_uses_independent_scalar_filters(monkeypatch):
    instances = []

    class ScalarFilter:
        def __init__(self, base, residual, window_size):
            assert window_size == 3
            self.values = []
            instances.append(self)

        def update(self, value):
            assert isinstance(value, float)
            self.values.append(value)

        def predict(self, steps_ahead):
            return self.values[-1] + steps_ahead

    monkeypatch.setattr(model_adapters, "GDF", ScalarFilter)
    history = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]])
    np.testing.assert_allclose(model_adapters.predict_gdf(history, 2, 5), [5, 32, 302])
    model_adapters.predict_gdf(history, 1, 5)
    assert len(instances) == 6  # No state shared between coordinates or calls.
    assert instances[0].values == [1, 2, 3]


@pytest.mark.parametrize("mode", ["direct", "recursive"])
def test_compare_separates_tasks_and_reuses_only_observed_images(frames, settings, mode):
    settings["mode"] = mode
    calls = []

    def cnn(path):
        calls.append(path)
        return fake_cnn(path)

    rows = evaluate(frames, 5, settings, gdf_predict=moving_gdf, cnn_estimate=cnn)
    assert len(rows) == 27  # 3 current poses + 3 windows * 2 horizons * 4 forecasts.
    assert sorted(path.stem for path in calls) == [f"frame{i}" for i in range(5)]
    assert all(row["status"] == "ok" and row["comparison_included"] for row in rows)
    assert all(row["position_error_m"] == pytest.approx(0) for row in rows if row["method"] == "gdf")
    assert all(row["position_error_m"] == pytest.approx(0.5) for row in rows if row["method"] == "cnn_gdf")
    assert all(row["horizon_frames"] == 0 for row in rows if row["task"] == "pose_estimation")
    assert all(row["rotation_error_deg"] is None for row in rows if row["task"] == "forecast")
    for row in rows:
        assert row["target_index"] == row["origin_index"] + row["horizon_frames"]
        assert row["horizon_seconds"] == row["horizon_frames"] * 5


def test_model_failures_are_recorded_and_common_cases_match(frames, settings):
    def sometimes_fails(history, steps_ahead, dt_seconds):
        if history[-1, 0] == 2:
            raise FloatingPointError("example numerical failure")
        return moving_gdf(history, steps_ahead, dt_seconds)

    rows = evaluate(frames, 5, settings, gdf_predict=sometimes_fails, cnn_estimate=fake_cnn)
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(failed) == 2
    assert all(row["method"] == "gdf" and row["pred_x"] is None for row in failed)
    assert all("example numerical failure" in row["error"] for row in failed)
    first_window = [row for row in rows if row["task"] == "forecast" and row["origin_index"] == 2]
    assert all(not row["comparison_included"] for row in first_window)
    assert any(row["status"] == "ok" for row in first_window)  # Baseline successes remain in raw data.
    for row in summarize(rows):
        if row["task"] == "forecast":
            assert row["included"] == 2 and row["excluded"] == 1


def test_failed_cnn_images_are_cached(frames, settings):
    calls = []

    def cnn(path):
        calls.append(path)
        if path.stem == "frame1":
            raise ValueError("unreadable image")
        return fake_cnn(path)

    rows = evaluate(frames, 5, settings, gdf_predict=moving_gdf, cnn_estimate=cnn)
    assert calls.count(Path("frame1.png")) == 1
    assert any("unreadable image" in row["error"] for row in rows)


def test_nonfinite_predictions_are_failures_not_zero_errors(frames, settings):
    settings["run"] = "gdf"
    rows = evaluate(frames, 5, settings, gdf_predict=lambda *args: np.full(3, np.nan))
    assert all(row["position_error_m"] is None for row in rows if row["method"] == "gdf")
    assert all(row["included"] == 0 for row in summarize(rows))


def test_cnn_only_needs_no_future_labels(frames, settings):
    settings.update(run="cnn", samples=1, horizons=[100])
    rows = evaluate(frames[:3], 5, settings, cnn_estimate=fake_cnn)
    assert len(rows) == 1 and rows[0]["target_index"] == 2


def test_invalid_selection_is_rejected_before_inference(frames, settings):
    settings["samples"] = 100
    with pytest.raises(ValueError, match="exceed"):
        evaluate(frames, 5, settings)
    settings.update(samples=1, history=2)
    with pytest.raises(ValueError, match="at least three"):
        select_origins(len(frames), settings)


def test_cli_checks_bounds_before_loading_checkpoint(monkeypatch, frames):
    monkeypatch.setattr(main, "load_trajectory", lambda *args: (frames, 5))
    monkeypatch.setattr(main, "make_cnn_estimator", lambda *args: pytest.fail("Must check bounds first"))
    with pytest.raises(SystemExit) as error:
        main.main(["--run", "compare", "--samples", "100"])
    assert error.value.code == 2


def test_missing_checkpoint_has_readable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="CNN checkpoint not found"):
        model_adapters.make_cnn_estimator(tmp_path / "missing.pt")


def test_exports_include_metrics_settings_and_headless_figures(frames, settings, monkeypatch):
    rows = evaluate(frames, 5, settings, gdf_predict=moving_gdf, cnn_estimate=fake_cnn)
    folder, summary = save_results(rows, settings, 5)
    assert len(list(csv.DictReader((folder / "predictions.csv").open()))) == len(rows)
    assert len(list(csv.DictReader((folder / "summary.csv").open()))) == len(summary)
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["result_type"] == "development_diagnostic"
    assert metadata["settings"]["checkpoint"] == "unused.pt"
    for name in ("cnn_errors.png", "forecast_errors.png", "forecast_example.png"):
        with Image.open(folder / name) as image:
            assert image.width > 0 and image.height > 0

    # A second run creates another folder rather than overwriting the first.
    monkeypatch.setattr("results.save_figures", lambda *args: None)
    second, _ = save_results(rows, settings, 5)
    assert folder != second and (folder / "summary.csv").is_file()


def test_all_failed_comparisons_export_na_and_empty_figures(frames, settings):
    settings.update(run="gdf", samples=1)
    rows = evaluate(frames, 5, settings, gdf_predict=lambda *args: np.full(3, np.nan))
    folder, summary = save_results(rows, settings, 5)
    assert all(row["mean_position_error_m"] is None for row in summary)
    assert (folder / "forecast_errors.png").is_file()
    assert (folder / "forecast_example.png").is_file()
