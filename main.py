"""Run small CNN/GDF development experiments without editing model files."""

import argparse
from pathlib import Path

from data_loader import load_trajectory
from evaluation import evaluate, forecast, select_origins
from model_adapters import make_cnn_estimator
from results import print_summary, save_results

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["cnn", "gdf", "combined", "compare"], default="gdf")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "shirtv1")
    parser.add_argument("--trajectory", choices=["roe1", "roe2"], default="roe1")
    parser.add_argument("--domain", choices=["synthetic", "lightbox"], default="synthetic")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pose_model.pt")
    parser.add_argument("--history", type=int, default=10, help="Consecutive observed frames")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10], help="Future frame offsets")
    parser.add_argument("--mode", choices=["direct", "recursive"], default="recursive")
    parser.add_argument("--start-index", type=int, default=0, help="First frame of the first history (zero-based)")
    parser.add_argument("--samples", type=int, default=20, help="Number of observation windows")
    parser.add_argument("--stride", type=int, default=10, help="Frames between window starts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    return parser


def main(argv=None):
    parser = build_parser()
    settings = vars(parser.parse_args(argv))
    try:
        frames, dt = load_trajectory(settings["data_dir"], settings["trajectory"], settings["domain"])
        select_origins(len(frames), settings)  # Check bounds before loading a CNN.
        estimator = None if settings["run"] == "gdf" else make_cnn_estimator(settings["checkpoint"])
        print(f"Development diagnostics: {settings['run']} / {settings['trajectory']} / {settings['domain']}")
        print(f"Timestep: {dt:g} s. Model files and checkpoint are used as-is; no training.")
        rows = evaluate(frames, dt, settings, cnn_estimate=estimator,
                        progress=lambda message: print(message, flush=True))
        folder, summary = save_results(rows, settings, dt)
        print_summary(summary)
        failures = sum(row["status"] == "failed" for row in rows)
        print(f"\nRecorded failures: {failures}. Details are in predictions.csv.")
        print(f"Results: {folder}")
        if settings["run"] != "gdf":
            print("CNN checkpoint training split is unrecorded; treat CNN results as development diagnostics.")
        return 1 if failures else 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Experiment error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
