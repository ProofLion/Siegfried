"""Print comparable tables and save CSV/PNG development results."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

METHOD_LABELS = {
    "cnn": "CNN",
    "gdf": "GDF (true poses)",
    "cnn_gdf": "CNN -> GDF",
    "persistence_labels": "Last true position",
    "persistence_cnn": "Last CNN position",
}


def summarize(rows):
    """Summarize common successful cases, with raw success/exclusion counts visible."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["task"], row["method"], row["input_source"], row["mode"], row["horizon_frames"])
        groups[key].append(row)
    summary = []
    for (task, method, source, mode, horizon), group in sorted(groups.items()):
        included = [row for row in group if row["comparison_included"]]
        positions = [row["position_error_m"] for row in included]
        rotations = [row["rotation_error_deg"] for row in included if row["rotation_error_deg"] is not None]
        summary.append({
            "task": task, "method": method, "input_source": source, "mode": mode,
            "horizon_frames": horizon, "horizon_seconds": group[0]["horizon_seconds"],
            "attempted": len(group), "successful": sum(row["status"] == "ok" for row in group),
            "included": len(included), "excluded": len(group) - len(included),
            "mean_position_error_m": float(np.mean(positions)) if positions else None,
            "median_position_error_m": float(np.median(positions)) if positions else None,
            "mean_rotation_error_deg": float(np.mean(rotations)) if rotations else None,
        })
    return summary


def print_summary(summary):
    def number(value):
        return "N/A" if value is None else f"{value:.4f}"

    for task in ("pose_estimation", "forecast"):
        selected = [row for row in summary if row["task"] == task]
        if not selected:
            continue
        print(f"\n{task.replace('_', ' ').title()} (position: m; rotation: degrees)")
        print(f"{'Method':<20} {'H(fr/s)':<11} {'Used/OK/Tried':<15} {'Pos mean':>10} {'Pos median':>11} {'Rot mean':>10}")
        for row in selected:
            counts = f"{row['included']}/{row['successful']}/{row['attempted']}"
            horizon = f"{row['horizon_frames']}/{row['horizon_seconds']:g}"
            label = METHOD_LABELS[row["method"]]
            print(f"{label:<20} {horizon:<11} {counts:<15} "
                  f"{number(row['mean_position_error_m']):>10} "
                  f"{number(row['median_position_error_m']):>11} "
                  f"{number(row['mean_rotation_error_deg']):>10}")
    print("\nUsed = common successful cases; OK = individual successes; Tried = attempts.")
    if any(row["task"] == "forecast" for row in summary):
        print("Forecast orientation: not implemented. N/A means unavailable, not zero error.")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(folder, rows, summary, dt_seconds, settings):
    # File output works without a desktop window, including automated tests.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    context = f"{settings['trajectory']} / {settings['domain']}"
    current = [row for row in rows if row["task"] == "pose_estimation" and row["status"] == "ok"]
    if any(row["task"] == "pose_estimation" for row in rows):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        times = [row["target_index"] * dt_seconds for row in current]
        axes[0].plot(times, [row["position_error_m"] for row in current], "o-")
        axes[1].plot(times, [row["rotation_error_deg"] for row in current], "o-")
        axes[0].set_ylabel("Position error (m)")
        axes[1].set_ylabel("Rotation error (degrees)")
        axes[1].set_xlabel("Time from first frame (s)")
        fig.suptitle(f"CNN current-image estimates — development diagnostics\n{context}")
        if not current:
            axes[0].text(0.5, 0.5, "No successful estimates", ha="center", transform=axes[0].transAxes)
        fig.savefig(folder / "cnn_errors.png", dpi=160)
        plt.close(fig)

    future_summary = [row for row in summary if row["task"] == "forecast"]
    if not future_summary:
        return
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for method in sorted({row["method"] for row in future_summary}):
        points = [row for row in future_summary if row["method"] == method and row["included"]]
        points.sort(key=lambda row: row["horizon_frames"])
        if points:
            axis.plot([row["horizon_seconds"] for row in points],
                      [row["mean_position_error_m"] for row in points], "o-", label=METHOD_LABELS[method])
    axis.set(xlabel="Time ahead (s)", ylabel="Mean position error (m)",
             title=f"Common successful forecasts — development diagnostics\n"
                   f"{context} | {settings['mode']} | history: {settings['history']} frames")
    if axis.lines:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No common successful forecasts", ha="center", transform=axis.transAxes)
    fig.savefig(folder / "forecast_errors.png", dpi=160)
    plt.close(fig)

    future = [row for row in rows if row["task"] == "forecast" and row["comparison_included"]]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, layout="constrained")
    if future:
        origin = min(row["origin_index"] for row in future)
        example = [row for row in future if row["origin_index"] == origin]
        truth = {row["horizon_seconds"]: row for row in example}
        times = sorted(truth)
        for axis, coordinate in zip(axes, ("x", "y", "z")):
            axis.plot(times, [truth[time][f"true_{coordinate}"] for time in times], "k.-", label="truth")
            for method in sorted({row["method"] for row in example}):
                points = sorted((row for row in example if row["method"] == method),
                                key=lambda row: row["horizon_seconds"])
                axis.plot([row["horizon_seconds"] for row in points],
                          [row[f"pred_{coordinate}"] for row in points], "o--", label=METHOD_LABELS[method])
            axis.set_ylabel(f"{coordinate} (m)")
        axes[0].legend(fontsize="small")
        fig.suptitle(f"Selected forecast targets — development diagnostics\n"
                     f"{context} | {settings['mode']} | origin index: {origin}")
    else:
        axes[0].text(0.5, 0.5, "No common successful forecasts", ha="center", transform=axes[0].transAxes)
    axes[-1].set_xlabel("Time after last observation (s)")
    fig.savefig(folder / "forecast_example.png", dpi=160)
    plt.close(fig)


def save_results(rows, settings, dt_seconds):
    """Create a unique run directory so earlier experiments are never overwritten."""
    root = Path(settings["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{settings['run']}-{settings['trajectory']}-{settings['domain']}-"
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    summary = summarize(rows)
    write_csv(folder / "predictions.csv", rows)
    write_csv(folder / "summary.csv", summary)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_type": "development_diagnostic",
        "note": "Checkpoint training split is unrecorded; these are not verified held-out results.",
        "gdf_orientation": "not implemented; position metrics only",
        "comparison_policy": "Forecast metrics use common successful cases across all selected forecast methods.",
        "dt_seconds": dt_seconds,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()},
    }
    (folder / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_figures(folder, rows, summary, dt_seconds, settings)
    return folder, summary
