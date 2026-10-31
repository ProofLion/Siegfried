"""Read SHIRT labels and image paths; image pixels are loaded only when needed."""

import json
from pathlib import Path

import numpy as np


def load_trajectory(data_dir, trajectory="roe1", domain="synthetic"):
    """Return ordered frame dictionaries and seconds between frames."""
    if trajectory not in ("roe1", "roe2") or domain not in ("synthetic", "lightbox"):
        raise ValueError("Choose roe1/roe2 and synthetic/lightbox.")
    folder = Path(data_dir) / trajectory
    with (folder / f"{trajectory}.json").open() as file:
        labels = json.load(file)
    with (folder / "metadata.json").open() as file:
        metadata = json.load(file)

    frames = []
    # SHIRT filenames are zero-padded, so sorting gives chronological order.
    for label in sorted(labels, key=lambda item: item["filename"]):
        frames.append({
            "image_path": folder / domain / "images" / label["filename"],
            # Position in meters, followed by quaternion [qw, qx, qy, qz].
            "pose": np.array(label["r_Vo2To_vbs_true"] + label["q_vbs2tango_true"]),
        })
    return frames, metadata["pSim"]["cam_step"]
