"""Image similarity using MSE and SSIM (not 3D pose accuracy)."""

import argparse

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load_grayscale(image_path):
    """Load grayscale pixels as floats in [0, 1]."""
    with Image.open(image_path) as image:
        return np.array(image.convert("L"), dtype=float) / 255.0


def compare_images(reference_path, candidate_path):
    """Return MSE (lower is better) and SSIM (higher is better)."""
    reference = load_grayscale(reference_path)
    candidate = load_grayscale(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError("Images must have matching dimensions.")
    if min(reference.shape) < 7:
        raise ValueError("Images must be at least 7 by 7 pixels for SSIM.")

    return {
        "mse": float(np.mean((reference - candidate) ** 2)),
        "ssim": float(structural_similarity(reference, candidate, data_range=1.0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    try:
        print(compare_images(args.reference, args.candidate))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Image comparison error: {error}\n")
