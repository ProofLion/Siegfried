"""Basic image comparison checks using temporary files."""

import numpy as np
from PIL import Image
import pytest

from image_comparison import compare_images


def test_identical_images(tmp_path):
    path = tmp_path / "same.png"
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(path)
    scores = compare_images(path, path)
    assert scores["mse"] == pytest.approx(0)
    assert scores["ssim"] == pytest.approx(1)


def test_changed_images(tmp_path):
    dark, bright = tmp_path / "dark.png", tmp_path / "bright.png"
    Image.new("L", (16, 16), 0).save(dark)
    Image.new("L", (16, 16), 255).save(bright)
    scores = compare_images(dark, bright)
    assert scores["mse"] == pytest.approx(1)  # Normalized floats, not uint8.
    assert scores["ssim"] < 1


def test_mismatched_sizes(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("L", (16, 16)).save(first)
    Image.new("L", (20, 16)).save(second)
    with pytest.raises(ValueError, match="matching dimensions"):
        compare_images(first, second)
