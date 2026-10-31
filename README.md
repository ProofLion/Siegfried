# Siegfried: SHIRT trajectory prediction

CSCI 4622 Project skeleton. Uses the existing `shirtv1/` dataset folder.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```powershell
uv sync --locked
uv run python main.py
```

Edit the settings at the top of `main.py` to choose the trajectory, image domain,
model, history length, horizon, and direct/recursive prediction. The baseline
repeats the last observed pose. For now, observations come from true labels;
synthetic and lightbox share these labels. An image estimator is a future TODO.

- `data_loader.py`: loads ordered pose labels and image paths.
- `baseline.py`: working example of the model interface.
- `grey_dynamic_filter.py`, `unscented_kalman_filter.py`, `pose_estimator.py`:
  teammate TODOs.
- `image_comparison.py`: working MSE and SSIM image comparison.

Filters use `predict(history, steps_ahead, dt_seconds)`. History has shape
`(N, 7)`; return one pose `[x, y, z, qw, qx, qy, qz]` (meters, unit quaternion).
SHIRT's timestep is 5 seconds. Direct mode predicts only the final pose;
recursive mode feeds one-step predictions back into the history.

Compare images or run the small tests:

```powershell
uv run python image_comparison.py shirtv1/roe1/synthetic/images/img000001.jpg shirtv1/roe1/synthetic/images/img000002.jpg
uv run pytest
```

Images are compared in grayscale on [0, 1], without resizing. Sizes must match
and be at least 7 by 7 pixels. Lower MSE is better (identical = 0); higher SSIM
is better (identical = 1). These measure appearance, not pose accuracy.
Keep `uv.lock` in Git so teammates use the same dependency versions.
