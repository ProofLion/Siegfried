# Siegfried: SHIRT trajectory prediction

CSCI 4622 runner for testing the team's CNN and Grey Dynamic Filter (GDF).
Uses the existing `shirtv1/` dataset and `pose_model.pt` checkpoint.

## Setup and running

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```powershell
uv sync --locked
uv run python main.py --run gdf
```

Choose a mode with `--run`:

| Mode | What it evaluates |
| --- | --- |
| `cnn` | Current-image pose estimates: position (m) and rotation (degrees) |
| `gdf` | Future positions from true past positions |
| `combined` | Future positions from CNN-estimated past positions |
| `compare` | All three, with separate pose-estimation and forecast tables |

For a small customized run, or to see all options:

```powershell
uv run python main.py --run compare --trajectory roe2 --domain lightbox --samples 5 --mode direct
uv run python main.py --help
```

Defaults: ROE1 synthetic, 10 observed frames, horizons 1/5/10, recursive mode,
20 windows, and stride 10. Horizons start after the last observation; SHIRT
frames are 5 seconds apart. Direct mode jumps to a target; recursive mode feeds
one-step predictions back into the history. Stride spaces windows while each
history keeps consecutive frames. `--start-index` is the first history's start.

## Results

Each run creates a folder under `outputs/` containing:

- `summary.csv`: mean/median position error, mean rotation error, and sample counts.
- `predictions.csv`: individual predictions, true values, errors, and failures.
- `run.json`: experiment settings.
- PNG figures: CNN errors, forecast errors by horizon, and an example x/y/z forecast.

Forecasts are compared with holding the last observed position fixed, using
common successful windows. Failed cases remain in the CSV; `N/A` means unavailable.
The checkpoint's training split is unrecorded, so CNN results are development
diagnostics. GDF orientation is unfinished, and numerical failures are recorded.

## Files and tests

- `main.py`: commands and orchestration.
- `evaluation.py`: metrics, forecast windows, and CNN estimate caching.
- `model_adapters.py`: wrappers around the existing CNN and scalar x/y/z GDFs.
- `results.py`: tables and exports.
- `pose_estimator.py`, `grey_dynamic_filter.py`, `models/`: teammates' model code.

Poses use `[x, y, z, qw, qx, qy, qz]` (meters, scalar-first quaternion).
Run `uv run pytest` for integration tests using small fixtures, without training.
`image_comparison.py` separately compares image appearance with MSE/SSIM.
Keep `uv.lock` in Git; leave virtual environments and generated outputs out.
