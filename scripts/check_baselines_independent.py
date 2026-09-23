"""
Independent cross-check of the baseline numbers: recompute stay-put and
constant-velocity RMSE straight from the raw Kaggle CSVs with pandas, in the
ORIGINAL un-mirrored field coordinates, using none of the Stage 0 / Dataset /
collate / metrics code. If this matches eval_baselines.py, then Stage 0's
mirroring, last-frame selection and velocity features, plus the metric
implementation, are all confirmed end to end.
"""

import numpy as np
import pandas as pd

VAL_WEEKS = ["17", "18"]

inputs = pd.concat(
    [pd.read_csv(f"data/train/input_2023_w{w}.csv") for w in VAL_WEEKS], ignore_index=True
)
outputs = pd.concat(
    [pd.read_csv(f"data/train/output_2023_w{w}.csv") for w in VAL_WEEKS], ignore_index=True
)

keys = ["game_id", "play_id", "nfl_id"]
last = (
    inputs[inputs.player_to_predict]
    .sort_values("frame_id")
    .groupby(keys)
    .tail(1)[keys + ["x", "y", "s", "dir"]]
    .rename(columns={"x": "x0", "y": "y0"})
)
rows = outputs.merge(last, on=keys, how="inner")
assert len(rows) == len(outputs), "some output rows have no matching last input frame"

t = rows["frame_id"].to_numpy() * 0.1  # seconds ahead
dir_rad = np.radians(rows["dir"].to_numpy())
vx = rows["s"].to_numpy() * np.sin(dir_rad)  # dir: clockwise from +y, in the RAW field frame
vy = rows["s"].to_numpy() * np.cos(dir_rad)


def rmse(px, py):
    sse = ((rows["x"] - px) ** 2 + (rows["y"] - py) ** 2).sum()
    return float(np.sqrt(sse / (2 * len(rows))))


print(f"raw-CSV recomputation, weeks {VAL_WEEKS}: N = {len(rows):,} rows")
print(f"  stay_put          {rmse(rows['x0'], rows['y0']):.8f}")
print(f"  constant_velocity {rmse(rows['x0'] + vx * t, rows['y0'] + vy * t):.8f}")
