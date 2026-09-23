# BDB 2026 Trajectory Prediction

Predicts where NFL players will be, frame by frame, between the moment a pass is
thrown and the moment it arrives — using pre-throw player tracking data from the
[NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
competition. Built with Claude Code under my direction.

Given tracking data up to the throw — every player's position, speed, and
orientation, plus where the ball is targeted to land — the model predicts the
targeted receiver's and relevant defenders' positions for every frame until the ball
arrives. Both the input window (time before the throw) and output window (time the
ball is in the air) vary play to play, making this a variable-length
sequence-to-sequence problem.

## Pipeline

**Data prep** (`scripts/stage0_preprocess.py`)
- Standardizes play direction so the offense always moves the same way (mirrors
  x-coordinates and motion angles for plays running the other direction)
- Converts motion/orientation angles to sin/cos to avoid the 0°/360° wraparound
- Computes features relative to the ball's landing spot and the line of scrimmage
- Truncates/pads variable-length sequences to a consistent shape per batch

**Normalization** (`src/normalization.py`)
- Z-score stats computed on the training split only
- Tested round trip back to real field coordinates for interpreting predictions

**Model** (`src/encoder.py`, `src/context_pool.py`, `src/decoder.py`)
- Encoder–decoder LSTM
- Encoder reads each play's pre-throw window: the target player's own motion, plus a
  pooled (mean/max) summary of every other player on the field
- Decoder runs autoregressively, predicting a position displacement per frame and
  summing from the player's last known position

**Training** (`scripts/train.py`)
- Adam, gradient clipping, GPU-accelerated
- Validation computed two ways each epoch: teacher-forced (sees the true previous
  position) and free-running (feeds its own predictions forward, matching real
  inference conditions). Checkpointing and early stopping are driven by the
  free-running number.

**Baselines** (`src/baselines.py`)
- Stay-put and constant-velocity reference predictors

## Results

RMSE in yards between predicted and true (x, y) position, matching the
competition's own scoring formula:

| predictor | validation RMSE (yards) |
|---|---|
| stay put | 4.32 |
| constant velocity | 1.65 |
| **this model** | **0.82** |

Roughly half the error of the constant-velocity baseline, on validation weeks
(17–18 of the 2023 season) held out from training. Training stopped early at epoch
10 of a possible 40, once validation performance stopped improving.

## Next steps

- **Scheduled sampling** — the model is trained teacher-forced but must run
  free-running at inference; free-running validation RMSE is noisier and improves
  for fewer epochs than the teacher-forced number, a classic exposure-bias signature.
- **Learned player-context aggregation** — replace mean/max pooling with something
  that learns which other players matter most (e.g. attention).
- **Error breakdown by play type / player role** to find where the model struggles.

## Project layout

```
scripts/    one script per pipeline step (data prep, baselines, training, verification)
src/        model and data code the scripts import
data/       not tracked in git — see Data below
checkpoints/, results/   training outputs (checkpoints not tracked; logs are)
```

## Data

`data/` is git-ignored. To reproduce: download the
[competition data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
from Kaggle into `data/train/`, then run `scripts/stage0_preprocess.py` followed by
`scripts/compute_norm_stats.py`.

Tracking data provided by the NFL Next Gen Stats team via the NFL Big Data Bowl 2026,
licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Citation:
Michael Lopez, Tom Bliss, Ally Blake, Yao Yan, Martyna Plomecka, and Addison Howard,
*NFL Big Data Bowl 2026 - Prediction*, Kaggle, 2025.

## Setup

```
python -m venv venv
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

See `requirements.txt` for a note on picking the right torch build for your own
hardware — the pin above is specific to the machine this was developed on.
