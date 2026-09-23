# BDB 2026 Trajectory Prediction

Predicts where NFL players will be, frame by frame, between the moment a pass is
thrown and the moment it's caught (or hits the ground). Uses pre-throw player
tracking data from the
[NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
competition. Built with Claude Code under my direction.

The model gets tracking data up through the throw: every player's position, speed,
orientation, and where the ball is targeted to land. From that it predicts the
targeted receiver's and relevant defenders' positions for each remaining frame until
the ball arrives. The tricky part is that both windows are variable length. Some
plays have long pre-snap motion, some throws hang in the air for three seconds and
some for barely half a second. So this ends up being a variable-length
sequence-to-sequence problem, not a fixed-size regression.

## Pipeline

### Data prep (`scripts/stage0_preprocess.py`)
Standardizes play direction so the offense always moves the same way on screen
(mirrors x-coordinates and motion angles for plays running the other direction).
Motion and orientation angles get converted to sin/cos so the model doesn't see a
fake jump between 359° and 0°. Also computes features relative to the ball's landing
spot and the line of scrimmage, and truncates/pads the variable-length sequences to
a consistent shape per batch.

### Normalization (`src/normalization.py`)
Z-score stats, computed on the training split only. Includes a tested round trip
back to real field coordinates so predictions can actually be interpreted, not just
compared in normalized space.

### Model (`src/encoder.py`, `src/context_pool.py`, `src/decoder.py`)
An encoder-decoder LSTM. The encoder reads each play's pre-throw window: the target
player's own motion plus a pooled (mean/max) summary of everyone else on the field.
The decoder then runs autoregressively, predicting a position displacement for each
output frame and summing those onto the player's last known position.

### Training (`scripts/train.py`)
Adam optimizer, gradient clipping, GPU-accelerated. Validation runs two ways each
epoch: teacher-forced (the model sees the true previous position) and free-running
(it feeds its own predictions forward, which is what actually happens at inference).
Checkpointing and early stopping are based on the free-running number, since that's
the one that reflects real performance.

### Baselines (`src/baselines.py`)
Stay-put and constant-velocity predictors, just to have something to beat.

## Results

RMSE in yards between predicted and true (x, y) position, using the same formula
the competition scores submissions with:

| predictor | validation RMSE (yards) |
|---|---|
| stay put | 4.32 |
| constant velocity | 1.65 |
| **this model** | **0.82** |

About half the error of the constant-velocity baseline, evaluated on validation
weeks (17-18 of the 2023 season) that the model never trained on. Training stopped
early at epoch 10 of a possible 40 once validation performance plateaued.

## Next steps

- **Scheduled sampling.** The model trains teacher-forced but has to run
  free-running at inference. Free-running validation RMSE is noisier and stops
  improving sooner than the teacher-forced number, which is a pretty textbook sign
  of exposure bias.
- **A smarter way to combine player context.** Mean/max pooling is simple and it
  works, but something like attention could let the model actually learn which
  other players on the field matter most for a given prediction.
- **Break down errors by play type and player role** to see where the model is
  actually struggling instead of just looking at the aggregate number.

## Project layout

```
scripts/    one script per pipeline step (data prep, baselines, training, verification)
src/        model and data code the scripts import
data/       not tracked in git, see Data below
checkpoints/, results/   training outputs (checkpoints not tracked, logs are)
```

## Data

`data/` is git-ignored. To reproduce, download the
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

Check `requirements.txt` for a note on picking the right torch build for your own
hardware. The pin above is specific to the machine this was developed on.
