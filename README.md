# BDB 2026 Trajectory Prediction

A learning project: predicting where NFL players end up between the moment a pass is
thrown and the moment it arrives, using the [NFL Big Data Bowl 2026 - Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)
Kaggle competition dataset.

## About this project

I'm a business/analytics major, not a CS student — this was a learning project, not
professional-grade software. I drove every design decision here (what to predict, how
to structure the data, how the model should work, what to measure and why) and
understood the reasoning at each step, but I leaned heavily on Claude for
implementation: writing the actual code, catching bugs, and running the verification
checks described below. I'm not claiming to have written this solo or that it reflects
production engineering practice. What I do think is real: every non-trivial claim in
this repo — a transform is correct, a metric matches the competition's formula, a
component handles its edge cases — was checked against actual data or a hand-computed
example before being relied on, not just asserted. That verification habit, and
understanding *why* each piece works the way it does, is the actual skill this project
was for.

## What the model does

Given NFL Next Gen Stats tracking data up to the moment a pass is thrown — every
player's position, speed, and orientation, plus where the ball is targeted to land —
predict where the targeted receiver and relevant defenders will be, frame by frame,
until the ball arrives. It's a variable-length sequence-to-sequence problem: both the
input window (how long the play had been developing before the throw) and the output
window (how long the ball is in the air) differ from play to play.

## Pipeline

**1. Data prep** (`scripts/stage0_preprocess.py`) — Turns the raw per-frame tracking
CSVs into fixed-shape training examples. Standardizes play direction so the offense
always drives the same way (mirroring x-coordinates and motion angles), converts
angles to sin/cos to avoid the 359°/0° discontinuity, computes features relative to
the ball's landing spot and the line of scrimmage, and truncates/pads sequences to
consistent lengths. Verified against hand-picked edge cases, including a play with
zero other tracked players on the field.

**2. Normalization** (`src/normalization.py`) — Z-score stats computed on the training
split only (no leakage from validation weeks), with a documented, tested round trip
back to real field coordinates for whenever a prediction needs interpreting.

**3. Model** (`src/encoder.py`, `src/context_pool.py`, `src/decoder.py`) — An
encoder–decoder LSTM. The encoder reads each play's pre-throw window (the target
player's own motion, plus a pooled summary of every other player on the field) and
produces a final hidden state. The decoder then runs autoregressively, one frame at a
time, predicting a small position displacement per step and summing those up from the
player's last known position. Every piece — the pooling's handling of the zero-players
edge case, the encoder's handling of variable-length padded input, the decoder's
displacement math — was checked against a manually-computed expected result before
being trusted.

**4. Training loop** (`scripts/train.py`) — Standard supervised training (Adam,
gradient clipping, GPU-accelerated), with one deliberate design choice worth calling
out: validation is measured two ways each epoch. One measurement lets the model see
the true previous position at each step ("teacher-forced"); the other makes it feed
its own predictions forward, exactly like it will have to at real inference time.
Checkpointing and early stopping are driven by the second, harder number, because
that's the one that actually matters.

**5. Baselines** (`src/baselines.py`) — Before trusting the model's numbers, I checked
them against two trivial predictors: "stay where you were" and "keep going at your
last known speed and direction." A model that can't beat a straight-line extrapolation
isn't doing anything useful.

## Results

The competition's own metric — RMSE in yards between predicted and true (x, y)
position — was verified to match the formula on Kaggle's evaluation page, and
cross-checked against an independent from-scratch recomputation off the raw CSVs.

| predictor | validation RMSE (yards) |
|---|---|
| stay put | 4.32 |
| constant velocity | 1.65 |
| **this model** | **0.82** |

The trained model roughly **halves the error** of the constant-velocity baseline on
the held-out validation weeks (weeks 17–18 of the 2023 season, never seen during
training). Training stopped early at epoch 10 of a possible 40, once validation
performance stopped improving for 5 consecutive epochs.

## What I'd improve next

**The exposure-bias gap.** During training the model always sees the *true* previous
position when predicting the next step. At real inference it only has its own
(imperfect) predictions to build on. My two validation numbers make this gap visible
rather than hiding it: the "sees the truth" number kept improving for a few more
epochs than the "must use its own predictions" number did, and the latter got
noticeably noisier as training went on. The standard fix — *scheduled sampling*,
gradually forcing the model to practice on its own predictions during training instead
of always the true ones — is a known next step, not a problem I discovered and left
unaddressed.

Other things I'd want to try with more time: a smarter way to combine information from
other players on the field (right now it's a simple pooled average/max rather than
anything that learns which players matter most), and checking whether the model's
errors are concentrated in particular play types or player roles rather than spread
evenly.

## Project layout

```
scripts/    one script per pipeline step (data prep, baselines, training, verification)
src/        the actual model/data code the scripts import
data/       not tracked in git — see Data below
checkpoints/, results/   training logs and outputs (checkpoints not tracked; logs are)
```

## Data

The dataset is not included in this repo — `data/` is git-ignored. To reproduce:
download the [competition data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)
from Kaggle, place it under `data/train/`, then run `scripts/stage0_preprocess.py`
followed by `scripts/compute_norm_stats.py`.

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
