"""
Reference baselines the LSTM has to beat. Both take a collated batch built from the
RAW (un-normalized) Dataset, so positions are in yards in the standardized (mirrored)
frame, and both return predictions in that same frame, shape (B, T_out, 2).

  stay_put            every output frame = the player's last observed position.
  constant_velocity   last observed position advanced at the last observed velocity,
                      pos_k = last + k * 0.1s * v,  k = 1..T_out.

Velocity comes from the tracking columns kept in self_seq: speed s (col 5, yards/sec)
and direction of motion as sin/cos (cols 7, 8). dir is measured clockwise from the +y
axis (checked against frame-to-frame displacement vectors in Stage 0), so the unit
direction vector is (sin(dir), cos(dir)) -- and because Stage 0 mirrored dir with
dir_std = 360 - dir, that same formula is already correct in the mirrored frame.
"""

import torch

STEP_SECONDS = 0.1  # NFL tracking is 10 frames per second

X_COL, Y_COL, S_COL, SIN_DIR_COL, COS_DIR_COL = 0, 1, 5, 7, 8


def _last_observed(batch: dict) -> torch.Tensor:
    """(B, F_self): each example's self_seq row at its true last input frame (not padding)."""
    self_seq = batch["self_seq"]
    last_idx = batch["input_lengths"] - 1
    return self_seq[torch.arange(self_seq.shape[0]), last_idx]


def stay_put(batch: dict) -> torch.Tensor:
    last = _last_observed(batch)
    T_out = batch["target_seq"].shape[1]
    pos = last[:, [X_COL, Y_COL]]
    return pos.unsqueeze(1).expand(-1, T_out, -1).clone()


def constant_velocity(batch: dict) -> torch.Tensor:
    last = _last_observed(batch)
    T_out = batch["target_seq"].shape[1]
    pos = last[:, [X_COL, Y_COL]]                                            # (B, 2)
    speed = last[:, S_COL]                                                   # (B,)
    unit = torch.stack([last[:, SIN_DIR_COL], last[:, COS_DIR_COL]], dim=1)  # (B, 2)
    vel = speed.unsqueeze(1) * unit                                          # (B, 2) yards/sec
    k = torch.arange(1, T_out + 1, dtype=pos.dtype) * STEP_SECONDS           # (T_out,) seconds ahead
    return pos.unsqueeze(1) + k.view(1, -1, 1) * vel.unsqueeze(1)
