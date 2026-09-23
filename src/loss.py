"""
Differentiable training loss: masked RMSE in yards, matching the competition
metric exactly (RMSE = sqrt(1/(2N) * sum((x_true-x_pred)^2 + (y_true-y_pred)^2))).

Deliberately reuses metrics.masked_sq_error -- the exact same function
RMSEAccumulator calls internally -- rather than reimplementing the formula, so
"matches metrics.py exactly" holds by construction, not just by a passing test.
Predictions/targets are un-normalized back to yards first (denormalize_xy),
since the competition metric is defined in yards and z-scored x has a
different std than z-scored y, so comparing in normalized units would silently
reweight the two axes.
"""

import torch

from metrics import masked_sq_error
from normalization import denormalize_xy


def masked_rmse_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor,
                      mask: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    pred_norm, target_norm: (B, T, 2), normalized (z-scored) x, y
    mask: (B, T) bool -- True on real (non-padded) output frames
    stats: norm_stats dict (from normalization.compute_norm_stats)
    Returns: scalar tensor, differentiable w.r.t. pred_norm.
    """
    pred_yards = denormalize_xy(pred_norm, stats)      # (B, T, 2), float64
    target_yards = denormalize_xy(target_norm, stats)  # (B, T, 2), float64
    sq = masked_sq_error(pred_yards, target_yards, mask)  # (B, T), 0 on padding
    n = mask.sum()
    return torch.sqrt(sq.sum() / (2 * n))
