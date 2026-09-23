"""
The competition metric (NFL Big Data Bowl 2026 - Prediction, Evaluation page):

    RMSE = sqrt( 1/(2N) * sum_i [ (x_true,i - x_pred,i)^2 + (y_true,i - y_pred,i)^2 ] )

N is the number of scored rows -- one row per (game_id, play_id, nfl_id, frame_id),
i.e. one per predicted player per output frame. So:
  * it is a flat ("micro") average over every real frame of every example, NOT an
    average of per-example or per-batch RMSEs -- longer plays contribute more rows;
  * the 2N means x and y squared errors are averaged together, so this is the
    Euclidean-distance RMSE divided by sqrt(2);
  * units are yards.
Padded frames (mask == False) never count toward the sum or toward N.

RMSEAccumulator keeps running sums (sum of squared error, row count) and only takes
the square root at the very end, which is what makes multi-batch evaluation equal to a
single pass over the whole split.
"""

import math

import torch


def masked_sq_error(pred_xy: torch.Tensor, true_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, T, 2), (B, T, 2), (B, T) bool -> (B, T) of (dx^2 + dy^2), exactly 0 on padded frames."""
    sq = ((pred_xy.double() - true_xy.double()) ** 2).sum(dim=-1)
    return sq * mask.to(sq.dtype)


def _rmse(sse: float, n: int) -> float:
    return math.sqrt(sse / (2 * n)) if n > 0 else float("nan")


class RMSEAccumulator:
    def __init__(self):
        self.sse = 0.0
        self.n = 0
        self.by_step = {}  # 1-based output frame k -> [sse, n]
        self.by_tag = {}   # arbitrary per-example label (e.g. role) -> [sse, n]

    def update(self, pred_xy, true_xy, mask, tags=None):
        sq = masked_sq_error(pred_xy, true_xy, mask)  # (B, T)
        self.sse += sq.sum().item()
        self.n += int(mask.sum().item())

        sse_k, n_k = sq.sum(dim=0), mask.sum(dim=0)
        for k in range(sq.shape[1]):
            if n_k[k] > 0:
                slot = self.by_step.setdefault(k + 1, [0.0, 0])
                slot[0] += sse_k[k].item()
                slot[1] += int(n_k[k].item())

        if tags is not None:
            sse_i, n_i = sq.sum(dim=1), mask.sum(dim=1)
            for i, tag in enumerate(tags):
                slot = self.by_tag.setdefault(tag, [0.0, 0])
                slot[0] += sse_i[i].item()
                slot[1] += int(n_i[i].item())

    def compute(self) -> float:
        return _rmse(self.sse, self.n)

    def compute_by_step(self) -> dict:
        return {k: _rmse(s, n) for k, (s, n) in sorted(self.by_step.items())}

    def compute_by_tag(self) -> dict:
        return {t: (_rmse(s, n), n) for t, (s, n) in self.by_tag.items()}
