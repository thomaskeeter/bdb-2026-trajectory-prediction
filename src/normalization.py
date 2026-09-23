"""
Z-score normalization + the inverse mapping back to real field coordinates.

Stats are computed over the TRAIN split only (weeks 1-16) and then reused
unchanged for the val split (and anything later), so nothing about the val
weeks leaks into the model's inputs.

What gets normalized (everything else is left alone):
  self_seq      cols 0-6   x, y, dx_ball, dy_ball, dx_los, s, a
                (cols 7-10 are sin/cos of angles, already in [-1, 1])
  context_seq   cols 0-3   dx_to_target, dy_to_target, s, a -- REAL players only.
                Padded slots must stay exactly 0, so they are never touched.
                (cols 4-7 sin/cos, cols 8-11 role one-hot: unchanged)
  static_feats  cols 0-2   ball_land_x, ball_land_y, num_frames_output
                (cols 3-6 role one-hot: unchanged)
  target_seq    both cols  normalized with the self_seq x/y stats, since targets
                live in the same (mirrored) field coordinate frame as x/y inputs.

Mapping a model prediction back to real field coordinates is two steps, in
this order:  1) un-normalize (x*std + mean)   2) un-mirror (x = 120 - x on
'left' plays).  to_field_coords() does both.
"""

import torch

SELF_NORM_COLS = [0, 1, 2, 3, 4, 5, 6]
CTX_NORM_COLS = [0, 1, 2, 3]
STATIC_NORM_COLS = [0, 1, 2]

SELF_COL_NAMES = ["x", "y", "dx_ball", "dy_ball", "dx_los", "s", "a"]
CTX_COL_NAMES = ["dx_to_target", "dy_to_target", "s", "a"]
STATIC_COL_NAMES = ["ball_land_x", "ball_land_y", "num_frames_output"]

FIELD_LENGTH = 120.0
MIN_STD = 1e-6


def _finish(total, total_sq, n):
    mean = total / n
    var = (total_sq / n - mean ** 2).clamp(min=0.0)
    std = var.sqrt().clamp(min=MIN_STD)
    return mean, std


def compute_norm_stats(examples) -> dict:
    """examples: iterable of RAW (un-normalized) train examples."""
    s_sum = torch.zeros(len(SELF_NORM_COLS), dtype=torch.float64)
    s_sq = torch.zeros_like(s_sum)
    s_n = 0

    c_sum = torch.zeros(len(CTX_NORM_COLS), dtype=torch.float64)
    c_sq = torch.zeros_like(c_sum)
    c_n = 0

    t_sum = torch.zeros(len(STATIC_NORM_COLS), dtype=torch.float64)
    t_sq = torch.zeros_like(t_sum)
    t_n = 0

    for ex in examples:
        s = ex["self_seq"][:, SELF_NORM_COLS].double()
        s_sum += s.sum(0)
        s_sq += (s ** 2).sum(0)
        s_n += s.shape[0]

        mask = ex["context_mask"]
        if mask.any():
            # real players only: padded slots are zeros and would drag the stats toward 0
            c = ex["context_seq"][:, mask][..., CTX_NORM_COLS].double().reshape(-1, len(CTX_NORM_COLS))
            c_sum += c.sum(0)
            c_sq += (c ** 2).sum(0)
            c_n += c.shape[0]

        t = ex["static_feats"][STATIC_NORM_COLS].double()
        t_sum += t
        t_sq += t ** 2
        t_n += 1

    self_mean, self_std = _finish(s_sum, s_sq, s_n)
    ctx_mean, ctx_std = _finish(c_sum, c_sq, c_n)
    static_mean, static_std = _finish(t_sum, t_sq, t_n)

    return {
        "self_mean": self_mean, "self_std": self_std, "self_n": s_n,
        "ctx_mean": ctx_mean, "ctx_std": ctx_std, "ctx_n": c_n,
        "static_mean": static_mean, "static_std": static_std, "static_n": t_n,
    }


def normalize_example(ex: dict, stats: dict) -> dict:
    """Returns a new dict; the stored example is not modified."""
    out = dict(ex)

    self_mean = stats["self_mean"].float()
    self_std = stats["self_std"].float()
    ctx_mean = stats["ctx_mean"].float()
    ctx_std = stats["ctx_std"].float()
    static_mean = stats["static_mean"].float()
    static_std = stats["static_std"].float()

    self_seq = ex["self_seq"].clone()
    self_seq[:, SELF_NORM_COLS] = (self_seq[:, SELF_NORM_COLS] - self_mean) / self_std
    out["self_seq"] = self_seq

    context_seq = ex["context_seq"].clone()
    mask = ex["context_mask"]
    if mask.any():
        real = context_seq[:, mask]  # (T, k, F_ctx) -- boolean indexing returns a copy
        real[..., CTX_NORM_COLS] = (real[..., CTX_NORM_COLS] - ctx_mean) / ctx_std
        context_seq[:, mask] = real
    out["context_seq"] = context_seq

    static_feats = ex["static_feats"].clone()
    static_feats[STATIC_NORM_COLS] = (static_feats[STATIC_NORM_COLS] - static_mean) / static_std
    out["static_feats"] = static_feats

    # target x, y share the self_seq x, y stats (cols 0, 1)
    out["target_seq"] = (ex["target_seq"] - self_mean[:2]) / self_std[:2]

    return out


def denormalize_xy(xy_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    """(..., 2) normalized [x, y] -> mirrored-frame yards. float64 to keep roundoff tiny."""
    mean = stats["self_mean"][:2].double()
    std = stats["self_std"][:2].double()
    return xy_norm.double() * std + mean


def unmirror_xy(xy: torch.Tensor, is_left) -> torch.Tensor:
    """Mirrored-frame yards -> real field yards. x = 120 - x on 'left' plays; y unchanged.

    xy: (..., 2).  is_left: bool or bool tensor whose shape is a prefix of xy's
    batch shape, e.g. (B,) for xy of shape (B, T, 2).
    """
    is_left = torch.as_tensor(is_left, dtype=torch.bool)
    flip = is_left.reshape(is_left.shape + (1,) * (xy.dim() - 1 - is_left.dim()))
    x = torch.where(flip, FIELD_LENGTH - xy[..., 0], xy[..., 0])
    return torch.stack([x, xy[..., 1]], dim=-1)


def to_field_coords(xy_norm: torch.Tensor, is_left, stats: dict) -> torch.Tensor:
    """Model-space (normalized, mirrored) prediction -> real field coordinates."""
    return unmirror_xy(denormalize_xy(xy_norm, stats), is_left)
