"""
Context pooling: collapses the 16 (padded, masked) context-player slots per frame
into one fixed-size summary vector per frame, via mean and max pooling over the
REAL slots only.

Input:  context_seq  (B, T, 16, F_ctx)
        context_mask (B, 16) bool -- which of the 16 slots are real, constant
                     across the whole sequence (players don't appear/disappear
                     mid-play in this dataset; see Stage 0 notes)
Output: (B, T, 2*F_ctx) -- [mean_pool || max_pool] per frame

Two edge cases, both realized in practice by the 0-player play
(game_id=2023112606, play_id=4180), where all 16 slots are masked out:

  mean: sum over 0 real players, divided by 0 -> NaN unless guarded.
        Fix: divide by max(count, 1) instead of count directly. With 0 real
        players the sum is already exactly 0 (padded slots are zeros), so
        0 / max(0,1) = 0 -- a clean zero, not a fabricated non-zero value.

  max:  max over 0 real players is -inf (an empty max is -inf by convention,
        which is why we can't just set padded slots to -inf and take a plain
        max -- with 0 real players every slot is -inf and the max is -inf).
        Fix: replace masked slots with -inf before the max (so real slots
        always win over padding), then replace any resulting -inf (only
        possible when there were 0 real players) with 0.
"""

import torch

NEG_INF = float("-inf")


def pool_context(context_seq: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    """
    context_seq:  (B, T, P, F) float
    context_mask: (B, P) bool
    returns:      (B, T, 2*F) float -- [mean_pool, max_pool] concatenated on the last dim
    """
    B, T, P, F = context_seq.shape
    mask = context_mask[:, None, :, None].expand(B, T, P, F)  # (B, T, P, F)

    # --- mean: sum over real slots / max(count, 1) ---
    zeroed = torch.where(mask, context_seq, torch.zeros_like(context_seq))
    summed = zeroed.sum(dim=2)                                   # (B, T, F)
    count = context_mask.sum(dim=1).to(context_seq.dtype)        # (B,) real-player count per example
    count = count.clamp(min=1.0)                                 # guard: never divide by 0
    mean_pool = summed / count[:, None, None]                    # (B, T, F)

    # --- max: -inf on padding so real slots always win, then -inf -> 0 (only when count was 0) ---
    neg_inf_filled = torch.where(mask, context_seq, torch.full_like(context_seq, NEG_INF))
    max_pool = neg_inf_filled.max(dim=2).values                  # (B, T, F)
    max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)

    return torch.cat([mean_pool, max_pool], dim=-1)              # (B, T, 2F)
