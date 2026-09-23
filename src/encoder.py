"""
Trajectory encoder: fuses self_seq + pooled context + broadcast static features per
frame, then runs a unidirectional multi-layer LSTM over the input window.

Per-frame input to the LSTM = concat(
    self_seq[t]            (F_self,)      target player's own state
    pool_context(...)[t]   (2*F_ctx,)     mean||max over real context players
    static_feats            (F_static,)   ball_land_x/y, num_frames_output, role
                                           one-hot -- same vector every frame
) -> Linear -> ReLU -> LSTM input at step t.

pack_padded_sequence(..., input_lengths) is used so the LSTM never processes a
padded frame for any example -- this is what makes the final (h, c) reflect each
example's true last real frame regardless of how much padding the rest of the
batch needed.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from context_pool import pool_context


class TrajectoryEncoder(nn.Module):
    def __init__(self, f_self: int, f_ctx: int, f_static: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        input_dim = f_self + 2 * f_ctx + f_static
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, self_seq, context_seq, context_mask, static_feats, input_lengths):
        """
        self_seq:      (B, T_in, F_self)
        context_seq:   (B, T_in, 16, F_ctx)
        context_mask:  (B, 16) bool
        static_feats:  (B, F_static)
        input_lengths: (B,) long -- real (post-truncation) length per example

        Returns:
          outputs: (B, T_in, hidden_size) per-frame hidden states, 0 past each
                   example's own real length (pad_packed_sequence default)
          (h, c):  each (num_layers, B, hidden_size) -- final state, taken from
                   each example's OWN last real frame, not from padding
        """
        B, T_in, _ = self_seq.shape

        pooled = pool_context(context_seq, context_mask)              # (B, T_in, 2*F_ctx)
        static_b = static_feats.unsqueeze(1).expand(-1, T_in, -1)     # (B, T_in, F_static)
        fused = torch.cat([self_seq, pooled, static_b], dim=-1)
        x = self.act(self.input_proj(fused))                          # (B, T_in, hidden_size)

        # pack: LSTM steps only over each example's real frames.
        # enforce_sorted=False -- our batches are length-bucketed but not
        # necessarily sorted descending within the batch, and this avoids
        # requiring that.
        lengths_cpu = input_lengths.detach().to("cpu")
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T_in)

        return outputs, (h, c)
