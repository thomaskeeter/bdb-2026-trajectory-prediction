"""
Thin orchestration wrapper: encoder -> decoder, given a collated batch dict.
Exists so the training loop (and later, inference) doesn't have to repeat the
"pull out last_position, wire encoder state into decoder" plumbing every time.
"""

import torch
import torch.nn as nn

from decoder import TrajectoryDecoder
from encoder import TrajectoryEncoder


class TrajectoryModel(nn.Module):
    def __init__(self, f_self: int, f_ctx: int, f_static: int,
                 hidden_size: int = 128, num_layers: int = 2, static_embed_dim: int = 32):
        super().__init__()
        self.encoder = TrajectoryEncoder(f_self, f_ctx, f_static, hidden_size, num_layers)
        self.decoder = TrajectoryDecoder(f_ctx, f_static, hidden_size, num_layers, static_embed_dim)

    def forward(self, batch: dict, teacher_forcing: bool = True):
        _, (h, c) = self.encoder(
            batch["self_seq"], batch["context_seq"], batch["context_mask"],
            batch["static_feats"], batch["input_lengths"],
        )

        B = batch["self_seq"].shape[0]
        last_idx = batch["input_lengths"] - 1
        last_position = batch["self_seq"][torch.arange(B, device=batch["self_seq"].device), last_idx][:, :2]

        T_out_max = batch["target_seq"].shape[1]
        pred_disp, pred_abs = self.decoder(
            (h, c), last_position,
            batch["context_seq"], batch["context_mask"], batch["input_lengths"],
            batch["static_feats"], batch["output_lengths"],
            target_seq=batch["target_seq"], T_out_max=T_out_max, teacher_forcing=teacher_forcing,
        )
        return pred_disp, pred_abs
