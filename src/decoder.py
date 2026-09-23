"""
Autoregressive trajectory decoder. Initialized from the encoder's final (h, c),
runs step-by-step, predicting a (dx, dy) displacement per output frame. Absolute
positions are the running cumulative sum of those displacements, anchored on the
LAST OBSERVED input position (not frame 0 of some fixed axis).

Per decode step t (0-indexed; output frame k = t+1), the LSTM's input is:
  prev_disp   (2,)   the displacement that produced the PREVIOUS frame -- during
                      training this is teacher-forced from the true target (not
                      the model's own prior prediction); zero at t=0, since frame
                      0 IS the last observed input frame (nothing "produced" it
                      within this decode).
  progress    (2,)   [k / output_len, 1 - k / output_len] -- how far through this
                      example's own flight this step is. Uses each example's true
                      output_len, not the batch's padded T_out_max.
  conditioning (2*F_ctx + static_embed_dim,)  CONSTANT every step: the pooled
                      context summary at the encoder's last real input frame
                      (context stops at the throw -- this is all the decoder
                      knows about other players) concatenated with a learned
                      embedding of static_feats (ball landing spot,
                      num_frames_output, role).

Batching note: like the rest of this pipeline, a batch runs to T_out_max (the
batch's own max output_len), not each example's individual output_len -- the
LSTM step-loop can't easily skip individual (batch-index, step) cells the way
pack_padded_sequence skips whole trailing frames on the encoder side. Steps
past an example's true output_len are just discarded later via output_mask
(same masked-loss pattern already proven for the encoder/collate pipeline).
This doesn't corrupt any real step: causally, step t only ever depends on
steps < t, computed identically regardless of what happens after.
"""

import torch
import torch.nn as nn

from context_pool import pool_context


class TrajectoryDecoder(nn.Module):
    def __init__(self, f_ctx: int, f_static: int,
                 hidden_size: int = 128, num_layers: int = 2, static_embed_dim: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.static_embed = nn.Sequential(nn.Linear(f_static, static_embed_dim), nn.ReLU())
        cond_dim = 2 * f_ctx + static_embed_dim
        step_input_dim = 2 + 2 + cond_dim  # prev_disp(2) + progress(2) + conditioning

        self.lstm = nn.LSTM(
            input_size=step_input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, 2)  # -> (dx, dy)

    def forward(self, encoder_state, last_position, context_seq, context_mask, input_lengths,
                static_feats, output_lengths, target_seq=None, T_out_max=None, teacher_forcing=True):
        """
        encoder_state: (h, c), each (num_layers, B, hidden_size) from TrajectoryEncoder
        last_position: (B, 2) -- each example's x, y at its true last INPUT frame
                        (same normalized units as target_seq, since both use the
                        self_seq x/y normalization stats)
        context_seq, context_mask, input_lengths: same as the encoder's, used here
                        to recompute the pooled context summary and pick out each
                        example's true last real input frame
        static_feats:   (B, F_static)
        output_lengths: (B,) long -- each example's true output_len
        target_seq:     (B, T_out_max, 2) true absolute positions, required when
                        teacher_forcing=True (training). Padded frames (beyond an
                        example's real output_len) are 0 and never used except as
                        harmless input to already-masked-out future steps.
        T_out_max:      steps to run; inferred from target_seq if not given.
        teacher_forcing: if True, each step's input displacement is the TRUE prior
                        displacement (from target_seq); if False, it's the model's
                        own previous prediction (inference mode).

        Returns:
          pred_disp: (B, T_out_max, 2) predicted per-step displacement
          pred_abs:  (B, T_out_max, 2) predicted absolute positions = last_position
                     + cumulative sum of pred_disp (NOT of the teacher-forced true
                     displacements -- this is what the model itself would produce)
        """
        h, c = encoder_state
        B = last_position.shape[0]
        device = last_position.device

        if T_out_max is None:
            assert target_seq is not None, "T_out_max must be given when target_seq is None"
            T_out_max = target_seq.shape[1]
        if teacher_forcing:
            assert target_seq is not None, "teacher_forcing=True requires target_seq"

        # --- fixed conditioning vector: pooled context at the encoder's last real frame ---
        pooled_all = pool_context(context_seq, context_mask)          # (B, T_in, 2*F_ctx)
        last_in_idx = (input_lengths - 1).clamp(min=0)
        context_summary = pooled_all[torch.arange(B, device=device), last_in_idx]  # (B, 2*F_ctx)
        static_emb = self.static_embed(static_feats)                  # (B, static_embed_dim)
        conditioning = torch.cat([context_summary, static_emb], dim=-1)  # (B, cond_dim) -- same every step

        # --- precompute the teacher-forced "previous displacement" input for every step ---
        # true_disp[:, t] = the true displacement that produced true frame t (0-indexed);
        # prev_target[:, t] = last_position if t==0 else target_seq[:, t-1]
        if teacher_forcing:
            prev_target = torch.cat([last_position.unsqueeze(1), target_seq[:, :-1]], dim=1)
            true_disp = target_seq - prev_target                       # (B, T_out_max, 2)
            # step t's INPUT is the displacement from step t-1 (zero/start-token at t=0)
            teacher_forced_input = torch.cat(
                [torch.zeros(B, 1, 2, device=device), true_disp[:, :-1]], dim=1
            )  # (B, T_out_max, 2)

        output_lengths_f = output_lengths.to(device=device, dtype=last_position.dtype).clamp(min=1)

        pred_disp = torch.zeros(B, T_out_max, 2, device=device, dtype=last_position.dtype)
        pred_abs = torch.zeros(B, T_out_max, 2, device=device, dtype=last_position.dtype)

        running_pos = last_position.clone()  # anchor: last observed position
        prev_disp_input = torch.zeros(B, 2, device=device, dtype=last_position.dtype)  # only used when NOT teacher forcing

        for t in range(T_out_max):
            k = t + 1
            frac_done = k / output_lengths_f  # (B,) -- can exceed 1 past an example's true length; unused there
            progress = torch.stack([frac_done, 1.0 - frac_done], dim=-1)  # (B, 2)

            # teacher forcing: this step's input is precomputed (the true displacement into
            # step t-1, zero at t=0) -- no running update needed, just index column t directly.
            step_prev_disp = teacher_forced_input[:, t] if teacher_forcing else prev_disp_input

            step_in = torch.cat([step_prev_disp, progress, conditioning], dim=-1).unsqueeze(1)  # (B,1,D)
            out, (h, c) = self.lstm(step_in, (h, c))
            step_disp = self.output_head(out.squeeze(1))  # (B, 2)

            pred_disp[:, t] = step_disp
            running_pos = running_pos + step_disp
            pred_abs[:, t] = running_pos

            if not teacher_forcing:
                prev_disp_input = step_disp.detach()

        return pred_disp, pred_abs
