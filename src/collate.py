"""
collate_fn: turns a list of Dataset examples (already length-similar, thanks
to the LengthBucketBatchSampler, but not identical) into one padded batch.

Padding target: THIS BATCH's own max input_len / max output_len -- not the
global dataset max -- since that's the whole point of length bucketing.
"""

import torch


def collate_fn(batch: list[dict]) -> dict:
    B = len(batch)
    input_lens = [ex["input_len"] for ex in batch]
    output_lens = [ex["output_len"] for ex in batch]
    T_in = max(input_lens)
    T_out = max(output_lens)

    F_self = batch[0]["self_seq"].shape[1]
    max_ctx = batch[0]["context_seq"].shape[1]
    F_ctx = batch[0]["context_seq"].shape[2]
    F_static = batch[0]["static_feats"].shape[0]

    self_seq = torch.zeros(B, T_in, F_self)
    context_seq = torch.zeros(B, T_in, max_ctx, F_ctx)
    context_mask = torch.zeros(B, max_ctx, dtype=torch.bool)
    static_feats = torch.zeros(B, F_static)
    target_seq = torch.zeros(B, T_out, 2)
    output_mask = torch.zeros(B, T_out, dtype=torch.bool)
    input_lengths = torch.zeros(B, dtype=torch.long)
    output_lengths = torch.zeros(B, dtype=torch.long)
    is_left = torch.tensor([ex["is_left"] for ex in batch], dtype=torch.bool)

    for i, ex in enumerate(batch):
        L_in, L_out = ex["input_len"], ex["output_len"]
        self_seq[i, :L_in] = ex["self_seq"]
        context_seq[i, :L_in] = ex["context_seq"]
        context_mask[i] = ex["context_mask"]
        static_feats[i] = ex["static_feats"]
        target_seq[i, :L_out] = ex["target_seq"]
        output_mask[i, :L_out] = True
        input_lengths[i] = L_in
        output_lengths[i] = L_out

    return {
        "self_seq": self_seq,               # (B, T_in, F_self)
        "context_seq": context_seq,         # (B, T_in, max_ctx, F_ctx)
        "context_mask": context_mask,       # (B, max_ctx)
        "static_feats": static_feats,       # (B, F_static)
        "target_seq": target_seq,           # (B, T_out, 2)  -- padded frames are 0, ignore via output_mask
        "output_mask": output_mask,         # (B, T_out) bool -- True = real frame, use in loss
        "input_lengths": input_lengths,     # (B,) for pack_padded_sequence
        "output_lengths": output_lengths,   # (B,)
        "is_left": is_left,                 # (B,) bool -- NOT a model input; used to un-mirror predictions
        "game_id": [ex["game_id"] for ex in batch],
        "play_id": [ex["play_id"] for ex in batch],
        "nfl_id": [ex["nfl_id"] for ex in batch],
    }
