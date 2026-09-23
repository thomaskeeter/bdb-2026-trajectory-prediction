"""
Basic overfitting sanity check -- NOT real training. Repeatedly trains on the
SAME single fixed batch to confirm the loss can actually go down, i.e. that
gradients are flowing correctly end to end through encoder -> decoder -> loss.
This says nothing about generalization; it only rules out a broken/no-op
training step.
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from collate import collate_fn
from dataset import BDBTrajectoryDataset
from loss import masked_rmse_loss
from model import TrajectoryModel
from sampler import LengthBucketBatchSampler

torch.manual_seed(0)
N_STEPS = 30


def main():
    stats = torch.load("data/processed/norm_stats.pt")
    ds = BDBTrajectoryDataset(split="train", norm_stats=stats)
    loader = DataLoader(
        ds, batch_sampler=LengthBucketBatchSampler(ds.get_lengths(), batch_size=32),
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))  # ONE fixed batch, reused every step

    F_self = batch["self_seq"].shape[-1]
    F_ctx = batch["context_seq"].shape[-1]
    F_static = batch["static_feats"].shape[-1]
    model = TrajectoryModel(f_self=F_self, f_ctx=F_ctx, f_static=F_static, hidden_size=128, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"overfitting a single fixed batch (size {batch['self_seq'].shape[0]}) "
          f"for {N_STEPS} steps -- sanity check only, not real training\n")

    model.train()
    losses = []
    for step in range(N_STEPS):
        opt.zero_grad()
        _, pred_abs = model(batch, teacher_forcing=True)
        loss = masked_rmse_loss(pred_abs, batch["target_seq"], batch["output_mask"], stats)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 3 == 0 or step == N_STEPS - 1:
            print(f"  step {step:2d}: loss = {loss.item():.4f} yards RMSE")

    non_increasing = sum(1 for i in range(1, len(losses)) if losses[i] <= losses[i - 1])
    print(f"\nloss[0] = {losses[0]:.4f}  loss[-1] = {losses[-1]:.4f}  decreased: {losses[-1] < losses[0]}")
    print(f"non-increasing steps: {non_increasing}/{len(losses)-1}")
    print("for context: constant_velocity baseline RMSE was ~1.6-1.7 yards on the full train/val split")
    assert losses[-1] < losses[0], "loss did not decrease over the fixed batch -- training step is likely broken"

    print("\nOVERFIT SANITY CHECK: PASSED")


if __name__ == "__main__":
    main()
