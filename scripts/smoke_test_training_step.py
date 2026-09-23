"""
Smoke test: prove one full training step (forward -> loss -> backward ->
optimizer.step()) completes without error on a real batch, with no dead
gradient branches (every parameter gets a finite, non-None gradient) and the
optimizer step actually changes the weights.
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


def main():
    stats = torch.load("data/processed/norm_stats.pt")
    ds = BDBTrajectoryDataset(split="train", norm_stats=stats)
    loader = DataLoader(
        ds, batch_sampler=LengthBucketBatchSampler(ds.get_lengths(), batch_size=32),
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    F_self = batch["self_seq"].shape[-1]
    F_ctx = batch["context_seq"].shape[-1]
    F_static = batch["static_feats"].shape[-1]
    model = TrajectoryModel(f_self=F_self, f_ctx=F_ctx, f_static=F_static, hidden_size=128, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"batch size: {batch['self_seq'].shape[0]}  T_in_max: {batch['self_seq'].shape[1]}  "
          f"T_out_max: {batch['target_seq'].shape[1]}")

    model.train()
    opt.zero_grad()
    pred_disp, pred_abs = model(batch, teacher_forcing=True)
    loss = masked_rmse_loss(pred_abs, batch["target_seq"], batch["output_mask"], stats)
    print(f"forward pass ok. loss = {loss.item():.6f} yards RMSE (untrained, random init)")
    loss.backward()

    missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
    nan_grad = [n for n, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"params with NO gradient (should be empty): {missing_grad}")
    print(f"params with NaN gradient (should be empty): {nan_grad}")
    assert not missing_grad and not nan_grad, "dead or broken gradient found"

    w_before = model.encoder.input_proj.weight.detach().clone()
    opt.step()
    w_after = model.encoder.input_proj.weight.detach().clone()
    changed = not torch.equal(w_before, w_after)
    print(f"optimizer.step() completed. weights actually changed: {changed}")
    assert changed

    print("\nSINGLE TRAINING STEP: PASSED")


if __name__ == "__main__":
    main()
