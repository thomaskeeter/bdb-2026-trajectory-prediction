"""
Real training loop: epochs over the full train split, per-epoch validation
(teacher-forced AND free-running), checkpointing, early stopping.

Signed-off config: batch_size=128, lr=1e-3 (Adam), grad clip max_norm=5.0,
max_epochs=40, early-stop patience=5 on val_rmse_free (the free-running/
autoregressive val RMSE -- what an actual submission would score, since the
competition gives no ground truth to teacher-force against at inference).
val_rmse_tf (teacher-forced) is tracked alongside purely as a diagnostic, to
watch the exposure-bias gap between the two.

checkpoints/latest.pt -- overwritten every epoch, for resuming.
checkpoints/best.pt   -- saved only when val_rmse_free improves.
Both under checkpoints/, which *.pt in .gitignore already excludes.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from collate import collate_fn
from dataset import BDBTrajectoryDataset
from loss import masked_rmse_loss
from metrics import RMSEAccumulator
from model import TrajectoryModel
from normalization import denormalize_xy
from sampler import LengthBucketBatchSampler

CHECKPOINT_DIR = Path("checkpoints")
NORM_STATS_PATH = "data/processed/norm_stats.pt"

DEFAULTS = dict(
    batch_size=128,
    lr=1e-3,
    grad_clip=5.0,
    max_epochs=40,
    patience=5,
    hidden_size=128,
    num_layers=2,
    seed=42,
)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(split: str, stats: dict, batch_size: int, weeks=None) -> DataLoader:
    ds = BDBTrajectoryDataset(split=split, norm_stats=stats, weeks=weeks)
    sampler = LengthBucketBatchSampler(ds.get_lengths(), batch_size=batch_size)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn), ds


def run_train_epoch(model, loader, opt, stats, device, grad_clip):
    """One pass over train, updating weights. Returns epoch train RMSE (accumulated
    the same way RMSEAccumulator does elsewhere -- sum of squared error / sum of real
    frames -- NOT a naive average of per-batch loss.item() values, since that would
    over-weight small batches relative to their actual number of scored frames."""
    model.train()
    acc = RMSEAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        opt.zero_grad()
        _, pred_abs = model(batch, teacher_forcing=True)
        loss = masked_rmse_loss(pred_abs, batch["target_seq"], batch["output_mask"], stats)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        # reuse the already-computed (teacher-forced) predictions for epoch-level
        # logging -- no extra forward pass needed
        pred_yards = denormalize_xy(pred_abs.detach(), stats)
        target_yards = denormalize_xy(batch["target_seq"], stats)
        acc.update(pred_yards, target_yards, batch["output_mask"])
    return acc.compute()


@torch.no_grad()
def run_val_epoch(model, loader, stats, device, teacher_forcing: bool):
    model.eval()
    acc = RMSEAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        _, pred_abs = model(batch, teacher_forcing=teacher_forcing)
        pred_yards = denormalize_xy(pred_abs, stats)
        target_yards = denormalize_xy(batch["target_seq"], stats)
        acc.update(pred_yards, target_yards, batch["output_mask"])
    return acc.compute()


def train(max_epochs=None, patience=None, batch_size=None, lr=None, grad_clip=None,
          hidden_size=None, num_layers=None, seed=None, label=""):
    cfg = dict(DEFAULTS)
    overrides = dict(max_epochs=max_epochs, patience=patience, batch_size=batch_size, lr=lr,
                      grad_clip=grad_clip, hidden_size=hidden_size, num_layers=num_layers, seed=seed)
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{label}config: {cfg}")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    stats = torch.load(NORM_STATS_PATH)
    train_loader, train_ds = make_loader("train", stats, cfg["batch_size"])
    val_loader, val_ds = make_loader("val", stats, cfg["batch_size"])
    print(f"train: {len(train_ds):,} examples ({len(train_loader)} batches)  "
          f"val: {len(val_ds):,} examples ({len(val_loader)} batches)")

    F_self = 11
    F_ctx = 12
    F_static = 7
    model = TrajectoryModel(f_self=F_self, f_ctx=F_ctx, f_static=F_static,
                             hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_rmse_free = float("inf")
    epochs_since_improve = 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        train_rmse = run_train_epoch(model, train_loader, opt, stats, device, cfg["grad_clip"])
        val_rmse_tf = run_val_epoch(model, val_loader, stats, device, teacher_forcing=True)
        val_rmse_free = run_val_epoch(model, val_loader, stats, device, teacher_forcing=False)
        elapsed = time.time() - t0

        improved = val_rmse_free < best_val_rmse_free
        if improved:
            best_val_rmse_free = val_rmse_free
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        status = "*improved*" if improved else f"no improve ({epochs_since_improve}/{cfg['patience']})"
        print(f"epoch {epoch:3d}/{cfg['max_epochs']}  "
              f"train {train_rmse:.4f}  val_tf {val_rmse_tf:.4f}  val_free {val_rmse_free:.4f}  "
              f"{status}  [{elapsed:.1f}s]")

        history.append(dict(epoch=epoch, train_rmse=train_rmse, val_rmse_tf=val_rmse_tf,
                             val_rmse_free=val_rmse_free, improved=improved, elapsed=elapsed))

        ckpt = dict(
            epoch=epoch, model_state_dict=model.state_dict(), optimizer_state_dict=opt.state_dict(),
            train_rmse=train_rmse, val_rmse_tf=val_rmse_tf, val_rmse_free=val_rmse_free,
            best_val_rmse_free=best_val_rmse_free, cfg=cfg,
        )
        torch.save(ckpt, CHECKPOINT_DIR / "latest.pt")
        if improved:
            torch.save(ckpt, CHECKPOINT_DIR / "best.pt")

        if epochs_since_improve >= cfg["patience"]:
            print(f"early stopping: no val_rmse_free improvement in {cfg['patience']} epochs")
            break

    print(f"\ndone. best val_rmse_free = {best_val_rmse_free:.4f} yards "
          f"(checkpoints/best.pt)")
    return history, best_val_rmse_free


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()
    train(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size, lr=args.lr)
