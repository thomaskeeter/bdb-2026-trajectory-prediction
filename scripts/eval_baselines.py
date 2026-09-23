"""
Evaluate the stay-put and constant-velocity baselines with the competition RMSE,
through the real Dataset -> BatchSampler -> collate_fn path (raw, un-normalized
data so everything is in yards).
"""

import sys

sys.path.insert(0, "src")
from baselines import constant_velocity, stay_put
from collate import collate_fn
from dataset import BDBTrajectoryDataset
from metrics import RMSEAccumulator
from normalization import unmirror_xy
from sampler import LengthBucketBatchSampler
from torch.utils.data import DataLoader

ROLE_COLS = {3: "Passer", 4: "Other Route Runner", 5: "Targeted Receiver", 6: "Defensive Coverage"}
SHOW_STEPS = [1, 2, 3, 5, 10, 15, 20, 30]


def role_tags(batch):
    onehot = batch["static_feats"][:, 3:7]
    return [ROLE_COLS[3 + int(r)] for r in onehot.argmax(dim=1)]


def evaluate(split: str, in_field_coords: bool = False):
    ds = BDBTrajectoryDataset(split=split)  # no norm_stats -> raw yards
    loader = DataLoader(
        ds,
        batch_sampler=LengthBucketBatchSampler(ds.get_lengths(), batch_size=64),
        collate_fn=collate_fn,
    )
    accs = {"stay_put": RMSEAccumulator(), "constant_velocity": RMSEAccumulator()}
    fns = {"stay_put": stay_put, "constant_velocity": constant_velocity}
    for batch in loader:
        true = batch["target_seq"]
        tags = role_tags(batch)
        for name, fn in fns.items():
            pred = fn(batch)
            t = true
            if in_field_coords:  # undo the x-mirroring on BOTH -- the metric must not care
                pred, t = unmirror_xy(pred, batch["is_left"]), unmirror_xy(true, batch["is_left"])
            accs[name].update(pred, t, batch["output_mask"], tags=tags)
    return ds, accs


def report(split):
    ds, accs = evaluate(split)
    n = next(iter(accs.values())).n
    print(f"\n{'=' * 78}\n{split.upper()} split: {len(ds):,} examples, N = {n:,} scored rows (player-frames)\n{'=' * 78}")
    print(f"{'baseline':<20}{'overall RMSE (yards)':>24}")
    for name, acc in accs.items():
        print(f"{name:<20}{acc.compute():>24.4f}")

    print("\nby role (RMSE yards | rows):")
    for name, acc in accs.items():
        parts = "   ".join(f"{role}: {r:.4f} ({cnt:,})" for role, (r, cnt) in sorted(acc.compute_by_tag().items()))
        print(f"  {name:<18}{parts}")

    print("\nby output frame k (0.1 s each) -- RMSE yards:")
    header = "  " + f"{'baseline':<20}" + "".join(f"{'k=' + str(k):>9}" for k in SHOW_STEPS)
    print(header)
    for name, acc in accs.items():
        by_step = acc.compute_by_step()
        print("  " + f"{name:<20}" + "".join(f"{by_step.get(k, float('nan')):>9.3f}" for k in SHOW_STEPS))

    # the metric is a distance in yards, so it must not care that x was mirrored
    _, field_accs = evaluate(split, in_field_coords=True)
    print("\nmirroring sanity check (metric in un-mirrored FIELD coords vs mirrored frame):")
    for name in accs:
        a, b = accs[name].compute(), field_accs[name].compute()
        print(f"  {name:<20}mirrored {a:.8f}   field {b:.8f}   |diff| = {abs(a - b):.2e}")


if __name__ == "__main__":
    for split in ("val", "train"):
        report(split)
