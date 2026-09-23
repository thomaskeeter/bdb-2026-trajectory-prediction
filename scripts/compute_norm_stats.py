"""
Compute z-score stats over the TRAIN split only (weeks 1-16) and save them to
data/processed/norm_stats.pt. Requires Stage 0 output (with is_left) to exist.
"""

import sys

import torch

sys.path.insert(0, "src")
from dataset import BDBTrajectoryDataset
from normalization import (
    CTX_COL_NAMES, SELF_COL_NAMES, STATIC_COL_NAMES, compute_norm_stats,
)

OUT_PATH = "data/processed/norm_stats.pt"


def main():
    raw_train = BDBTrajectoryDataset(split="train")  # no stats -> raw values
    print(f"train examples: {len(raw_train)}")
    stats = compute_norm_stats(raw_train[i] for i in range(len(raw_train)))
    torch.save(stats, OUT_PATH)

    def show(title, names, mean, std, n, unit):
        print(f"\n{title}  (n = {n:,} {unit})")
        print(f"  {'feature':<18}{'mean':>12}{'std':>12}")
        for name, m, s in zip(names, mean.tolist(), std.tolist()):
            print(f"  {name:<18}{m:>12.4f}{s:>12.4f}")

    show("self_seq", SELF_COL_NAMES, stats["self_mean"], stats["self_std"], stats["self_n"], "frames")
    show("context_seq (real players only)", CTX_COL_NAMES, stats["ctx_mean"], stats["ctx_std"], stats["ctx_n"], "player-frames")
    show("static_feats", STATIC_COL_NAMES, stats["static_mean"], stats["static_std"], stats["static_n"], "examples")
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
