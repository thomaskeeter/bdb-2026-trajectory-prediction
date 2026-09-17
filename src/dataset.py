"""
PyTorch Dataset over the Stage 0 preprocessed examples.

Loads the per-week .pt files (produced by scripts/stage0_preprocess.py) once at
construction, builds a (game_id, play_id, nfl_id) -> example lookup, and filters
to one split ("train" or "val") using the split tag Stage 0 already assigned.

__getitem__ does a plain dict/list lookup into already-built tensors -- no CSV
parsing, no groupby, no feature computation happens here. All of that is Stage
0's job; this class is deliberately dumb and fast.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

ALL_WEEKS = [f"{w:02d}" for w in range(1, 19)]


class BDBTrajectoryDataset(Dataset):
    def __init__(self, processed_dir: str = "data/processed", split: str = "train", weeks=None):
        """
        split: "train", "val", or "all" (skip split filtering entirely)
        weeks: optional explicit list of week strings (e.g. ["01","02"]) to load,
               instead of all 18 -- mainly useful for quick smoke tests.
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        weeks = weeks if weeks is not None else ALL_WEEKS

        self._by_key = {}
        for week in weeks:
            path = self.processed_dir / f"week{week}_examples.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected preprocessed file not found: {path}. "
                    f"Run scripts/stage0_preprocess.py first."
                )
            week_examples = torch.load(path, weights_only=False)
            for ex in week_examples:
                key = (ex["game_id"], ex["play_id"], ex["nfl_id"])
                if key in self._by_key:
                    raise ValueError(f"Duplicate example key {key} found across loaded weeks")
                self._by_key[key] = ex

        if split == "all":
            self._keys = list(self._by_key.keys())
        else:
            self._keys = [k for k, ex in self._by_key.items() if ex["split"] == split]

        if len(self._keys) == 0:
            raise ValueError(
                f"No examples found for split={split!r} across weeks={weeks}. "
                f"Check that Stage 0 was run on weeks with that split tag."
            )

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict:
        key = self._keys[idx]
        ex = self._by_key[key]
        # Return as-is: self_seq/context_seq/context_mask/static_feats/target_seq
        # are already torch tensors; input_len/output_len are plain ints.
        # Padding to a common batch shape happens later, in collate_fn -- not here.
        return ex

    def get_lengths(self):
        """(input_len, output_len) per example, in the same order as __getitem__ --
        used by the length-bucketed BatchSampler without touching the tensors."""
        return [(self._by_key[k]["input_len"], self._by_key[k]["output_len"]) for k in self._keys]
