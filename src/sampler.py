"""
Length-bucketed BatchSampler.

Recipe (agreed design): each epoch, shuffle the full example index, chunk it
into pools of size batch_size * K, sort each pool by (output_len, input_len),
slice each sorted pool into batches of batch_size, then shuffle the resulting
batch order. This keeps sequences within a batch length-similar (less padding
waste) while keeping batch composition and batch order different every epoch.

No examples are dropped: every pool -- including a short final pool -- is
still sliced into batches, so a batch may occasionally be smaller than
batch_size (only possible at the very end of a pool), but every index in the
dataset appears in exactly one batch.
"""

import torch
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size: int, pool_multiplier: int = 50):
        """
        lengths: list of (input_len, output_len), aligned with the Dataset's
                 index order (e.g. from BDBTrajectoryDataset.get_lengths()).
        batch_size: target batch size.
        pool_multiplier (K): pool size = batch_size * K. Larger K -> more
                 global length-sorting (tighter buckets, less randomness
                 across epochs); smaller K -> more randomness, looser buckets.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_multiplier
        self.n = len(lengths)

    def __iter__(self):
        perm = torch.randperm(self.n).tolist()

        batches = []
        for pool_start in range(0, self.n, self.pool_size):
            pool = perm[pool_start: pool_start + self.pool_size]
            # sort this pool by output_len primary, input_len secondary
            pool.sort(key=lambda idx: (self.lengths[idx][1], self.lengths[idx][0]))
            for batch_start in range(0, len(pool), self.batch_size):
                batches.append(pool[batch_start: batch_start + self.batch_size])

        # shuffle batch order so early/late training steps aren't systematically
        # biased toward whichever pool happened to contain short/long sequences
        batch_order = torch.randperm(len(batches)).tolist()
        for i in batch_order:
            yield batches[i]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size
