"""
Timing benchmark ONLY -- not a training run. Times forward+backward+optimizer.step()
per batch on CPU vs CUDA, to make the device decision on measured wall time rather
than an assumption that "GPU is faster." The decoder is a sequential Python loop
over T_out_max steps, which is exactly the pattern where per-step kernel-launch
overhead can matter more than raw GPU throughput for a model this small.
"""

import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from collate import collate_fn
from dataset import BDBTrajectoryDataset
from loss import masked_rmse_loss
from model import TrajectoryModel
from sampler import LengthBucketBatchSampler

BATCH_SIZE = 64
N_WARMUP = 3
N_TIMED = 15


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run(device_name, batches, stats):
    device = torch.device(device_name)
    torch.manual_seed(0)
    F_self = batches[0]["self_seq"].shape[-1]
    F_ctx = batches[0]["context_seq"].shape[-1]
    F_static = batches[0]["static_feats"].shape[-1]
    model = TrajectoryModel(f_self=F_self, f_ctx=F_ctx, f_static=F_static, hidden_size=128, num_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    dev_batches = [move_batch(b, device) for b in batches]

    for b in dev_batches[:N_WARMUP]:
        opt.zero_grad()
        _, pred_abs = model(b, teacher_forcing=True)
        loss = masked_rmse_loss(pred_abs, b["target_seq"], b["output_mask"], stats)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for b in dev_batches[N_WARMUP:N_WARMUP + N_TIMED]:
        t0 = time.perf_counter()
        opt.zero_grad()
        _, pred_abs = model(b, teacher_forcing=True)
        loss = masked_rmse_loss(pred_abs, b["target_seq"], b["output_mask"], stats)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_t = sum(times) / len(times)
    return mean_t, min(times), max(times)


def main():
    stats = torch.load("data/processed/norm_stats.pt")
    ds = BDBTrajectoryDataset(split="train", norm_stats=stats)
    loader = DataLoader(
        ds, batch_sampler=LengthBucketBatchSampler(ds.get_lengths(), batch_size=BATCH_SIZE),
        collate_fn=collate_fn,
    )
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) >= N_WARMUP + N_TIMED:
            break

    T_out_maxes = [b["target_seq"].shape[1] for b in batches[N_WARMUP:N_WARMUP + N_TIMED]]
    print(f"batch_size={BATCH_SIZE}, {len(batches)} batches pulled from the real train loader")
    print(f"T_out_max across the {N_TIMED} timed batches: min={min(T_out_maxes)} max={max(T_out_maxes)} "
          f"(this is the decoder's sequential step count per batch)\n")

    steps_per_epoch = len(ds) // BATCH_SIZE

    cpu_mean, cpu_min, cpu_max = run("cpu", batches, stats)
    print(f"CPU : mean {cpu_mean*1000:7.1f} ms/batch  (min {cpu_min*1000:.1f}, max {cpu_max*1000:.1f})  "
          f"-> ~{cpu_mean*steps_per_epoch:.1f}s/epoch ({steps_per_epoch} steps)")

    if torch.cuda.is_available():
        gpu_mean, gpu_min, gpu_max = run("cuda", batches, stats)
        print(f"CUDA: mean {gpu_mean*1000:7.1f} ms/batch  (min {gpu_min*1000:.1f}, max {gpu_max*1000:.1f})  "
              f"-> ~{gpu_mean*steps_per_epoch:.1f}s/epoch ({steps_per_epoch} steps)")
        print(f"\nspeedup (CPU time / CUDA time): {cpu_mean/gpu_mean:.2f}x")
    else:
        print("CUDA not available in this process.")


if __name__ == "__main__":
    main()
