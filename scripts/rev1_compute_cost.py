"""Compute cost for R2-Q: memory and time of one pretraining step, and inference cost.

Pretraining: the model of lightning_pretrain_ssl.get_model for each objective,
trained as there (Adam, lr 1e-3, 16-bit mixed precision with gradient scaling)
on two views of real reference spectra. Measures peak GPU memory
(torch.cuda.max_memory_allocated / reserved) and GPU time per optimizer step,
excluding data loading and augmentation, which run on the CPU workers. Wall
time per epoch of the real runs is taken from their logs, not from here.

Inference: the fine-tuning classifier (encoder + linear head, 30 classes, as in
lightning_finetune_pred) in eval mode, fp32, on the GPU and on the CPU threads
of the job: throughput at batch 256 and latency of a single spectrum.

Run as a job on the GPU node (hpc/rev1/q1_compute_cost.sbatch); writes
results/rev1_compute_cost.json.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightning_pretrain_ssl as lps  # noqa: E402
from nets.resnet import ResNet  # noqa: E402

CONFIGS = [("simclrv2", 1024), ("mocov3", 1024), ("byol", 1024), ("byol", 2048)]


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def pretrain_step_cost(pre, bs, x, warmup=5, steps=30):
    cfg = OmegaConf.load(f"configs/bacteria-id/pretraining/phys/{pre}.yaml")
    cfg.pre, cfg.augtype = pre, "phys"
    model = lps.get_model(cfg).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.RandomState(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(warmup + steps):
        idx = rng.choice(len(x), size=bs, replace=False)
        x1 = torch.tensor(x[idx, None, :], device="cuda")
        x2 = torch.tensor(x[rng.choice(len(x), size=bs, replace=False), None, :], device="cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x1, x2)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    out = {"objective": pre, "batch_size": bs,
           "params_total": n_params(model), "params_encoder": n_params(model.backbone) if hasattr(model, "backbone") else None,
           "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
           "gpu_s_per_step_median": float(np.median(times)),
           "steps_per_epoch": int(np.ceil(len(x) / bs))}
    out["gpu_s_per_epoch_compute_only"] = out["gpu_s_per_step_median"] * out["steps_per_epoch"]
    del model, opt
    torch.cuda.empty_cache()
    return out


def classifier():
    m = ResNet([100] * 6, [2] * 6, input_dim=696, in_channels=64, n_classes=30)
    return m.eval()


def inference_cost(device, x, batch=256, reps=20):
    m = classifier().to(device)
    xb = torch.tensor(x[:batch, None, :], device=device)
    x1 = torch.tensor(x[:1, None, :], device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(3):
            m(xb); m(x1)
        sync(); t0 = time.perf_counter()
        for _ in range(reps):
            m(xb)
        sync(); tb = (time.perf_counter() - t0) / reps
        lat = []
        for _ in range(reps * 5):
            sync(); t0 = time.perf_counter(); m(x1); sync(); lat.append(time.perf_counter() - t0)
    return {"device": device, "params": n_params(m), "batch": batch,
            "spectra_per_s": batch / tb, "single_spectrum_latency_ms_median": 1000 * float(np.median(lat))}


def main():
    torch.backends.cudnn.benchmark = False
    x = np.load("data/bacteria-id/preprocessed/X_reference.npy").astype(np.float32)
    res = {"gpu": torch.cuda.get_device_name(0), "gpu_total_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
           "torch": torch.__version__, "cpu": platform.processor() or platform.machine(),
           "cpu_threads": torch.get_num_threads(), "n_reference_spectra": int(len(x)),
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    res["pretraining"] = [pretrain_step_cost(p, b, x) for p, b in CONFIGS]
    res["inference"] = [inference_cost("cuda", x), inference_cost("cpu", x)]
    print(json.dumps(res, indent=1))
    os.makedirs("results", exist_ok=True)
    with open("results/rev1_compute_cost.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
