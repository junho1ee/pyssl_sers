#!/usr/bin/env python
"""Summarize revision step 0: downstream accuracy vs pretraining epoch.

Reads the rev1_s0_e<epoch> fine-tuning runs written by hpc/rev1/s0_ckpt_probe.sbatch.
Test accuracy is scored at the sample level from y_results.pt, the convention of
scripts/recompute_paper_tables.py, so the epoch-2999 rows are directly comparable
with Table 4. Validation accuracy is read at the epoch of minimum validation loss,
which is the checkpoint that was tested, so that a pretraining epoch can also be
chosen without looking at the test subset.
"""

import argparse
import json
import os
import statistics as st

import numpy as np
import pandas as pd
import torch

ENCODERS = [
    ("simclrv2", "version_bs1024", 1024),
    ("mocov3", "version_bs1024", 1024),
    ("byol", "version_bs1024", 1024),
    ("byol", "version_1", 2048),
]
EPOCHS = [499, 999, 1499, 1999, 2499, 2999]
N_REFERENCE = 60000  # unlabeled Bacteria-ID reference spectra
# Table 4 and the batch size 2048 control as published, for the epoch-2999 check.
PUBLISHED = {
    ("simclrv2", "version_bs1024"): 80.5,
    ("mocov3", "version_bs1024"): 78.9,
    ("byol", "version_bs1024"): 73.2,
    ("byol", "version_1"): 78.6,
}


def val_at_best(metrics_csv):
    m = pd.read_csv(metrics_csv)
    loss_col = "valid_loss_epoch" if "valid_loss_epoch" in m else "valid_loss"
    acc_col = "valid_acc_epoch" if "valid_acc_epoch" in m else "valid_acc"
    v = m.dropna(subset=[loss_col])
    best = v.loc[v[loss_col].idxmin()]
    return float(best[acc_col]), float(best[loss_col])


def fold_runs(root, pre, ver, ep):
    base = os.path.join(root, pre, ver, f"rev1_s0_e{ep}", "no_aug")
    out = []
    for fold in range(10):
        cv = os.path.join(base, f"cv{fold}")
        yr = os.path.join(cv, "y_results.pt")
        if not os.path.exists(yr):
            continue
        d = torch.load(yr, map_location="cpu")
        yt = d["y_true"].cpu().numpy().astype(int).ravel()
        yp = d["y_pred"].cpu().numpy().astype(int).ravel()
        va, vl = val_at_best(os.path.join(cv, "version_0", "metrics.csv"))
        out.append({"fold": fold, "test": 100.0 * float((yt == yp).mean()),
                    "val_acc": va, "val_loss": vl})
    return out


def mean_sd(v):
    return (st.mean(v), st.stdev(v)) if len(v) > 1 else (v[0], 0.0) if v else (float("nan"), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/rev1_s0_summary.json")
    args = ap.parse_args()
    root = os.path.join(args.repo, "results/bacteria-id/finetuning/class30/phys")

    summary = {}
    for pre, ver, bs in ENCODERS:
        steps = -(-N_REFERENCE // bs)
        print(f"\n{pre} {ver} (batch {bs}, {steps} updates/epoch)")
        print(f"{'epoch':>6} {'updates':>8} {'n':>3} {'test':>13} {'val acc':>13}")
        rows = {}
        for ep in EPOCHS:
            runs = fold_runs(root, pre, ver, ep)
            t = mean_sd([r["test"] for r in runs])
            v = mean_sd([r["val_acc"] for r in runs])
            upd = (ep + 1) * steps
            rows[ep] = {"updates": upd, "n": len(runs), "test": t, "val_acc": v,
                        "folds": runs}
            print(f"{ep + 1:>6} {upd:>8} {len(runs):>3} "
                  f"{t[0]:6.2f} ± {t[1]:4.2f} {v[0]:6.2f} ± {v[1]:4.2f}")
        best_val = max(rows, key=lambda e: rows[e]["val_acc"][0])
        best_test = max(rows, key=lambda e: rows[e]["test"][0])
        print(f"  epoch chosen by validation: {best_val + 1} -> test "
              f"{rows[best_val]['test'][0]:.2f}; best by test: {best_test + 1}")
        print(f"  epoch 3000 here {rows[2999]['test'][0]:.2f} vs published "
              f"{PUBLISHED[(pre, ver)]}")
        summary[f"{pre}/{ver}"] = {"batch_size": bs, "epochs": rows,
                                   "chosen_by_validation": best_val,
                                   "published_e2999": PUBLISHED[(pre, ver)]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
