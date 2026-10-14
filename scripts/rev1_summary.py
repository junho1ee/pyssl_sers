#!/usr/bin/env python
"""Summarize the revision fine-tuning runs of steps 0, 0c, 1, and 2.

Every encoder is scored as in scripts/rev1_s0_summary.py: test accuracy at the
sample level from y_results.pt, and validation accuracy at the epoch of minimum
validation loss. All runs share the ten folds of Table 4, so differences
between two encoders are paired by fold.

Seed replicates (step 1) are reported at the published pretraining epoch (3000)
and at the epoch with the highest mean validation accuracy, chosen separately
for every seed. Ablations (step 2) are compared with the epoch-500 and
epoch-1000 checkpoints of the original seed 0 SimCLR v2 encoder.
Step 3 compares edge-held inputs (the four extrapolated points set to the last
measured value) with the original inputs.
Missing runs are reported as missing, so the script can be rerun as jobs finish.
"""

import argparse
import json
import os
import statistics as st

import numpy as np
import torch

from rev1_s0_summary import val_at_best

ROOT = "results/bacteria-id/finetuning/class30"
EPOCHS = [499, 999, 1499, 1999, 2499, 2999]
CONFIGS = [  # (view, objective, batch size, seed 0 version)
    ("phys", "simclrv2", 1024, "version_bs1024"),
    ("phys", "mocov3", 1024, "version_bs1024"),
    ("phys", "byol", 1024, "version_bs1024"),
    ("phys", "byol", 2048, "version_1"),
    ("crop", "simclrv2", 1024, "version_bs1024"),
    ("crop", "mocov3", 1024, "version_bs1024"),
    ("crop", "byol", 1024, "version_bs1024"),
    ("crop", "byol", 2048, "version_1"),
]
ABLATIONS = [
    "phys_no_powerline", "phys_no_emg", "phys_no_shift", "phys_no_wander",
    "phys_only_powerline", "phys_only_emg", "phys_only_shift", "phys_only_wander",
    "phys_shift_x0p1", "phys_all_x0p5", "phys_noise_x2", "phys_slow_x0p1",
]


def folds(repo, aug, pre, ver, tag):
    """Per-fold test and validation accuracy, or None unless all ten exist."""
    base = os.path.join(repo, ROOT, aug, pre, ver, tag, "no_aug")
    test, val = [], []
    for fold in range(10):
        cv = os.path.join(base, f"cv{fold}")
        yr = os.path.join(cv, "y_results.pt")
        mc = os.path.join(cv, "version_0", "metrics.csv")
        if not (os.path.exists(yr) and os.path.exists(mc)):
            return None
        d = torch.load(yr, map_location="cpu")
        yt = d["y_true"].cpu().numpy().astype(int).ravel()
        yp = d["y_pred"].cpu().numpy().astype(int).ravel()
        test.append(100.0 * float((yt == yp).mean()))
        val.append(val_at_best(mc)[0])
    return np.array(test), np.array(val)


def folds_flat(repo, aug, pre, tag):
    """As folds(), for runs without a pretrained version in their path."""
    return folds(repo, aug, pre, "", tag)


def ms(v):
    v = list(v)
    if not v:
        return "missing"
    return f"{st.mean(v):6.2f} ± {st.stdev(v):4.2f}" if len(v) > 1 else f"{v[0]:6.2f}"


def paired(a, b):
    d = a - b
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "n_pos": int((d > 0).sum()), "n": len(d)}


def encoder(repo, aug, pre, ver, tag_prefix, epochs):
    runs = {ep: folds(repo, aug, pre, ver, f"{tag_prefix}_e{ep}") for ep in epochs}
    return {ep: r for ep, r in runs.items() if r is not None}


def chosen(runs):
    """Epoch with the highest mean validation accuracy among the checkpoints."""
    return max(runs, key=lambda ep: runs[ep][1].mean()) if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/rev1_summary.json")
    args = ap.parse_args()
    out = {"seeds": {}, "ablations": {}}

    # Steps 0, 0c, 1: every configuration, three seeds.
    print("Seed replicates: test accuracy (%) over ten folds")
    print(f"{'config':24s} {'seed':>4s} {'e3000':>15s} {'val-chosen':>10s} {'test there':>15s}")
    seed_runs = {}
    for aug, pre, bs, ver0 in CONFIGS:
        key = f"{aug}/{pre}/bs{bs}"
        rows = {}
        for seed in (0, 1, 2):
            ver, tag = (ver0, "rev1_s0") if seed == 0 else (f"rev1_bs{bs}_seed{seed}", "rev1_s1")
            runs = encoder(args.repo, aug, pre, ver, tag, EPOCHS)
            seed_runs[(key, seed)] = runs
            c = chosen(runs) if len(runs) == len(EPOCHS) else None
            e3 = runs.get(2999)
            rows[seed] = {
                "version": ver, "n_epochs_done": len(runs),
                "e3000": None if e3 is None else {"mean": float(e3[0].mean()), "sd": float(e3[0].std(ddof=1))},
                "val_chosen_epoch": None if c is None else c + 1,
                "val_chosen": None if c is None else {"mean": float(runs[c][0].mean()), "sd": float(runs[c][0].std(ddof=1))},
                "by_epoch": {ep + 1: {"test": runs[ep][0].tolist(), "val": runs[ep][1].tolist()} for ep in runs},
            }
            print(f"{key:24s} {seed:4d} {ms(e3[0]) if e3 is not None else 'missing':>15s} "
                  f"{(c + 1) if c is not None else '-':>10} "
                  f"{ms(runs[c][0]) if c is not None else 'missing':>15s}")
        done = [s for s in rows if rows[s]["e3000"]]
        if len(done) > 1:
            m3 = [rows[s]["e3000"]["mean"] for s in done]
            print(f"{'':24s} {'all':>4s} {ms(m3):>15s}   (SD over {len(done)} seed means)")
        out["seeds"][key] = rows

    # View effect and objective gaps per seed, paired by fold.
    print("\nPaired by fold, per seed (mean ± SD over folds, folds > 0):")
    comps = [("phys/simclrv2/bs1024", "crop/simclrv2/bs1024"),
             ("phys/mocov3/bs1024", "crop/mocov3/bs1024"),
             ("phys/byol/bs1024", "crop/byol/bs1024"),
             ("phys/byol/bs2048", "crop/byol/bs2048"),
             ("phys/simclrv2/bs1024", "phys/mocov3/bs1024"),
             ("phys/mocov3/bs1024", "phys/byol/bs2048")]
    out["paired"] = {}
    for a, b in comps:
        for seed in (0, 1, 2):
            ra, rb = seed_runs[(a, seed)], seed_runs[(b, seed)]
            if 2999 in ra and 2999 in rb:
                p = paired(ra[2999][0], rb[2999][0])
                out["paired"][f"{a} - {b} seed{seed} e3000"] = p
                print(f"  {a} - {b} seed {seed} e3000: {p['mean']:+6.2f} ± {p['sd']:4.2f} ({p['n_pos']}/{p['n']})")

    # Step 2: ablations against the original encoder at the same epochs.
    print("\nAblations (SimCLR v2, seed 0): test accuracy and paired difference from phys")
    ref = encoder(args.repo, "phys", "simclrv2", "version_bs1024", "rev1_s0", [499, 999])
    for ep in (499, 999):
        if ep in ref:
            print(f"  {'phys (reference)':22s} e{ep + 1}: {ms(ref[ep][0])}")
    for aug in ABLATIONS:
        runs = encoder(args.repo, aug, "simclrv2", "rev1_bs1024_seed0_e1000", "rev1_s2", [499, 999])
        out["ablations"][aug] = {}
        for ep in (499, 999):
            if ep not in runs:
                print(f"  {aug:22s} e{ep + 1}: missing")
                continue
            p = paired(runs[ep][0], ref[ep][0]) if ep in ref else None
            out["ablations"][aug][ep + 1] = {"test": runs[ep][0].tolist(), "val": runs[ep][1].tolist(),
                                             "minus_phys": p}
            diff = f"{p['mean']:+6.2f} ± {p['sd']:4.2f} ({p['n_pos']}/{p['n']})" if p else ""
            print(f"  {aug:22s} e{ep + 1}: {ms(runs[ep][0])}  {diff}")

    # Step 3: extrapolated points held at the last measured value.
    print("\nExtrapolated points (Table 4 protocol): original vs edge-held inputs")
    pairs = [("random init", ("phys", "no_pre", "rev1_s3"), ("phys", "no_pre/edgehold", "rev1_s3")),
             ("SimCLR v2 e3000", ("phys", "simclrv2/version_bs1024", "rev1_s0_e2999"),
              ("phys", "simclrv2/version_bs1024/edgehold", "rev1_s3"))]
    out["edgehold"] = {}
    for name, (a1, p1, t1), (a2, p2, t2) in pairs:
        r1 = folds(args.repo, a1, *p1.split("/", 1), t1) if "/" in p1 else folds_flat(args.repo, a1, p1, t1)
        r2 = folds(args.repo, a2, *p2.split("/", 1), t2)
        if r1 is None or r2 is None:
            print(f"  {name}: missing")
            continue
        p = paired(r2[0], r1[0])
        out["edgehold"][name] = {"original": r1[0].tolist(), "edgehold": r2[0].tolist(), "edgehold_minus_original": p}
        print(f"  {name:16s} original {ms(r1[0])}  edge-held {ms(r2[0])}  "
              f"difference {p['mean']:+6.2f} ± {p['sd']:4.2f} ({p['n_pos']}/{p['n']})")

    # Steps 3b/3c: pretraining inputs x fine-tuning inputs, SimCLR v2 epoch 1000,
    # for the phys view (3b) and the crop view (3c).
    cells = {
        ("original", "original"): ("version_bs1024", "rev1_s0_e999"),
        ("original", "edge-held"): ("version_bs1024/edgehold", "rev1_s3_e999"),
        ("edge-held", "original"): ("rev1_bs1024_seed0_e1000_edgehold", "rev1_s3_e999"),
        ("edge-held", "edge-held"): ("rev1_bs1024_seed0_e1000_edgehold/edgehold", "rev1_s3_e999"),
    }
    grid = {}
    for view, key in (("phys", "edgehold_2x2"), ("crop", "edgehold_2x2_crop")):
        print(f"\nPretraining inputs x fine-tuning inputs (SimCLR v2 {view} seed 0, epoch 1000)")
        out[key] = {}
        ref22 = folds(args.repo, view, "simclrv2", "version_bs1024", "rev1_s0_e999")
        for (pre_in, ft_in), (ver, tag) in cells.items():
            r = folds(args.repo, view, "simclrv2", ver, tag)
            if r is None:
                print(f"  pretrain {pre_in:9s} fine-tune {ft_in:9s}: missing")
                continue
            grid[view, pre_in, ft_in] = r[0]
            p = paired(r[0], ref22[0]) if ref22 is not None else None
            out[key][f"{pre_in}/{ft_in}"] = {"test": r[0].tolist(), "minus_original_original": p}
            print(f"  pretrain {pre_in:9s} fine-tune {ft_in:9s}: {ms(r[0])}  "
                  f"vs original/original {p['mean']:+6.2f} ± {p['sd']:4.2f} ({p['n_pos']}/{p['n']})")
    # View effect (phys minus crop) per cell, and its change from original inputs.
    print("\nView effect phys minus crop by input cell (SimCLR v2 seed 0, epoch 1000)")
    out["edgehold_view_effect"] = {}
    for pre_in, ft_in in cells:
        if not all((v, pre_in, ft_in) in grid for v in ("phys", "crop")):
            print(f"  pretrain {pre_in:9s} fine-tune {ft_in:9s}: missing")
            continue
        eff = grid["phys", pre_in, ft_in] - grid["crop", pre_in, ft_in]
        rec = {"effect": paired(grid["phys", pre_in, ft_in], grid["crop", pre_in, ft_in])}
        line = f"  pretrain {pre_in:9s} fine-tune {ft_in:9s}: {eff.mean():+6.2f} ± {eff.std(ddof=1):4.2f} ({(eff > 0).sum()}/10)"
        if (pre_in, ft_in) != ("original", "original") and all((v, "original", "original") in grid for v in ("phys", "crop")):
            base = grid["phys", "original", "original"] - grid["crop", "original", "original"]
            rec["minus_original_effect"] = paired(eff, base)
            q = rec["minus_original_effect"]
            line += f"  change from original {q['mean']:+6.2f} ± {q['sd']:4.2f} ({q['n_pos']}/{q['n']})"
        out["edgehold_view_effect"][f"{pre_in}/{ft_in}"] = rec
        print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
