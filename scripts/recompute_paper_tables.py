#!/usr/bin/env python
"""Recompute every Bacteria-ID number in the manuscript from stored predictions.

Reads results/bacteria-id/finetuning/**/cv*/y_results.pt and scores them at the
sample level (all test spectra pooled per fold), which is the convention used by
results/label_efficiency.json and results/matched_ssl_comparison.json.  The
`test_acc` stored in test_results.json is a batch-average and therefore differs
from the sample-level value by ~0.05 pp whenever the last batch is short.

Also derives the 8-empiric-treatment accuracy from the 30-isolate predictions
using the antibiotic map of scripts/generate_bacteria_figures.py, so the
"8-treatments" column of Table 2 no longer needs a separate set of runs.
"""

import argparse
import json
import os
import re
import statistics as st
from collections import defaultdict

import numpy as np
import torch

# 30 isolates -> 8 empiric treatment groups (scripts/generate_bacteria_figures.py)
ATCC_GROUPINGS = {
    3: 0, 4: 0, 9: 0, 10: 0, 2: 0, 8: 0, 11: 0, 22: 0,
    12: 2, 13: 2,
    14: 3, 18: 3, 15: 3, 20: 3, 21: 3, 16: 3, 17: 3,
    23: 4, 24: 4,
    26: 5, 27: 5, 28: 5, 29: 5, 25: 5, 6: 5, 7: 5,
    5: 6,
    19: 1,
    0: 7, 1: 7,
}


def map8(a):
    out = np.empty_like(a)
    for k, v in ATCC_GROUPINGS.items():
        out[a == k] = v
    return out


def collect(root):
    """group -> {'acc30': [...], 'acc8': [...]} at sample level, one entry per fold."""
    groups = defaultdict(lambda: defaultdict(list))
    for dp, _, fn in os.walk(root):
        if "y_results.pt" not in fn:
            continue
        rel = os.path.relpath(dp, root)
        m = re.search(r"(.*)/cv(\d+)(?:/.*)?$", rel)
        if not m:
            continue
        group = m.group(1)
        d = torch.load(os.path.join(dp, "y_results.pt"), map_location="cpu")
        yt = d["y_true"].detach().cpu().numpy().astype(int).ravel()
        yp = d["y_pred"].detach().cpu().numpy().astype(int).ravel()
        groups[group]["acc"].append(100.0 * float((yt == yp).mean()))
        if group.startswith("class30/"):
            groups[group]["acc8"].append(100.0 * float((map8(yt) == map8(yp)).mean()))
    return groups


def agg(v):
    return [st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)]


def close(a, b, tol=5e-3):
    return abs(a - b) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/paper_tables.json")
    args = ap.parse_args()

    root = os.path.join(args.repo, "results/bacteria-id/finetuning")
    g = collect(root)
    table = {k: {"acc": agg(v["acc"]), **({"acc8": agg(v["acc8"])} if v["acc8"] else {})}
             for k, v in sorted(g.items())}

    # ---- validation against the two curated json files -----------------------
    sweep = {
        "byol": "class30/phys/byol/version_1/labelsweep_n%s",
        "mocov3": "class30/phys/mocov3/version_bs1024/labelsweep_n%s",
        "simclrv2": "class30/phys/simclrv2/version_bs1024/labelsweep_n%s",
        "no_pre": "class30/phys/no_pre/labelsweep_n%s",
        "supervised": "class30/phys/supervised/version_ho_adam_es10_aug/reuse_head/labelsweep_n%s",
    }
    matched = {
        "byol_phys": "class30/phys/byol/version_bs1024/matched_bs1024/no_aug",
        "byol_crop": "class30/crop/byol/version_bs1024/matched_bs1024/no_aug",
        "mocov3_phys": "class30/phys/mocov3/version_bs1024/matched_bs1024/no_aug",
        "mocov3_crop": "class30/crop/mocov3/version_bs1024/matched_bs1024/no_aug",
        "simclrv2_phys": "class30/phys/simclrv2/version_bs1024/matched_bs1024/no_aug",
        "simclrv2_crop": "class30/crop/simclrv2/version_bs1024/matched_bs1024/no_aug",
    }
    checks, bad = [], 0
    lp = os.path.join(args.repo, "results/label_efficiency.json")
    if os.path.exists(lp):
        ref = json.load(open(lp))
        for k, tmpl in sweep.items():
            for n in ("10", "20", "50", "100"):
                got = table.get(tmpl % n, {}).get("acc")
                exp = ref[k][n]
                ok = got is not None and close(got[0], exp[0]) and close(got[1], exp[1])
                bad += not ok
                checks.append((f"label_efficiency[{k}][{n}]", exp, got, ok))
    mp = os.path.join(args.repo, "results/matched_ssl_comparison.json")
    if os.path.exists(mp):
        ref = json.load(open(mp))
        for k, p in matched.items():
            got = table.get(p, {}).get("acc")
            exp = ref[k]
            ok = got is not None and close(got[0], exp[0]) and close(got[1], exp[1])
            bad += not ok
            checks.append((f"matched[{k}]", exp, got, ok))

    print("== validation against curated json ==")
    for name, exp, got, ok in checks:
        gs = "MISSING" if got is None else f"{got[0]:7.3f}+-{got[1]:.3f} n={got[2]}"
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:34s} json={exp[0]:7.3f}+-{exp[1]:.3f} n={exp[2]}   recomputed={gs}")
    print(f"  -> {len(checks)-bad}/{len(checks)} reproduced\n")

    print("== every run group (sample-level) ==")
    for k, v in table.items():
        a = v["acc"]
        extra = f"   acc8={v['acc8'][0]:6.2f}+-{v['acc8'][1]:.2f}" if "acc8" in v else ""
        print(f"  {a[0]:7.3f} +- {a[1]:5.3f}  n={a[2]:2d}  {k}{extra}")

    op = os.path.join(args.repo, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(table, open(op, "w"), indent=1, sort_keys=True)
    print(f"\nwrote {op}  ({len(table)} run groups)")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
