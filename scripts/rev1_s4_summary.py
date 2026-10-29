#!/usr/bin/env python
"""Summarize revision step 4: temperature, momentum, and intensity scaling.

Each step 4 encoder (batch size 1024, seed 0, 1000 epochs) is compared at
epochs 500 and 1000 with the same epochs of the original seed 0 run of its
objective (temperature 0.5; momentum 0.999 for MoCo v3, 0.996 for BYOL), and
the intensity-scaling view with the complete physically motivated view, paired
over the ten Table 4 folds. Reads results/rev1_summary.json for the reference
runs; writes results/rev1_s4_summary.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev1_summary import folds  # noqa: E402

RUNS = [  # (label, view, objective, version, reference key)
    ("SimCLR v2, temperature 0.2", "phys", "simclrv2", "rev1_bs1024_seed0_e1000_t0p2", "phys/simclrv2/bs1024"),
    ("SimCLR v2, temperature 1.0", "phys", "simclrv2", "rev1_bs1024_seed0_e1000_t1p0", "phys/simclrv2/bs1024"),
    ("MoCo v3, temperature 0.2", "phys", "mocov3", "rev1_bs1024_seed0_e1000_t0p2", "phys/mocov3/bs1024"),
    ("MoCo v3, temperature 1.0", "phys", "mocov3", "rev1_bs1024_seed0_e1000_t1p0", "phys/mocov3/bs1024"),
    ("MoCo v3, momentum 0.99", "phys", "mocov3", "rev1_bs1024_seed0_e1000_m0p99", "phys/mocov3/bs1024"),
    ("BYOL, momentum 0.99", "phys", "byol", "rev1_bs1024_seed0_e1000_m0p99", "phys/byol/bs1024"),
    ("SimCLR v2, view + intensity scaling", "phys_plus_scale", "simclrv2", "rev1_bs1024_seed0_e1000", "phys/simclrv2/bs1024"),
]


def main():
    seeds = json.load(open("results/rev1_summary.json"))["seeds"]
    out = {}
    for label, aug, pre, ver, ref in RUNS:
        out[label] = {}
        for ep in (500, 1000):
            r = folds(".", aug, pre, ver, f"rev1_s4_e{ep - 1}")
            if r is None:
                out[label][f"e{ep}"] = "missing"
                continue
            t, v = np.array(r[0]), np.array(r[1])
            base_t = np.array(seeds[ref]["0"]["by_epoch"][str(ep)]["test"])
            base_v = np.array(seeds[ref]["0"]["by_epoch"][str(ep)]["val"])
            d = t - base_t
            out[label][f"e{ep}"] = {
                "test": float(t.mean()), "test_sd": float(t.std(ddof=1)), "val": float(v.mean()),
                "reference_test": float(base_t.mean()), "reference_val": float(base_v.mean()),
                "minus_reference": {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_higher": int((d > 0).sum())},
            }
    json.dump(out, open("results/rev1_s4_summary.json", "w"), indent=1)
    for label, res in out.items():
        cells = []
        for ep, x in res.items():
            if x == "missing":
                cells.append(f"{ep} missing")
            else:
                m = x["minus_reference"]
                cells.append(f"{ep} {x['test']:.2f} (ref {x['reference_test']:.2f}; {m['mean']:+.2f} +- {m['sd']:.2f}, {m['n_higher']}/10)")
        print(f"{label:38s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
