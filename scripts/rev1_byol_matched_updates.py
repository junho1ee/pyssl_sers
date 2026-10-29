#!/usr/bin/env python
"""BYOL at batch sizes 1024 and 2048 compared at matched numbers of updates (R2-D).

One epoch is 59 optimizer updates at batch size 1024 and 30 at 2048, so the
checkpoints at 500, 1000, and 1500 epochs of batch size 1024 are matched with
those at 1000, 2000, and 3000 epochs of batch size 2048. Differences (2048 minus
1024) are paired over the ten Table 4 folds, separately for each pretraining run.
Reads results/rev1_summary.json; writes results/rev1_byol_matched_updates.json.
"""

import json

import numpy as np


def main():
    d = json.load(open("results/rev1_summary.json"))["seeds"]
    a, b = d["phys/byol/bs1024"], d["phys/byol/bs2048"]
    out = {}
    for e1, e2 in [(500, 1000), (1000, 2000), (1500, 3000)]:
        rows = []
        for s in "012":
            diff = np.array(b[s]["by_epoch"][str(e2)]["test"]) - np.array(a[s]["by_epoch"][str(e1)]["test"])
            rows.append({"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)), "n_2048_higher": int((diff > 0).sum())})
        out[f"bs1024_e{e1}_vs_bs2048_e{e2}"] = rows
        print(f"bs1024 e{e1} vs bs2048 e{e2}:", "; ".join(f"seed{i} {r['mean']:+.2f} +- {r['sd']:.2f} ({r['n_2048_higher']}/10)" for i, r in enumerate(rows)))
    json.dump(out, open("results/rev1_byol_matched_updates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
