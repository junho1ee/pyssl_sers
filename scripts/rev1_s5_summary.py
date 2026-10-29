#!/usr/bin/env python
"""Summarize revision step 5: Table 4 encoders on edge-held fine-tuning and test inputs.

For every physically motivated encoder of Table 4 (three seeds, at epoch 3000 and
at the validation-chosen epoch) and for supervised pretraining with and without
its reference classifier, compares test accuracy on edge-held inputs (the four
extrapolated points held at the last measured value) with the original inputs,
paired over the ten common folds. It then recomputes on edge-held inputs the
comparisons that section 3.1 states on the original inputs, against the
minimal-preprocessing random baseline, which does not depend on the edge points
(random initialization on the full pipeline: -0.20 +- 1.60, step 3).

Reads results/rev1_summary.json for the original per-fold values of the
self-supervised encoders. Writes results/rev1_s5_summary.json.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev1_summary import folds  # noqa: E402

KEYS = [("simclrv2", "phys/simclrv2/bs1024"), ("mocov3", "phys/mocov3/bs1024"),
        ("byol", "phys/byol/bs1024"), ("byol", "phys/byol/bs2048")]


def test_folds(ver, tag, pre="supervised"):
    r = folds(".", "phys", pre, ver, tag)
    return None if r is None else np.array(r[0])


def pair(a, b):
    d = np.asarray(a) - np.asarray(b)
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "n_higher": int((d > 0).sum())}


def main():
    seeds = json.load(open("results/rev1_summary.json"))["seeds"]
    out = {"encoders": {}, "comparisons": {}}
    edge = {}
    for pre, key in KEYS:
        for s in "012":
            e = seeds[key][s]
            for ep in sorted({3000, e["val_chosen_epoch"]}):
                orig = np.array(e["by_epoch"][str(ep)]["test"])
                eh = test_folds(e["version"], f"edgehold/rev1_s5_e{ep - 1}", pre)
                name = f"{key}/seed{s}/e{ep}"
                if eh is None:
                    out["encoders"][name] = "missing"
                    continue
                edge[(key, s, ep)] = eh
                out["encoders"][name] = {"original": float(orig.mean()), "edgehold": float(eh.mean()),
                                         "edgehold_sd": float(eh.std(ddof=1)), "edge_minus_original": pair(eh, orig),
                                         "val_chosen": ep == e["val_chosen_epoch"]}
    sup = {}
    for name, ver, tag_o, tag_e in [
        ("supervised_reuse", "version_ho_adam_es10_aug/reuse_head", "matched_table3", "edgehold/rev1_s5"),
        ("supervised_fresh", "version_ho_adam_es10_aug", "matched_table3_freshhead", "edgehold/rev1_s5_freshhead"),
    ]:
        o, eh = test_folds(ver, tag_o), test_folds(ver, tag_e)
        if o is None or eh is None:
            out["encoders"][name] = "missing"
            continue
        sup[name] = (o, eh)
        out["encoders"][name] = {"original": float(o.mean()), "edgehold": float(eh.mean()),
                                 "edgehold_sd": float(eh.std(ddof=1)), "edge_minus_original": pair(eh, o)}
    minimal = folds(".", "phys", "no_pre", "minimal", "")
    minimal = None if minimal is None else np.array(minimal[0])
    cmp = out["comparisons"]
    for s in "012":
        k = ("phys/simclrv2/bs1024", s, 3000)
        if k not in edge:
            continue
        orig_s = np.array(seeds[k[0]][s]["by_epoch"]["3000"]["test"])
        if "supervised_fresh" in sup:
            o, eh = sup["supervised_fresh"]
            cmp[f"simclr_minus_supervised_fresh/seed{s}"] = {"original": pair(orig_s, o), "edgehold": pair(edge[k], eh)}
        if "supervised_reuse" in sup:
            o, eh = sup["supervised_reuse"]
            cmp[f"simclr_minus_supervised_reuse/seed{s}"] = {"original": pair(orig_s, o), "edgehold": pair(edge[k], eh)}
        if minimal is not None:
            cmp[f"simclr_minus_minimal_random/seed{s}"] = {"original": pair(orig_s, minimal), "edgehold": pair(edge[k], minimal)}
        for key in ("phys/mocov3/bs1024", "phys/byol/bs1024", "phys/byol/bs2048"):
            for which in ("e3000", "val_chosen"):
                ep_s = 3000 if which == "e3000" else seeds["phys/simclrv2/bs1024"][s]["val_chosen_epoch"]
                ep_o = 3000 if which == "e3000" else seeds[key][s]["val_chosen_epoch"]
                a, b = edge.get(("phys/simclrv2/bs1024", s, ep_s)), edge.get((key, s, ep_o))
                if a is None or b is None:
                    continue
                oa = np.array(seeds["phys/simclrv2/bs1024"][s]["by_epoch"][str(ep_s)]["test"])
                ob = np.array(seeds[key][s]["by_epoch"][str(ep_o)]["test"])
                cmp[f"simclr_minus_{key.split('/', 1)[1].replace('/', '_')}/{which}/seed{s}"] = {
                    "original": pair(oa, ob), "edgehold": pair(a, b)}
    json.dump(out, open("results/rev1_s5_summary.json", "w"), indent=1)
    for k, v in out["encoders"].items():
        if v == "missing":
            print(f"{k:40s} missing")
        else:
            d = v["edge_minus_original"]
            print(f"{k:40s} {v['original']:6.2f} -> {v['edgehold']:6.2f}  ({d['mean']:+.2f} +- {d['sd']:.2f}, {d['n_higher']}/10)")
    for k, v in cmp.items():
        print(f"{k:55s} original {v['original']['mean']:+.2f} ({v['original']['n_higher']}/10)  edge-held {v['edgehold']['mean']:+.2f} ({v['edgehold']['n_higher']}/10)")


if __name__ == "__main__":
    main()
