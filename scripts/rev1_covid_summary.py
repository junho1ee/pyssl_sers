#!/usr/bin/env python
"""Spectrum- and subject-level COVID-19 results with confidence intervals.

Reads results/covid_rev1/<task>/<tag>/rep<NN>.npz (covid_rev1_eval.py). For each
repeat, metrics are computed per spectrum and per subject; a subject's score is
the mean score of its spectra (probability for ResNets, decision value for the
SVM and the logistic-regression probes), thresholded at 0.5 or 0 respectively.

Across the 50 repeats the script reports the mean, the standard deviation, the
2.5-97.5th percentiles, and a 95% confidence interval for the mean corrected for
the overlap between random training partitions (Nadeau and Bengio, Machine
Learning 52, 239-281, 2003): mean +- t(0.975, J-1) * sqrt((1/J + n_test/n_train)
* s^2). Every model is scored on the same partitions, so differences from the
SVM baseline are paired by repeat and tested with the same correction; runs on
a fraction of the training subjects (tag suffix _f<fraction>) are compared with
the SVM trained on the same fraction. With a small training partition the
correction term n_test/n_train is large and so are the intervals.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

TASKS = ["covid_vs_suspected", "covid_vs_healthy", "suspected_vs_healthy"]
METRICS = ["accuracy", "sensitivity", "specificity", "auc"]
REFERENCE = "svm_anova"
RANDOM = "resnet_random_noaug_full"  # second reference for the ResNets
# Table 5 as published (spectrum level, subject-grouped partitions, unseeded).
PUBLISHED = {
    "svm_anova": [90.2, 91.6, 75.6],
    "resnet_random_noaug_full": [85.1, 88.3, 68.2],
    "resnet_random_aug_full": [80.8, 89.3, 61.5],
    "resnet_supervised_aug_full": [83.2, 90.3, 64.6],
    "resnet_byol_aug_full": [85.9, 91.8, 66.7],
    "resnet_mocov3_aug_full": [86.5, 91.4, 65.8],
    "resnet_simclrv2_aug_full": [85.1, 90.9, 67.6],
}


def scores(y, s, thr):
    pred = (s > thr).astype(int)
    tp = np.sum((pred == 1) & (y == 1)); tn = np.sum((pred == 0) & (y == 0))
    out = {"accuracy": 100.0 * np.mean(pred == y),
           "sensitivity": 100.0 * tp / max(1, np.sum(y == 1)),
           "specificity": 100.0 * tn / max(1, np.sum(y == 0)),
           "auc": 100.0 * roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan}
    return out


def per_repeat(path):
    z = np.load(path)
    y, s, subj = z["y"], z["score"], z["subject"]
    thr = 0.5 if str(z["score_kind"]) == "prob" else 0.0
    spec = scores(y, s, thr)
    ids = np.unique(subj)
    ys = np.array([y[subj == i][0] for i in ids])
    ss = np.array([s[subj == i].mean() for i in ids])
    subj_m = scores(ys, ss, thr)
    ratio = len(y) / (int(z["n_train"]) + int(z["n_val"]))
    return spec, subj_m, ratio, len(ids)


def corrected_ci(v, ratio):
    v = np.asarray(v, float)
    j = len(v)
    se = np.sqrt((1.0 / j + ratio) * v.var(ddof=1))
    h = stats.t.ppf(0.975, j - 1) * se
    return [float(v.mean() - h), float(v.mean() + h)], se


def describe(v, ratio):
    v = np.asarray(v, float)
    ci, _ = corrected_ci(v, ratio)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p2.5_97.5": [float(x) for x in np.percentile(v, [2.5, 97.5])],
            "ci95_corrected": ci, "n": len(v)}


def paired(a, b, ratio):
    d = np.asarray(a, float) - np.asarray(b, float)
    ci, se = corrected_ci(d, ratio)
    t = d.mean() / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), len(d) - 1) if se > 0 else np.nan
    return {"mean": float(d.mean()), "ci95_corrected": ci, "p_corrected": float(p),
            "n_better": int((d > 0).sum()), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/covid_rev1")
    ap.add_argument("--out", default="results/rev1_covid_summary.json")
    ap.add_argument("--n_repeats", type=int, default=50)
    a = ap.parse_args()

    out = {}
    for ti, task in enumerate(TASKS):
        out[task] = {}
        raw = {}
        for d in sorted(glob.glob(os.path.join(a.root, task, "*"))):
            files = sorted(glob.glob(os.path.join(d, "rep*.npz")))
            if len(files) != a.n_repeats:
                print(f"{task} {os.path.basename(d)}: {len(files)}/{a.n_repeats} repeats, skipped")
                continue
            reps = [per_repeat(f) for f in files]
            raw[os.path.basename(d)] = reps
        print(f"\n{task}: accuracy (%) mean ± SD [corrected 95% CI]; AUC (%)")
        print(f"  {'model':34s} {'spectrum level':>30s} {'subject level':>30s} {'subj AUC':>8s}  "
              f"{'subj acc minus SVM (same fraction)':>36s} {'minus random-init ResNet, no aug':>36s}")
        for tag, reps in raw.items():
            ratio = float(np.mean([r[2] for r in reps]))
            rec = {"n_test_subjects_median": float(np.median([r[3] for r in reps]))}
            for level, k in (("spectrum", 0), ("subject", 1)):
                rec[level] = {m: describe([r[k][m] for r in reps], ratio) for m in METRICS}
            m = re.search(r"_f([0-9.]+)$", tag)  # fraction suffix only, not "_full"
            frac = "_f" + m.group(1) if m else ""
            for key, ref in (("minus_reference", REFERENCE + frac), ("minus_random", RANDOM + frac)):
                if key == "minus_random" and not tag.startswith("resnet_"):
                    continue
                if tag != ref and ref in raw:
                    rec[key.replace("minus", "ref")] = ref
                    rec[key] = {
                        level: paired([r[k]["accuracy"] for r in reps],
                                      [r[k]["accuracy"] for r in raw[ref]], ratio)
                        for level, k in (("spectrum", 0), ("subject", 1))}
            if tag in PUBLISHED:
                rec["published_spectrum_accuracy"] = PUBLISHED[tag][ti]
            out[task][tag] = rec
            sp, su = rec["spectrum"]["accuracy"], rec["subject"]["accuracy"]
            fmt = lambda x: f"{x['mean']:5.1f} ± {x['sd']:4.1f} [{x['ci95_corrected'][0]:5.1f}, {x['ci95_corrected'][1]:5.1f}]"  # noqa: E731
            diffs = []
            for key in ("minus_reference", "minus_random"):
                p = rec.get(key, {}).get("subject")
                diffs.append(f"{p['mean']:+5.1f} [{p['ci95_corrected'][0]:+5.1f}, {p['ci95_corrected'][1]:+5.1f}] p={p['p_corrected']:.2f}" if p else "")
            pub = f"  (published {rec['published_spectrum_accuracy']})" if "published_spectrum_accuracy" in rec else ""
            print(f"  {tag:34s} {fmt(sp):>30s} {fmt(su):>30s} {rec['subject']['auc']['mean']:8.1f}  {diffs[0]:>36s} {diffs[1]:>36s}{pub}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
