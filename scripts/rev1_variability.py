#!/usr/bin/env python
"""Spectral variability in Bacteria-ID against the size of the augmentations.

Answers R1-4 (are the physically motivated perturbations too large?) and R1-5
(intra- versus inter-isolate variability), and characterizes the four grid
points above 1792.4 cm-1 that are cubic extrapolations (R2-H).

All magnitudes are root-mean-square values over the 692 measured grid points of
the preprocessed model inputs (the four extrapolated points are excluded, and
analysed separately at the end):

* within isolate: a spectrum minus the mean spectrum of its isolate in the same
  subset, split by a Gaussian filter (sigma 25 points, about 50 cm-1) into a
  slow, baseline-like part and a fast, noise-like part;
* across cultures: the mean spectrum of an isolate in the test subset (separately
  cultured samples) minus its mean in the fine-tuning subset (same 2 s
  integration time), and minus its mean in the reference subset (1 s);
* between isolates: the distance from each isolate mean to the nearest mean of
  a different isolate, in the reference set;
* augmentations: the change one draw of a view makes to a spectrum, and the
  difference between the two views of a spectrum that an objective is trained
  to treat as the same.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402

N_MEASURED = 692  # grid points at or below 1792.4 cm-1
SIGMA = 25


def rms(a, axis=-1):
    return np.sqrt(np.mean(a**2, axis=axis))


def pct(v, q=(5, 50, 95)):
    return [round(float(x), 4) for x in np.percentile(v, q)]


def class_means(x, y):
    classes = np.unique(y)
    return classes, np.stack([x[y == c].mean(0) for c in classes])


def view_stats(x, augtypes, n=2000, seed=0):
    """RMS change from one view, and RMS difference between two views."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    out = {}
    for a in augtypes:
        t = utils.get_trans_from_augtype(a, p=1.0)
        random.seed(seed)
        np.random.seed(seed)
        one, two = [], []
        for i in idx:
            s = x[i][None, :].astype(np.float32)
            v1 = t(s).numpy()[0, :N_MEASURED]
            v2 = t(s).numpy()[0, :N_MEASURED]
            one.append(rms(v1 - s[0, :N_MEASURED]))
            two.append(rms(v1 - v2))
        out[a] = {"one_view_minus_input": pct(one), "view1_minus_view2": pct(two)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bacteria-id/preprocessed")
    ap.add_argument("--out", default="results/rev1_variability.json")
    args = ap.parse_args()

    load = lambda n: np.load(os.path.join(args.data, n))  # noqa: E731
    xr, yr = load("X_reference.npy"), load("y_reference.npy").astype(int)
    xt, yt = load("X_test.npy"), load("y_test.npy").astype(int)
    xf, yf = load("X_finetune.npy"), load("y_finetune.npy").astype(int)
    m_r, m_t = xr[:, :N_MEASURED], xt[:, :N_MEASURED]
    res = {}

    # Within isolate, reference set.
    cls, mu_r = class_means(m_r, yr)
    resid = m_r - mu_r[np.searchsorted(cls, yr)]
    slow = gaussian_filter1d(resid, SIGMA, axis=1, mode="nearest")
    fast = resid - slow
    res["within_isolate"] = {
        "total_rms": pct(rms(resid)),
        "slow_rms": pct(rms(slow)),
        "slow_max_abs": pct(np.abs(slow).max(1)),
        "fast_rms": pct(rms(fast)),
    }

    # Across cultures and between isolates.
    cls_t, mu_t = class_means(m_t, yt)
    cls_f, mu_f = class_means(xf[:, :N_MEASURED], yf)
    assert np.array_equal(cls, cls_t) and np.array_equal(cls, cls_f)
    d = np.sqrt(((mu_r[:, None, :] - mu_r[None, :, :]) ** 2).mean(-1))
    np.fill_diagonal(d, np.inf)
    nearest = d.min(1)
    res["nearest_other_isolate_rms"] = pct(nearest, (0, 50, 100))
    for name, mu in [("test_minus_finetune", mu_f), ("test_minus_reference", mu_r)]:
        shift = rms(mu_t - mu)
        res[f"across_cultures_{name}_rms"] = pct(shift, (0, 50, 100))
        res[f"isolates_where_{name}_exceeds_nearest_isolate"] = int((shift > nearest).sum())
    # share of the per-point variance that lies between isolate means
    tot = m_r.var(0)
    between = mu_r[np.searchsorted(cls, yr)].var(0)
    res["between_isolate_variance_share"] = pct(between / tot, (5, 50, 95))

    # Augmentations, on the same reference spectra.
    augtypes = [a for a in utils.AUGTYPES]
    res["augmentations"] = view_stats(xr, augtypes)

    # The four extrapolated points.
    tail_r, tail_t, tail_f = xr[:, N_MEASURED:], xt[:, N_MEASURED:], xf[:, N_MEASURED:]
    res["extrapolated"] = {
        "values_pct_1_50_99": pct(tail_r.ravel(), (1, 50, 99)),
        "share_of_spectra_outside_0_1": round(float(((tail_r < 0) | (tail_r > 1)).any(1).mean()), 4),
    }
    # Do the four points alone identify the isolate? Trained on the fine-tuning
    # set, scored on the test set; chance is 1/30.
    sc = StandardScaler().fit(tail_f)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(tail_f), yf)
    res["extrapolated"]["test_acc_from_4_points"] = round(float(lr.score(sc.transform(tail_t), yt)), 4)
    sc = StandardScaler().fit(xf[:, N_MEASURED - 4:N_MEASURED])
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(xf[:, N_MEASURED - 4:N_MEASURED]), yf)
    res["extrapolated"]["test_acc_from_last_4_measured_points"] = round(
        float(lr.score(sc.transform(xt[:, N_MEASURED - 4:N_MEASURED]), yt)), 4)

    print(json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
