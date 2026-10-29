#!/usr/bin/env python
"""Isolate structure of the frozen representations (R1-5).

Every encoder is used frozen, as in the COVID-19 probes: the 2200-dimensional
flattened backbone output of each spectrum. The measured input itself (the 692
measured grid points) is analysed in the same way for reference. Distances are
cosine distances; for the input they are also computed as Euclidean distances,
which reproduces the input-space comparison of section 2.2.2.

For every space, with isolate means taken within one subset:
* shift beyond separation: the number of the 30 isolates whose test-subset mean
  lies farther from its own mean in the fine-tuning (or reference) subset than
  the nearest other isolate mean lies in the reference subset (section 2.2.2);
* nearest-mean identity: the number of isolates whose test-subset mean is
  closest to the mean of the same isolate in the fine-tuning (or reference)
  subset;
* nearest-mean accuracy: each test spectrum assigned to the closest
  fine-tuning-subset isolate mean, with no training;
* 10-nearest-neighbour accuracy against the fine-tuning subset;
* spread ratio: median distance of a test spectrum to its own isolate mean in
  the test subset, divided by the median distance from an isolate mean to the
  nearest other isolate mean there;
* S. aureus: per MRSA/MSSA isolate, the share of its test spectra closer to the
  fine-tuning-subset mean of its own class than to that of the other class.

The reference subset is subsampled to 200 spectra per isolate (seed 0).
Writes results/rev1_representation.json.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets.resnet import ResNet  # noqa: E402

N_MEASURED = 692
STAPH = {16: ("MRSA 1", 0), 17: ("MRSA 2", 0), 14: ("MSSA 1", 1), 18: ("MSSA 2", 1), 15: ("MSSA 3", 1)}
PRE = "results/bacteria-id/pretraining"


def encoder_paths():
    enc = {"supervised": f"{PRE}/phys/supervised/lightning_logs/version_ho_adam_es10_aug/checkpoints/last.ckpt"}
    for aug, pre, vers in [
        ("phys", "simclrv2", ["version_bs1024", "rev1_bs1024_seed1", "rev1_bs1024_seed2"]),
        ("phys", "mocov3", ["version_bs1024", "rev1_bs1024_seed1", "rev1_bs1024_seed2"]),
        ("phys", "byol", ["version_bs1024", "rev1_bs1024_seed1", "rev1_bs1024_seed2"]),
        ("phys", "byol_bs2048", ["version_1", "rev1_bs2048_seed1", "rev1_bs2048_seed2"]),
        ("crop", "simclrv2", ["version_bs1024", "rev1_bs1024_seed1", "rev1_bs1024_seed2"]),
    ]:
        d = pre.replace("_bs2048", "")
        for seed, v in enumerate(vers):
            enc[f"{aug}/{pre}/seed{seed}"] = f"{PRE}/{aug}/{d}/lightning_logs/{v}/checkpoints/epoch=2999-loss=*.ckpt"
    return enc


def resolve(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {hits}")
    return hits[0]


def backbone(path):
    net = ResNet([100] * 6, [2] * 6, input_dim=696, in_channels=64, n_classes=30, encodeout="flatten")
    net.fc = nn.Identity()
    if path is None:
        return net
    sd = torch.load(path, map_location="cpu")["state_dict"]
    tgt = net.state_dict()
    loaded = {}
    for k, v in sd.items():
        kk = k
        for prefix in ("model.backbone.", "model.online_encoder.0.", "backbone.", "model."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
                break
        if kk in tgt and tgt[kk].shape == v.shape:
            loaded[kk] = v
    missing = set(tgt) - set(loaded)
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} backbone tensors not found, e.g. {sorted(missing)[:3]}")
    net.load_state_dict(loaded)
    return net


@torch.no_grad()
def embed(net, X, device, batch=1024):
    net = net.to(device).eval()
    out = [net(torch.tensor(X[s:s + batch, None, :], dtype=torch.float32, device=device)).cpu().numpy()
           for s in range(0, len(X), batch)]
    return np.concatenate(out).astype(np.float64)


def unit(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def pdist(A, B, metric):
    if metric == "cosine":
        return 1.0 - unit(A) @ unit(B).T
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(sq, 0) / A.shape[1])  # RMS difference, as in section 2.2.2


def means(F, y):
    return np.stack([F[y == c].mean(0) for c in range(30)])


def analyse(F, y, metric):
    out = {}
    Ct = means(F["test"], y["test"])
    Cr = means(F["reference"], y["reference"])
    D_rr = pdist(Cr, Cr, metric)
    np.fill_diagonal(D_rr, np.inf)
    for sub in ("finetune", "reference"):
        Cs = means(F[sub], y[sub])
        D_ts = pdist(Ct, Cs, metric)
        # separation measured between reference-subset means, as in section 2.2.2
        out[f"shift_beyond_separation_{sub}"] = int(np.sum(np.diag(D_ts) > D_rr.min(1)))
        out[f"nearest_mean_identity_{sub}"] = int(np.sum(D_ts.argmin(1) == np.arange(30)))
    Cf = means(F["finetune"], y["finetune"])
    out["nearest_mean_accuracy"] = 100.0 * float(np.mean(pdist(F["test"], Cf, metric).argmin(1) == y["test"]))
    S = unit(F["test"]) @ unit(F["finetune"]).T
    nn10 = np.argsort(-S, axis=1)[:, :10]
    votes = y["finetune"][nn10]
    pred = np.array([np.bincount(v, minlength=30).argmax() for v in votes])
    out["knn10_accuracy"] = 100.0 * float(np.mean(pred == y["test"]))
    within = np.median(pdist(F["test"], Ct, metric)[np.arange(len(y["test"])), y["test"]])
    D_tt = pdist(Ct, Ct, metric)
    np.fill_diagonal(D_tt, np.inf)
    out["spread_ratio"] = float(within / np.median(D_tt.min(1)))
    staph_ft = np.isin(y["finetune"], list(STAPH))
    cls_ft = np.array([STAPH[c][1] for c in y["finetune"][staph_ft]])
    M = np.stack([F["finetune"][staph_ft][cls_ft == k].mean(0) for k in (0, 1)])
    per = {}
    for iso, (name, k) in STAPH.items():
        Di = pdist(F["test"][y["test"] == iso], M, metric)
        per[name] = 100.0 * float(np.mean(Di.argmin(1) == k))
    out["staph_own_class_share"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bacteria-id/preprocessed")
    ap.add_argument("--out", default="results/rev1_representation.json")
    ap.add_argument("--n_ref_per_isolate", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these encoder names")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = {}, {}
    for sub in ("reference", "finetune", "test"):
        X[sub] = np.load(f"{a.data}/X_{sub}.npy").astype(np.float32)
        y[sub] = np.load(f"{a.data}/y_{sub}.npy").astype(int)
    rng = np.random.RandomState(0)
    keep = np.concatenate([rng.choice(np.flatnonzero(y["reference"] == c), a.n_ref_per_isolate, replace=False)
                           for c in range(30)])
    X["reference"], y["reference"] = X["reference"][keep], y["reference"][keep]

    res = {"n_reference": int(len(keep)), "n_finetune": int(len(y["finetune"])), "n_test": int(len(y["test"]))}
    inp = {s: X[s][:, :N_MEASURED].astype(np.float64) for s in X}
    res["input/euclidean"] = analyse(inp, y, "euclidean")
    res["input/cosine"] = analyse(inp, y, "cosine")
    torch.manual_seed(0)
    spaces = {"random": None, **encoder_paths()}
    for name, path in spaces.items():
        if a.only and name not in a.only:
            continue
        net = backbone(None if path is None else resolve(path))
        F = {s: embed(net, X[s], device) for s in X}
        res[name] = analyse(F, y, "cosine")
        print(name, {k: v for k, v in res[name].items() if k != "staph_own_class_share"}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
