"""Transfer evaluation on the archived COVID-19 serum Raman dataset.

Runs the three binary tasks under two partitioning protocols:

  spectrum  - repeated random 70/30 splits over individual spectra, i.e. the
              protocol of the archived workflow. Spectra from one subject can
              land in both partitions.
  subject   - repeated random 70/30 splits over subjects. Every spectrum of a
              subject stays on one side of the split.

Subject identity is recovered from the column ordering documented in the
Figshare readme: each subject contributes three averaged spectra, except
subjects 16-21 of the suspected group, which contribute two.
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy as scp
import torch
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import f_oneway

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import preprocess.preprocess as pp  # noqa: E402
import utils  # noqa: E402
from nets.resnet import ResNet  # noqa: E402

WAVE_MIN, WAVE_MAX, N_POINTS = 400, 1800, 696
CLIP_RANGE = (400, 1800)
BASELINE_LAMBDA, BASELINE_P = 1e5, 5e-3
SAVGOL_WINDOW, SAVGOL_POLYORDER = 11, 3

GROUPS = ("covid", "suspected", "healthy")
MAT_KEY = {"covid": "raw_COVID", "suspected": "raw_Suspected", "healthy": "raw_Helthy"}
TASKS = {
    "covid_vs_suspected": ("covid", "suspected"),
    "covid_vs_healthy": ("covid", "healthy"),
    "suspected_vs_healthy": ("suspected", "healthy"),
}


def subject_ids(group, n_spectra):
    """Spectra-per-subject documented in the Figshare readme."""
    if group == "suspected":
        counts = [3] * 15 + [2] * 6 + [3] * 33
    else:
        counts = [3] * (n_spectra // 3)
    assert sum(counts) == n_spectra, (group, sum(counts), n_spectra)
    return np.repeat(np.arange(len(counts)), counts)


def preprocess_block(raman_shift, peaks):
    data = np.concatenate((raman_shift[None, :], peaks), axis=0)
    data = pp.clip_data_by_shift(data, CLIP_RANGE)
    data = pp.baseline_als(data.T, lam=BASELINE_LAMBDA, p=BASELINE_P).T
    shift, value = data[0, :], data[1:, :]
    value = scp.signal.savgol_filter(value, SAVGOL_WINDOW, SAVGOL_POLYORDER, axis=1)
    value = preprocessing.minmax_scale(value, axis=1)
    grid = np.linspace(WAVE_MIN, WAVE_MAX, N_POINTS)
    out = np.zeros((value.shape[0], N_POINTS), dtype=np.float32)
    for i in range(value.shape[0]):
        out[i] = scp.interpolate.interp1d(shift.ravel(), value[i].ravel(), kind="cubic")(grid)
    return out


def load_dataset(mat_path, cache):
    if os.path.exists(cache):
        z = np.load(cache)
        return {g: (z[g + "_X"], z[g + "_s"]) for g in GROUPS}
    mat = scp.io.loadmat(mat_path)
    shift = mat["wave_number"][0]
    out = {}
    for g in GROUPS:
        peaks = mat[MAT_KEY[g]].T
        X = preprocess_block(shift, peaks)
        s = subject_ids(g, X.shape[0])
        out[g] = (X, s)
        print(f"[data] {g}: {X.shape[0]} spectra, {len(np.unique(s))} subjects")
    np.savez(cache, **{f"{g}_X": out[g][0] for g in GROUPS},
             **{f"{g}_s": out[g][1] for g in GROUPS})
    return out


def build_backbone(pre, ckpt_root):
    net = ResNet([100] * 6, [2] * 6, input_dim=N_POINTS, in_channels=64,
                 n_classes=2, encodeout="flatten")
    if pre in ("no_pre_aug", "no_pre_noaug"):
        return net, 0
    path = {
        "supervised": f"{ckpt_root}/supervised/lightning_logs/version_ho_adam_es10_aug/checkpoints/last.ckpt",
        "byol": f"{ckpt_root}/byol/lightning_logs/version_1/checkpoints/last.ckpt",
        "mocov3": f"{ckpt_root}/mocov3/lightning_logs/version_bs1024/checkpoints/last.ckpt",
        "simclrv2": f"{ckpt_root}/simclrv2/lightning_logs/version_bs1024/checkpoints/last.ckpt",
    }[pre]
    sd = torch.load(path, map_location="cpu")["state_dict"]
    tgt = net.state_dict()
    loaded = {}
    for k, v in sd.items():
        kk = k
        for prefix in ("model.backbone.", "model.online_encoder.0.", "backbone.", "model."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
                break
        if kk in tgt and tgt[kk].shape == v.shape and not kk.startswith("fc."):
            loaded[kk] = v
    net.load_state_dict(loaded, strict=False)
    n_conv = len([k for k in loaded if "conv" in k or "bn" in k])
    if n_conv == 0:
        raise RuntimeError(f"no backbone weights matched from {path}")
    return net, len(loaded)


def split_indices(y, groups, protocol, seed, test_size=0.3, valid_size=0.1):
    idx = np.arange(len(y))
    if protocol == "subject":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_pool, te = next(gss.split(idx, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
        rel_tr, rel_va = next(gss2.split(tr_pool, y[tr_pool], groups[tr_pool]))
        return tr_pool[rel_tr], tr_pool[rel_va], te
    tr_pool, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    tr, va = train_test_split(tr_pool, test_size=valid_size, random_state=seed, stratify=y[tr_pool])
    return tr, va, te


def metrics(true, pred):
    tp = int(((pred == 1) & (true == 1)).sum()); tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum()); fn = int(((pred == 0) & (true == 1)).sum())
    return {"accuracy": 100.0 * (tp + tn) / max(1, len(true)),
            "sensitivity": 100.0 * tp / max(1, tp + fn),
            "specificity": 100.0 * tn / max(1, tn + fp)}


def run_svm(X, y, groups, protocol, seed, alpha=0.05):
    """SVM baseline reproducing the feature selection of the source study.

    Raman-shift positions whose class means differ at the given ANOVA level are
    selected on the training partition only, then an RBF SVM is fitted on those
    positions. Selection inside the loop keeps the hold-out partition unseen.
    """
    tr, va, te = split_indices(y, groups, protocol, seed)
    tr = np.concatenate([tr, va])
    p = np.array([f_oneway(X[tr][y[tr] == 1, j], X[tr][y[tr] == 0, j]).pvalue
                  for j in range(X.shape[1])])
    keep = np.where(np.nan_to_num(p, nan=1.0) < alpha)[0]
    if keep.size < 2:
        keep = np.argsort(np.nan_to_num(p, nan=1.0))[:2]
    sc = StandardScaler().fit(X[tr][:, keep])
    clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(sc.transform(X[tr][:, keep]), y[tr])
    out = metrics(y[te], clf.predict(sc.transform(X[te][:, keep])))
    out["n_test"] = int(len(te)); out["n_test_subjects"] = int(len(np.unique(groups[te])))
    out["n_features"] = int(keep.size)
    return out


def run_one(X, y, groups, pre, protocol, seed, device, epochs, patience, lr, batch_size):
    if pre == "svm":
        return run_svm(X, y, groups, protocol, seed)
    tr, va, te = split_indices(y, groups, protocol, seed)
    net, _ = build_backbone(pre, ARGS.ckpt_root)
    net = net.to(device)
    transform = utils.get_trans_from_augtype("phys") if pre != "no_pre_noaug" else None

    xt = torch.tensor(X[:, None, :], dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    lossf = nn.CrossEntropyLoss()

    best, best_state, bad = float("inf"), None, 0
    xva, yva = xt[va].to(device), yt[va].to(device)
    for _ in range(epochs):
        net.train()
        perm = np.random.RandomState(seed + _).permutation(len(tr))
        for s in range(0, len(tr), batch_size):
            b = tr[perm[s:s + batch_size]]
            xb = X[b]
            if transform is not None:
                xb = np.stack([np.asarray(transform(xb[i][:, None])).squeeze() for i in range(len(b))])
            xb = torch.tensor(np.asarray(xb)[:, None, :], dtype=torch.float32).to(device)
            opt.zero_grad()
            loss = lossf(net(xb), yt[b].to(device))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xva), yva).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xt[te].to(device)).argmax(1).cpu().numpy()
    true = y[te]
    tp = int(((pred == 1) & (true == 1)).sum()); tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum()); fn = int(((pred == 0) & (true == 1)).sum())
    return {
        "accuracy": 100.0 * (tp + tn) / max(1, len(true)),
        "sensitivity": 100.0 * tp / max(1, tp + fn),
        "specificity": 100.0 * tn / max(1, tn + fp),
        "n_test": int(len(true)),
        "n_test_subjects": int(len(np.unique(groups[te]))),
    }


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", default="data/covid/org/data.mat")
    parser.add_argument("--cache", default="data/covid/preprocessed/covid_696.npz")
    parser.add_argument("--ckpt_root", default="results/bacteria-id/pretraining/phys")
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--pre", required=True,
                        choices=["no_pre_noaug", "no_pre_aug", "supervised", "byol",
                                 "mocov3", "simclrv2", "svm"])
    parser.add_argument("--protocols", nargs="+", default=["spectrum", "subject"])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="results/covid")
    ARGS = parser.parse_args()

    os.makedirs(os.path.dirname(ARGS.cache), exist_ok=True)
    os.makedirs(ARGS.out, exist_ok=True)
    data = load_dataset(ARGS.mat, ARGS.cache)

    a, b = TASKS[ARGS.task]
    Xa, sa = data[a]
    Xb, sb = data[b]
    X = np.concatenate([Xa, Xb], 0)
    y = np.concatenate([np.ones(len(Xa), int), np.zeros(len(Xb), int)])
    groups = np.concatenate([sa, sb + 1000])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[task] {ARGS.task} pre={ARGS.pre} X={X.shape} "
          f"pos={int(y.sum())} neg={int((1-y).sum())} subjects={len(np.unique(groups))} dev={device}")

    result = {}
    for protocol in ARGS.protocols:
        runs = [run_one(X, y, groups, ARGS.pre, protocol, seed, device,
                        ARGS.epochs, ARGS.patience, ARGS.lr, ARGS.batch_size)
                for seed in range(ARGS.repeats)]
        agg = {}
        for m in ("accuracy", "sensitivity", "specificity"):
            v = np.array([r[m] for r in runs])
            agg[m] = [float(v.mean()), float(v.std(ddof=1))]
        agg["n_repeats"] = len(runs)
        agg["mean_test_spectra"] = float(np.mean([r["n_test"] for r in runs]))
        agg["mean_test_subjects"] = float(np.mean([r["n_test_subjects"] for r in runs]))
        result[protocol] = agg
        print(f"  [{protocol}] acc {agg['accuracy'][0]:.2f}+-{agg['accuracy'][1]:.2f}  "
              f"sens {agg['sensitivity'][0]:.2f}+-{agg['sensitivity'][1]:.2f}  "
              f"spec {agg['specificity'][0]:.2f}+-{agg['specificity'][1]:.2f}")

    path = os.path.join(ARGS.out, f"{ARGS.task}__{ARGS.pre}.json")
    json.dump(result, open(path, "w"), indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
