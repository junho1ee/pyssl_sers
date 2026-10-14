"""COVID-19 transfer evaluation for the revision (R1-6, R2-J, R2-L, R2-M).

Uses the subject-grouped 70/30 partitions of covid_transfer_eval.py (same
split_indices, same seeds, so repeat r here is repeat r of Table 5) and saves,
for every repeat, the score of every test spectrum together with its subject,
so that spectrum- and subject-level metrics and confidence intervals can be
computed afterwards (scripts/rev1_covid_summary.py). One file per repeat:
results/covid_rev1/<task>/<tag>/rep<NN>.npz.

Differences from covid_transfer_eval.py, which produced Table 5:
  * every repeat is seeded (weights of a random initialization and the new
    classification head, batch order, augmentation draws);
  * the physically motivated augmentation receives (1, 696) arrays, the layout
    used on Bacteria-ID. covid_transfer_eval.py passed (696, 1), under which
    the baseline shift did nothing, the baseline wander moved only the first
    point, and the powerline term became a constant offset; and it applied that
    augmentation to every initialization except no_pre_noaug. Here
    augmentation is a separate switch;
  * models added: SVM on the full spectrum, ResNet on ANOVA-selected positions
    (other positions set to zero), a logistic-regression probe of frozen
    encoder features, and training on a fraction of the training subjects.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utils
from covid_transfer_eval import TASKS, build_backbone, load_dataset, split_indices

INITS = {  # name here -> name in covid_transfer_eval.build_backbone
    "random": "no_pre_noaug",
    "supervised": "supervised",
    "byol": "byol",
    "mocov3": "mocov3",
    "simclrv2": "simclrv2",
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subsample_subjects(idx, y, groups, fraction, seed):
    """Keep a fraction of the subjects of each class (at least one)."""
    if fraction >= 1.0:
        return idx
    rng = np.random.RandomState(10_000 + seed)
    keep = []
    for c in (0, 1):
        subj = np.unique(groups[idx][y[idx] == c])
        n = max(1, int(np.ceil(fraction * len(subj))))
        keep.append(rng.choice(subj, size=n, replace=False))
    keep = np.concatenate(keep)
    return idx[np.isin(groups[idx], keep)]


def anova_mask(X, y, idx, alpha=0.05):
    """Positions whose class means differ at level alpha, training spectra only."""
    p = np.array([f_oneway(X[idx][y[idx] == 1, j], X[idx][y[idx] == 0, j]).pvalue
                  for j in range(X.shape[1])])
    p = np.nan_to_num(p, nan=1.0)
    keep = np.where(p < alpha)[0]
    if keep.size < 2:
        keep = np.argsort(p)[:2]
    return keep


def run_svm(X, y, tr, te, features):
    keep = anova_mask(X, y, tr) if features == "anova" else np.arange(X.shape[1])
    sc = StandardScaler().fit(X[tr][:, keep])
    clf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(sc.transform(X[tr][:, keep]), y[tr])
    return clf.decision_function(sc.transform(X[te][:, keep])), {"n_features": keep.size}


def embed(net, X, device, batch=256):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.tensor(X[s:s + batch, None, :], dtype=torch.float32, device=device)
            out.append(net(xb).cpu().numpy())
    return np.concatenate(out)


def run_probe(X, y, tr, te, init, device, ckpt_root):
    if init == "raw":
        f_tr, f_te = X[tr], X[te]
    else:
        net, _ = build_backbone(INITS[init], ckpt_root)
        net.fc = nn.Identity()
        net = net.to(device)
        f_tr, f_te = embed(net, X[tr], device), embed(net, X[te], device)
    sc = StandardScaler().fit(f_tr)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(f_tr), y[tr])
    return clf.decision_function(sc.transform(f_te)), {"n_features": f_tr.shape[1]}


def run_resnet(X, y, tr, va, te, init, aug, features, seed, device, a):
    if features == "anova":
        mask = np.zeros(X.shape[1], dtype=np.float32)
        mask[anova_mask(X, y, np.concatenate([tr, va]))] = 1.0
    else:
        mask = np.ones(X.shape[1], dtype=np.float32)
    net, _ = build_backbone(INITS[init], a.ckpt_root)
    net = net.to(device)
    transform = utils.get_trans_from_augtype("phys") if aug else None

    xt = torch.tensor(X[:, None, :] * mask, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.5, 0.999))
    lossf = nn.CrossEntropyLoss()
    xva, yva = xt[va].to(device), yt[va].to(device)
    best, best_state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(a.epochs):
        net.train()
        perm = np.random.RandomState(seed + epoch).permutation(len(tr))
        for s in range(0, len(tr), a.batch_size):
            b = tr[perm[s:s + a.batch_size]]
            xb = X[b]
            if transform is not None:
                xb = np.stack([transform(xb[i][None, :]).numpy()[0] for i in range(len(b))])
            xb = torch.tensor(xb[:, None, :] * mask, dtype=torch.float32, device=device)
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
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(xt[te].to(device)), 1)[:, 1].cpu().numpy()
    return prob, {"n_features": int(mask.sum()), "epochs_run": epoch + 1, "best_val_loss": best}


def tag_of(a):
    if a.model == "svm":
        t = f"svm_{a.features}"
    elif a.model == "probe":
        t = f"probe_{a.init}"
    else:
        t = f"resnet_{a.init}_{'aug' if a.aug else 'noaug'}_{a.features}"
    return t if a.train_fraction >= 1.0 else f"{t}_f{a.train_fraction:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", required=True, choices=["svm", "resnet", "probe"])
    ap.add_argument("--init", default="random", choices=list(INITS) + ["raw"])
    ap.add_argument("--aug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--features", default="full", choices=["full", "anova"])
    ap.add_argument("--train_fraction", type=float, default=1.0)
    ap.add_argument("--repeats", nargs=2, type=int, default=[0, 50], metavar=("FIRST", "STOP"))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mat", default="data/covid/org/data.mat")
    ap.add_argument("--cache", default="data/covid/preprocessed/covid_696.npz")
    ap.add_argument("--ckpt_root", default="results/bacteria-id/pretraining/phys")
    ap.add_argument("--out", default="results/covid_rev1")
    a = ap.parse_args()

    data = load_dataset(a.mat, a.cache)
    ga, gb = TASKS[a.task]
    X = np.concatenate([data[ga][0], data[gb][0]]).astype(np.float32)
    y = np.concatenate([np.ones(len(data[ga][0]), int), np.zeros(len(data[gb][0]), int)])
    groups = np.concatenate([data[ga][1], data[gb][1] + 1000])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = tag_of(a)
    out_dir = os.path.join(a.out, a.task, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[task] {a.task} {tag} X={X.shape} subjects={len(np.unique(groups))} dev={device}")

    for seed in range(*a.repeats):
        path = os.path.join(out_dir, f"rep{seed:02d}.npz")
        if os.path.exists(path):
            print(f"  rep {seed}: exists, skipped")
            continue
        tr, va, te = split_indices(y, groups, "subject", seed)
        tr = subsample_subjects(tr, y, groups, a.train_fraction, seed)
        va = subsample_subjects(va, y, groups, a.train_fraction, seed)
        seed_all(seed)
        if a.model == "svm":
            score, meta = run_svm(X, y, np.concatenate([tr, va]), te, a.features)
            kind = "decision"
        elif a.model == "probe":
            score, meta = run_probe(X, y, np.concatenate([tr, va]), te, a.init, device, a.ckpt_root)
            kind = "decision"
        else:
            score, meta = run_resnet(X, y, tr, va, te, a.init, a.aug, a.features, seed, device, a)
            kind = "prob"
        np.savez(path, test_idx=te, subject=groups[te], y=y[te], score=score,
                 score_kind=kind, n_train=len(tr), n_val=len(va),
                 n_train_subjects=len(np.unique(groups[tr])), **meta)
        thr = 0.5 if kind == "prob" else 0.0
        acc = 100.0 * np.mean((score > thr).astype(int) == y[te])
        print(f"  rep {seed}: spectrum acc {acc:.1f} n_test={len(te)} {meta}")


if __name__ == "__main__":
    main()
