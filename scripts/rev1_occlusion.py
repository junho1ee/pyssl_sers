#!/usr/bin/env python
"""Spectral regions used by the fine-tuned classifiers (R2-R).

Occlusion sensitivity on the independent test subset. A window of 20 grid points
(about 40 cm-1) is replaced by the straight line joining its two neighbouring
points, which removes any band inside it while keeping the background, and the
drop in test accuracy is recorded for windows starting every 10 points; the last
window covers the four extrapolated points (1794-1800 cm-1). Each model is the
set of ten Table 4 classifiers (one per fold); drops are averaged over folds.

Default models: SimCLR v2 with physically motivated views (seed 0, epoch 3000),
supervised pretraining with a reinitialized classifier, and random
initialization (full pipeline). Pass --model name=<dir containing cv0..cv9>
to replace them.

Writes results/rev1_occlusion.json and results/rev1_occlusion.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets.resnet import ResNet  # noqa: E402

FT = "results/bacteria-id/finetuning/class30/phys"
DEFAULT_MODELS = [
    f"SimCLR v2={FT}/simclrv2/version_bs1024/rev1_s0_e2999/no_aug",
    "Supervised pretraining=/fs/ess/PCCF0025/leej101/pyssl_sers/" + FT + "/supervised/version_ho_adam_es10_aug/matched_table3_freshhead/no_aug",
    f"Random initialization={FT}/no_pre/rev1_s3/no_aug",
]
WIDTH, STRIDE, N = 20, 10, 696
GRID = np.linspace(400, 1800, N)


def classifier(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    n_out = [v.shape[0] for k, v in sd.items() if k.startswith("fc.") and k.endswith("weight")][-1]
    net = ResNet([100] * 6, [2] * 6, input_dim=N, in_channels=64, n_classes=n_out, encodeout="flatten")
    if "fc.0.weight" in sd:
        a, b = sd["fc.0.weight"].shape[1], sd["fc.0.weight"].shape[0]
        net.fc = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, n_out))
    else:
        net.fc = nn.Linear(sd["fc.weight"].shape[1], n_out)
    net.load_state_dict(sd)
    return net.eval()


def occlude(X, s):
    e = s + WIDTH
    Xo = X.copy()
    if s == 0:
        Xo[:, s:e] = X[:, [e]]
    elif e >= N:
        Xo[:, s:] = X[:, [s - 1]]
    else:
        w = np.linspace(0, 1, WIDTH + 2)[1:-1]
        Xo[:, s:e] = X[:, [s - 1]] * (1 - w) + X[:, [e]] * w
    return Xo


@torch.no_grad()
def accuracy(net, X, y, device, batch=1500):
    pred = [net(torch.tensor(X[i:i + batch, None, :], device=device)).argmax(1).cpu().numpy()
            for i in range(0, len(X), batch)]
    return 100.0 * float(np.mean(np.concatenate(pred) == y))


def plot(res, mean_spectrum, png):
    """Figure 5 at its printed size (0.9 of the 153 mm text width): text 8.5-10 pt,
    and line styles and markers that stay distinct in greyscale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8.5})
    starts = [int(round((w[0] - 400) / (1400 / (N - 1)))) for w in res["window_cm1"]]
    centers = [GRID[s] + (GRID[min(s + WIDTH, N) - 1] - GRID[s]) / 2 for s in starts]
    styles = [("-", "o"), ("--", "s"), (":", "^")]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax2 = ax.twinx()
    ax2.plot(GRID, mean_spectrum, color="0.82", lw=1)
    ax2.set_yticks([])
    for (name, r), (ls, mk) in zip(res["models"].items(), styles):
        ax.plot(centers, r["drop_mean"], ls=ls, marker=mk, ms=3, lw=1.1, label=name)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Accuracy drop\n(percentage points)")
    ax.set_xlim(400, 1800)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(png, dpi=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None, help="name=<dir with cv0..cv9>")
    ap.add_argument("--x", default="data/bacteria-id/preprocessed/X_test.npy")
    ap.add_argument("--y", default="data/bacteria-id/preprocessed/y_test.npy")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out", default="results/rev1_occlusion.json")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from an existing --out JSON")
    a = ap.parse_args()
    if a.plot_only:
        plot(json.loads(Path(a.out).read_text()), np.load(a.x).astype(np.float32).mean(0), Path(a.out).with_suffix(".png"))
        print("wrote", Path(a.out).with_suffix(".png"))
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(a.x).astype(np.float32)
    y = np.load(a.y).astype(int)
    starts = list(range(0, N - WIDTH + 1, STRIDE))
    if starts[-1] != N - WIDTH:
        starts.append(N - WIDTH)
    occluded = [occlude(X, s) for s in starts]
    res = {"width_points": WIDTH, "stride_points": STRIDE,
           "window_cm1": [[float(GRID[s]), float(GRID[min(s + WIDTH, N) - 1])] for s in starts],
           "models": {}}
    for spec in a.model or DEFAULT_MODELS:
        name, root = spec.split("=", 1)
        base, drops = [], []
        for f in range(a.folds):
            net = classifier(f"{root}/cv{f}/best.ckpt").to(device)
            b = accuracy(net, X, y, device)
            base.append(b)
            drops.append([b - accuracy(net, Xo, y, device) for Xo in occluded])
        drops = np.array(drops)
        res["models"][name] = {"base_accuracy": base, "drop_mean": drops.mean(0).tolist(),
                               "drop_sd": drops.std(0, ddof=1).tolist() if a.folds > 1 else None}
        top = np.argsort(-drops.mean(0))[:5]
        print(f"{name}: base {np.mean(base):.2f}; largest drops",
              [(f"{GRID[starts[i]]:.0f}-{GRID[min(starts[i] + WIDTH, N) - 1]:.0f}", round(float(drops.mean(0)[i]), 2)) for i in top],
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    plot(res, X.mean(0), Path(a.out).with_suffix(".png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
