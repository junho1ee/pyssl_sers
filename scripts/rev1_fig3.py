"""Figure 3 for the revision (R1-3): binary MRSA/MSSA on the independent test subset.

Same model and predictions as the published Figure 3 (supervised-pretrained
ResNet with augmentation, five random fine-tuning splits, archived class2
checkpoints), with three changes requested by Referee 1:
  * cells and colour bar are labelled as row percentages of the true class;
  * every value carries a standard deviation over resampled test sets
    (class-stratified bootstrap of the 500 test spectra, the same resample
    applied to all five fine-tuned models), not over repeated runs;
  * the positive class of the ROC curve (MSSA, label 1) is stated.
The SD over the five fine-tuning splits is also written to the metrics file.
Test spectra of one isolate are correlated, and with two MRSA and three MSSA
isolates an isolate-level bootstrap is not possible, so the bootstrap SD is a
lower bound on the variability across independently cultured test sets;
per-isolate accuracy is reported for that reason.

Usage: python scripts/rev1_fig3.py --class2-root <.../ho_pre_adam_es10_last_ft_ho_split_adam_class2>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets.resnet import ResNet  # noqa: E402

LABELS = ["MRSA", "MSSA"]  # label 0, label 1 (positive class of the ROC curve)
ISOLATES = {16: "MRSA 1 (isogenic)", 17: "MRSA 2", 14: "MSSA 1", 18: "MSSA 2", 15: "MSSA 3"}


def fold_probabilities(root, x, folds=5):
    xt = torch.tensor(x[:, None, :], dtype=torch.float32)
    out = []
    for fold in range(folds):
        m = ResNet(hidden_sizes=[100] * 6, num_blocks=[2] * 6, input_dim=x.shape[1],
                   in_channels=64, n_classes=2)
        f = m.fc.in_features
        m.fc = torch.nn.Sequential(torch.nn.Linear(f, f), torch.nn.ReLU(), torch.nn.Linear(f, 2))
        sd = torch.load(Path(root) / f"cv{fold}" / "best.ckpt", map_location="cpu",
                        weights_only=False)["state_dict"]
        m.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
        m.eval()
        with torch.no_grad():
            lg = torch.cat([m(xt[i:i + 256]) for i in range(0, len(xt), 256)])
        out.append(torch.softmax(lg, 1)[:, 1].numpy())
    return np.stack(out)  # (folds, n_test), probability of MSSA


def metrics(y, p):
    """Pooled over folds: row-percent confusion matrix, accuracy, AUC."""
    yy = np.tile(y, p.shape[0])
    pp = p.ravel()
    pred = (pp > 0.5).astype(int)
    cm = confusion_matrix(yy, pred, labels=[0, 1]).astype(float)
    row = 100.0 * cm / cm.sum(1, keepdims=True)
    return row, 100.0 * np.mean(pred == yy), roc_auc_score(yy, pp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class2-root", required=True)
    ap.add_argument("--data-dir", default="data/bacteria-id/preprocessed")
    ap.add_argument("--out", default="figures/fig3.png")
    ap.add_argument("--json", default="results/rev1_fig3.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    x = np.load(f"{a.data_dir}/X_test_binary.npy")
    y = np.load(f"{a.data_dir}/y_test_binary.npy").astype(int)
    p = fold_probabilities(a.class2_root, x)

    # Archived predictions must match the recomputed ones.
    for fold in range(p.shape[0]):
        r = torch.load(Path(a.class2_root) / f"cv{fold}" / "y_results.pt", map_location="cpu")
        assert np.array_equal(r["y_true"].numpy().astype(int).ravel(), y)
        assert np.array_equal(r["y_pred"].numpy().astype(int).ravel(), (p[fold] > 0.5).astype(int)), fold

    # Isolate of every binary test spectrum, by exact match to the 30-isolate test subset.
    x30 = np.load(f"{a.data_dir}/X_test.npy")
    y30 = np.load(f"{a.data_dir}/y_test.npy").astype(int)
    pos = {row.tobytes(): i for i, row in enumerate(x30)}
    iso = np.array([y30[pos[row.tobytes()]] for row in x])
    assert set(iso) == set(ISOLATES)

    row, acc, auc = metrics(y, p)
    rng = np.random.RandomState(0)
    idx_c = [np.where(y == c)[0] for c in (0, 1)]
    boot = {"row": [], "acc": [], "auc": []}
    for _ in range(a.n_boot):
        b = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx_c])
        r_, a_, u_ = metrics(y[b], p[:, b])
        boot["row"].append(r_); boot["acc"].append(a_); boot["auc"].append(u_)
    row_sd = np.std(boot["row"], axis=0, ddof=1)
    acc_sd, auc_sd = np.std(boot["acc"], ddof=1), np.std(boot["auc"], ddof=1)
    per_fold_acc = [100.0 * np.mean((p[k] > 0.5) == y) for k in range(p.shape[0])]
    per_fold_auc = [roc_auc_score(y, p[k]) for k in range(p.shape[0])]
    per_iso = {ISOLATES[i]: 100.0 * float(np.mean((p[:, iso == i] > 0.5).astype(int) == (1 if "MSSA" in ISOLATES[i] else 0)))
               for i in ISOLATES}

    res = {"n_test": int(len(y)), "n_per_class": np.bincount(y).tolist(), "folds": int(p.shape[0]),
           "n_boot": a.n_boot, "positive_class": "MSSA",
           "confusion_row_percent": row.tolist(), "confusion_row_percent_boot_sd": row_sd.tolist(),
           "pooled_accuracy": acc, "pooled_accuracy_boot_sd": float(acc_sd),
           "auc": float(auc), "auc_boot_sd": float(auc_sd),
           "accuracy_per_fold": per_fold_acc, "accuracy_sd_over_folds": float(np.std(per_fold_acc, ddof=1)),
           "auc_per_fold": per_fold_auc, "auc_sd_over_folds": float(np.std(per_fold_auc, ddof=1)),
           "accuracy_per_isolate": per_iso}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ()}, indent=1))

    # Figure: layout and styling follow generate_bacteria_figures.plot_fig3.
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 2.4})
    fig = plt.figure(figsize=(13.2, 6.0))
    ax1 = fig.add_axes([0.08, 0.17, 0.37, 0.62])
    im = ax1.imshow(row, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{row[i, j]:.1f}%\n± {row_sd[i, j]:.1f}", ha="center", va="center",
                     fontsize=19, color="white" if row[i, j] >= 55 else "#222222")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(LABELS); ax1.set_yticklabels(LABELS, rotation=90, va="center")
    ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                    width=2.0, length=5, labelsize=22)
    ax1.set_xlabel("Predicted class", fontsize=20, labelpad=10)
    ax1.xaxis.set_label_position("top")
    ax1.set_ylabel("True class", fontsize=20, labelpad=10)
    for s in ax1.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.05)
    cbar.ax.tick_params(labelsize=16, width=2.0, length=5)
    cbar.set_label("Spectra of the true class (%)", fontsize=17)

    fpr, tpr, _ = roc_curve(np.tile(y, p.shape[0]), p.ravel(), pos_label=1)
    ax2 = fig.add_axes([0.66, 0.14, 0.34, 0.72])
    ax2.plot(fpr, tpr, color="navy", lw=3.2, label=f"AUC = {auc:.3f} ± {auc_sd:.3f}")
    ax2.plot([0, 1], [0, 1], color="black", lw=2.8, linestyle="--")
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("False Positive Rate", fontsize=24, labelpad=12)
    ax2.set_ylabel("True Positive Rate", fontsize=24, labelpad=12)
    ax2.tick_params(axis="both", labelsize=22, width=2.2, length=7)
    ax2.legend(loc="lower right", fontsize=17, frameon=False, title="Positive class: MSSA",
               title_fontsize=17)
    for s in ax2.spines.values():
        s.set_linewidth(2.4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {a.out}, {a.json}")


if __name__ == "__main__":
    main()
