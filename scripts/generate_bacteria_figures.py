import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from sklearn.metrics import auc, confusion_matrix, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.resnet import ResNet


ORDER = [16, 17, 14, 18, 15, 20, 21, 24, 23, 26, 27, 28, 29, 25, 6, 7, 5, 3, 4, 9, 10, 2, 8, 11, 22, 19, 12, 13, 0, 1]
STRAINS = {
    0: "C. albicans",
    1: "C. glabrata",
    2: "K. aerogenes",
    3: "E. coli 1",
    4: "E. coli 2",
    5: "E. faecium",
    6: "E. faecalis 1",
    7: "E. faecalis 2",
    8: "E. cloacae",
    9: "K. pneumoniae 1",
    10: "K. pneumoniae 2",
    11: "P. mirabilis",
    12: "P. aeruginosa 1",
    13: "P. aeruginosa 2",
    14: "MSSA 1",
    15: "MSSA 3",
    16: "MRSA 1 (isogenic)",
    17: "MRSA 2",
    18: "MSSA 2",
    19: "S. enterica",
    20: "S. epidermidis",
    21: "S. lugdunensis",
    22: "S. marcescens",
    23: "S. pneumoniae 2",
    24: "S. pneumoniae 1",
    25: "S. sanguinis",
    26: "Group A Strep.",
    27: "Group B Strep.",
    28: "Group C Strep.",
    29: "Group G Strep.",
}
ATCC_GROUPINGS = {
    3: 0,
    4: 0,
    9: 0,
    10: 0,
    2: 0,
    8: 0,
    11: 0,
    22: 0,
    12: 2,
    13: 2,
    14: 3,
    18: 3,
    15: 3,
    20: 3,
    21: 3,
    16: 3,
    17: 3,
    23: 4,
    24: 4,
    26: 5,
    27: 5,
    28: 5,
    29: 5,
    25: 5,
    6: 5,
    7: 5,
    5: 6,
    19: 1,
    0: 7,
    1: 7,
}
AB_ORDER = [3, 4, 5, 6, 0, 1, 2, 7]
ANTIBIOTICS = {
    0: "Meropenem",
    1: "Ciprofloxacin",
    2: "TZP",
    3: "Vancomycin",
    4: "Ceftriaxone",
    5: "Penicillin",
    6: "Daptomycin",
    7: "Caspofungin",
}


def load_cv_predictions(root, folds=5):
    y_true = []
    y_pred = []
    for fold in range(folds):
        result_path = Path(root) / f"cv{fold}" / "y_results.pt"
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        result = torch.load(result_path, map_location="cpu")
        y_true.append(result["y_true"].detach().cpu().numpy().astype(int))
        y_pred.append(result["y_pred"].detach().cpu().numpy().astype(int))
    return np.concatenate(y_true), np.concatenate(y_pred)


def load_binary_scores_from_checkpoints(root, x_test_fn, y_test_fn, folds=5, ckpt_name="best.ckpt"):
    x_test = np.load(x_test_fn)
    y_test = np.load(y_test_fn).astype(int)
    x_tensor = torch.tensor(x_test[:, None, :], dtype=torch.float32)

    all_true = []
    all_score = []
    all_pred = []
    for fold in range(folds):
        ckpt_path = Path(root) / f"cv{fold}" / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ROC requires class scores, but checkpoint is missing: {ckpt_path}"
            )

        model = ResNet(
            hidden_sizes=[100] * 6,
            num_blocks=[2] * 6,
            input_dim=x_test.shape[1],
            in_channels=64,
            n_classes=2,
        )
        feature_size = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(feature_size, feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_size, 2),
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            logits = []
            for start in range(0, len(x_tensor), 256):
                logits.append(model(x_tensor[start : start + 256]))
            logits = torch.cat(logits, dim=0)
            prob = torch.softmax(logits, dim=1).cpu().numpy()

        all_true.append(y_test)
        all_score.append(prob[:, 1])
        all_pred.append(prob.argmax(axis=1))

    return np.concatenate(all_true), np.concatenate(all_score), np.concatenate(all_pred)


def row_percent(cm):
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm * 100.0, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)


def annotate_matrix(ax, data, fontsize=9):
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if value >= 55 else "#222222",
            )


def style_heatmap_axis(ax, xlabels, ylabels, xtop=True, xrotation=90, yrotation=0, fontsize=12):
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=xrotation, ha="center", va="bottom" if xtop else "top")
    ax.set_yticklabels(ylabels, rotation=yrotation)
    if xtop:
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis="both", width=2.0, length=5, labelsize=fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fig2(class30_root, output, panel_a_output=None, panel_b_output=None):
    y_true, y_pred = load_cv_predictions(class30_root)
    strain_labels = [STRAINS[i] for i in ORDER]
    cm30 = row_percent(confusion_matrix(y_true, y_pred, labels=ORDER))

    y_true_ab = np.array([ATCC_GROUPINGS[i] for i in y_true])
    y_pred_ab = np.array([ATCC_GROUPINGS[i] for i in y_pred])
    ab_labels = [ANTIBIOTICS[i] for i in AB_ORDER]
    cm_ab = row_percent(confusion_matrix(y_true_ab, y_pred_ab, labels=AB_ORDER))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 2.4})
    fig = plt.figure(figsize=(12.2, 12.0))
    ax1 = fig.add_axes([0.20, 0.07, 0.76, 0.76])
    ax1.imshow(cm30, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
    annotate_matrix(ax1, cm30, fontsize=8.2)
    style_heatmap_axis(
        ax1,
        strain_labels,
        strain_labels,
        xtop=True,
        xrotation=90,
        yrotation=0,
        fontsize=13.8,
    )
    fig.patches.append(
        Rectangle(
            (0.62, 0.37),
            0.36,
            0.43,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="none",
            zorder=2,
        )
    )
    ax2 = fig.add_axes([0.68, 0.45, 0.27, 0.24], facecolor="white", zorder=3)
    ax2.imshow(cm_ab, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
    annotate_matrix(ax2, cm_ab, fontsize=8.3)
    style_heatmap_axis(
        ax2,
        ab_labels,
        ab_labels,
        xtop=True,
        xrotation=90,
        yrotation=0,
        fontsize=7.7,
    )
    if panel_a_output:
        save_single_heatmap(
            cm30,
            strain_labels,
            strain_labels,
            panel_a_output,
            figsize=(11.0, 11.0),
            tick_fontsize=12.5,
            annot_fontsize=7.8,
        )
    if panel_b_output:
        save_single_heatmap(
            cm_ab,
            ab_labels,
            ab_labels,
            panel_b_output,
            figsize=(5.0, 5.0),
            tick_fontsize=10.5,
            annot_fontsize=10.0,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_single_heatmap(
    cm,
    xlabels,
    ylabels,
    output,
    panel_label=None,
    figsize=(5, 5),
    tick_fontsize=12,
    annot_fontsize=12,
    colorbar=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
    annotate_matrix(ax, cm, fontsize=annot_fontsize)
    style_heatmap_axis(
        ax,
        xlabels,
        ylabels,
        xtop=True,
        xrotation=90,
        yrotation=0,
        fontsize=tick_fontsize,
    )
    if panel_label:
        ax.text(-0.26, 1.17, panel_label, transform=ax.transAxes, fontsize=26, fontweight="bold", va="top")
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=tick_fontsize)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_fig3(
    class2_root,
    output,
    panel_a_output=None,
    panel_b_output=None,
    x_test_fn="data/bacteria-id/preprocessed/X_test_binary.npy",
    y_test_fn="data/bacteria-id/preprocessed/y_test_binary.npy",
):
    y_true, y_pred = load_cv_predictions(class2_root)
    labels = [0, 1]
    cm = row_percent(confusion_matrix(y_true, y_pred, labels=labels))

    y_true_score, y_score, y_pred_score = load_binary_scores_from_checkpoints(
        class2_root, x_test_fn, y_test_fn
    )
    fpr, tpr, _ = roc_curve(y_true_score, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 2.4})
    fig = plt.figure(figsize=(13.2, 6.0))

    ax1 = fig.add_axes([0.08, 0.17, 0.37, 0.62])
    im = ax1.imshow(cm, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
    annotate_matrix(ax1, cm, fontsize=21)
    style_heatmap_axis(
        ax1,
        ["MRSA", "MSSA"],
        ["MRSA", "MSSA"],
        xtop=True,
        xrotation=90,
        yrotation=0,
        fontsize=24,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.05)
    cbar.ax.tick_params(labelsize=16, width=2.0, length=5)

    ax2 = fig.add_axes([0.58, 0.14, 0.37, 0.72])
    ax2.plot(fpr, tpr, color="navy", lw=3.2)
    ax2.plot([0, 1], [0, 1], color="black", lw=2.8, linestyle="--")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("False Positive Rate", fontsize=24, labelpad=12)
    ax2.set_ylabel("True Positive Rate", fontsize=24, labelpad=12)
    ax2.tick_params(axis="both", labelsize=22, width=2.2, length=7)
    for spine in ax2.spines.values():
        spine.set_linewidth(2.4)
    metrics_path = Path(output).with_suffix(".metrics.txt")
    metrics_path.write_text(f"ROC AUC (positive class = MSSA): {roc_auc:.6f}\n")

    if panel_a_output:
        save_single_heatmap(
            cm,
            ["MRSA", "MSSA"],
            ["MRSA", "MSSA"],
            panel_a_output,
            figsize=(4.8, 4.8),
            tick_fontsize=22,
            annot_fontsize=20,
            colorbar=True,
        )
    if panel_b_output:
        save_roc_panel(fpr, tpr, panel_b_output, roc_auc)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_roc_panel(fpr, tpr, output, roc_auc):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(fpr, tpr, color="navy", lw=3.0)
    ax.plot([0, 1], [0, 1], color="black", lw=2.5, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=21, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=21, labelpad=10)
    ax.tick_params(axis="both", labelsize=19, width=2.0, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class30-root")
    parser.add_argument("--class2-root")
    parser.add_argument("--fig2-out", default="docs/main2/figures/fig2.png")
    parser.add_argument("--fig3-out", default="docs/main2/figures/fig3.png")
    parser.add_argument("--fig2a-out", default="docs/main2/figures/fig2_a.png")
    parser.add_argument("--fig2b-out", default="docs/main2/figures/fig2_b.png")
    parser.add_argument("--fig3a-out", default="docs/main2/figures/fig3_a.png")
    parser.add_argument("--fig3b-out", default="docs/main2/figures/fig3_b.png")
    parser.add_argument("--x-test-binary", default="data/bacteria-id/preprocessed/X_test_binary.npy")
    parser.add_argument("--y-test-binary", default="data/bacteria-id/preprocessed/y_test_binary.npy")
    args = parser.parse_args()

    if args.class30_root:
        plot_fig2(args.class30_root, args.fig2_out, args.fig2a_out, args.fig2b_out)
    if args.class2_root:
        plot_fig3(
            args.class2_root,
            args.fig3_out,
            args.fig3a_out,
            args.fig3b_out,
            args.x_test_binary,
            args.y_test_binary,
        )


if __name__ == "__main__":
    main()
