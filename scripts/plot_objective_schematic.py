"""Figure 1: the three self-supervised objectives compared in this work.

Draws SimCLR v2, MoCo v3 and BYOL side by side so that the structural
progression is visible: two gradient-carrying branches contrasted against
negatives, one gradient branch plus a momentum branch contrasted against
negatives, and one gradient branch plus a momentum branch with no negatives
at all. A shared strip at the bottom shows how the pretrained encoder is
reused downstream.

The source spectrum and the two views are drawn from the data rather than
sketched: a preprocessed Bacteria-ID reference spectrum is passed twice through
the same physically motivated augmentation pipeline used for pretraining, so
the figure shows the actual difference between the two views the objectives are
asked to align. If the preprocessed data are not present the nodes fall back to
plain labels and the rest of the figure is unchanged.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

GRAD = "#CFE2F3"      # branch that receives gradients
GRAD_E = "#2E6DA4"
FROZEN = "#E8E8E8"    # momentum / stop-gradient branch
FROZEN_E = "#8A8A8A"
LOSS = "#FBE0C4"
LOSS_E = "#D2792A"
HEAD = "#E4D5F0"
HEAD_E = "#7A55A0"
SRC_C = "#222222"     # source spectrum
V1_C = "#2E5FA3"      # view 1
V2_C = "#C0392B"      # view 2

LX, RX, BW, BH = 2.7, 7.3, 3.0, 0.92


def load_traces(x_fn, index, seed):
    """Return the source spectrum and two physically motivated views of it.

    The augmentation modules operate on a (1, length) array, which is the shape
    the SSL dataset hands them, and they draw from both `random` and
    `numpy.random`, so the whole trio is seeded through utils.seed_all.
    """
    try:
        sys.path.insert(0, os.getcwd())
        import utils

        x = np.load(x_fn)[index].astype(np.float32)
        transform = utils.get_trans_from_augtype("phys", p=1.0)
        utils.seed_all(seed)
        v1 = np.asarray(transform(np.expand_dims(x, 0)), dtype=np.float32).ravel()
        v2 = np.asarray(transform(np.expand_dims(x, 0)), dtype=np.float32).ravel()
        return x, v1, v2
    except Exception as exc:  # preprocessed data or torch unavailable
        print(f"[warn] drawing plain nodes instead of spectra: {exc}")
        return None, None, None


def box(ax, x, y, text, fc, ec, w=BW, h=BH, style="round,pad=0.02,rounding_size=0.12",
        fs=10.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=1.5, linestyle=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)


def trace_box(ax, x, y, trace, color, w, h, fallback):
    """A white node whose interior is an actual spectrum."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor="#FFFFFF", edgecolor="#666666",
                                linewidth=1.4, zorder=4))
    if trace is None:
        ax.text(x, y, fallback, ha="center", va="center", fontsize=10, zorder=6)
        return
    t = np.asarray(trace, dtype=float)
    lo, hi = float(t.min()), float(t.max())
    t = (t - lo) / (hi - lo) if hi > lo else np.zeros_like(t)
    px, py = 0.10 * w, 0.14 * h
    xs = np.linspace(x - w / 2 + px, x + w / 2 - px, t.size)
    ys = y - h / 2 + py + t * (h - 2 * py)
    ax.plot(xs, ys, color=color, linewidth=0.7, solid_joinstyle="round", zorder=6)


def arrow(ax, p0, p1, ls="-", color="#333333", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def draw_panel(ax, kind, title, letter, traces):
    src, v1, v2 = traces
    ax.set_xlim(0, 10); ax.set_ylim(1.05, 11.75); ax.axis("off")
    ax.text(0.05, 11.55, letter, fontsize=15, fontweight="bold", ha="left", va="top")
    ax.text(5, 11.5, title, fontsize=12.5, ha="center", va="top", fontweight="bold")

    momentum = kind in ("moco", "byol")
    rc, re_ = (FROZEN, FROZEN_E) if momentum else (GRAD, GRAD_E)
    rls = "--" if momentum else "-"

    trace_box(ax, 5, 10.15, src, SRC_C, 2.9, 0.80, "spectrum $x$")
    ax.text(5, 10.80, "spectrum $x$", fontsize=10.5, ha="center", va="center")
    arrow(ax, (4.1, 9.72), (LX + 0.55, 9.10))
    arrow(ax, (5.9, 9.72), (RX - 0.55, 9.10))
    ax.text(3.25, 9.62, "$t$", fontsize=10.5, ha="center", color="#555555")
    ax.text(6.75, 9.62, "$t'$", fontsize=10.5, ha="center", color="#555555")

    trace_box(ax, LX, 8.62, v1, V1_C, BW, 0.80, "view $x_1$")
    trace_box(ax, RX, 8.62, v2, V2_C, BW, 0.80, "view $x_2$")
    ax.text(LX, 7.94, "view $x_1$", fontsize=10, ha="center", va="center")
    ax.text(RX, 7.94, "view $x_2$", fontsize=10, ha="center", va="center")
    arrow(ax, (LX, 7.68), (LX, 7.22)); arrow(ax, (RX, 7.68), (RX, 7.22))

    box(ax, LX, 6.75, "encoder $f_\\theta$", GRAD, GRAD_E)
    box(ax, RX, 6.75, "encoder $f_\\xi$" if momentum else "encoder $f_\\theta$", rc, re_, ls=rls)
    arrow(ax, (LX, 6.29), (LX, 5.76)); arrow(ax, (RX, 6.29), (RX, 5.76))

    box(ax, LX, 5.3, "projector $g_\\theta$", HEAD, HEAD_E)
    box(ax, RX, 5.3, "projector $g_\\xi$" if momentum else "projector $g_\\theta$", rc, re_, ls=rls)

    if momentum:
        ax.add_patch(FancyArrowPatch((LX + BW / 2, 6.75), (RX - BW / 2, 6.75),
                                     arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                                     color="#8A8A8A", linestyle=(0, (4, 2)),
                                     connectionstyle="arc3,rad=-0.32", zorder=3))
        ax.text(5, 7.35, "EMA", fontsize=10, ha="center", color="#6A6A6A", style="italic")
        ax.text(RX, 4.42, "stop-gradient", fontsize=9.5, ha="center", color="#6A6A6A",
                style="italic")
        arrow(ax, (LX, 4.84), (LX, 4.32))
        box(ax, LX, 3.86, "predictor $q_\\theta$", HEAD, HEAD_E)
        ly = 3.40
    else:
        ax.text(5, 4.55, "shared weights", fontsize=9.5, ha="center", color="#2E6DA4",
                style="italic")
        ly = 4.84

    arrow(ax, (LX, ly), (4.4, 2.62), rad=0.12)
    arrow(ax, (RX, 4.84 if momentum else ly), (5.6, 2.62), rad=-0.12)

    if kind == "simclr":
        txt, sub = "NT-Xent loss", "positives vs. $2N-2$ in-batch negatives"
    elif kind == "moco":
        txt, sub = "InfoNCE loss", "positives vs. momentum-encoded negatives"
    else:
        txt, sub = "normalized $\\ell_2$ regression", "no negatives"
    box(ax, 5, 2.18, txt, LOSS, LOSS_E, w=6.4, h=0.9, fs=11)
    ax.text(5, 1.42, sub, fontsize=9.8, ha="center", color="#444444", style="italic")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="figures/fig1.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--spectra", default="data/bacteria-id/preprocessed/X_reference.npy")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()

    traces = load_traces(a.spectra, a.index, a.seed)

    fig = plt.figure(figsize=(13.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[6.7, 1.15], hspace=0.02, wspace=0.05)

    for i, (kind, title) in enumerate([("simclr", "SimCLR v2"), ("moco", "MoCo v3"),
                                       ("byol", "BYOL")]):
        draw_panel(fig.add_subplot(gs[0, i]), kind, title, "ABC"[i], traces)

    ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 30); ax.set_ylim(0, 3.1); ax.axis("off")
    ax.text(0.15, 2.82, "D", fontsize=15, fontweight="bold", ha="left", va="top")
    ax.text(2.0, 1.5, "downstream", fontsize=11, ha="center", va="center")
    for x, t, fc, ec in [(7.0, "labeled spectrum", "#FFFFFF", "#666666"),
                         (13.0, "pretrained encoder $f_\\theta$", GRAD, GRAD_E),
                         (19.5, "classifier", HEAD, HEAD_E),
                         (25.5, "isolate / diagnosis", LOSS, LOSS_E)]:
        box(ax, x, 1.5, t, fc, ec, w=5.0, h=1.0, fs=10.5)
    for x0, x1 in [(9.5, 10.5), (16.0, 17.0), (22.0, 23.0)]:
        arrow(ax, (x0, 1.5), (x1, 1.5))
    ax.text(13.0, 0.42, "projectors and predictors are discarded", fontsize=9.8,
            ha="center", color="#444444", style="italic")

    fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
