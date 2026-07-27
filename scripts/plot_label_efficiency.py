"""Regenerate Figure 4: downstream label efficiency on the Bacteria-ID 30-isolate task.

Consumes results/label_efficiency.json, produced by aggregating the
`labelsweep_n{N}` runs of hpc/run_pyssl_bacteria_label_sweep.sbatch.
Every point is the mean +- standard deviation of the independent-test accuracy
over five random fine-tuning train/validation splits.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEVELS = [10, 20, 50, 100]

SERIES = [
    ("supervised", "Supervised pretraining", "#000000", "--"),
    ("byol", "BYOL pretraining", "#E8820C", "-"),
    ("no_pre", "w/o pretraining", "#1F7A2E", "-"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/label_efficiency.json")
    parser.add_argument("--output", default="docs/main/figures/fig4.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with open(args.data) as fh:
        stats = json.load(fh)

    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    for key, label, color, style in SERIES:
        if key not in stats:
            continue
        mu = np.array([stats[key][str(n)][0] for n in LEVELS]) / 100.0
        sd = np.array([stats[key][str(n)][1] for n in LEVELS]) / 100.0
        ax.plot(LEVELS, mu, style, color=color, linewidth=2.0, label=label)
        ax.fill_between(LEVELS, mu - sd, mu + sd, color=color, alpha=0.20, linewidth=0)

    ax.set_xscale("log")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(n) for n in LEVELS])
    ax.minorticks_off()
    ax.set_xlabel("Number of labels", fontsize=17)
    ax.set_ylabel("Test accuracy", fontsize=17)
    ax.set_title("30-Isolates classification over data size", fontsize=19)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="major", linestyle=":", color="0.6", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
    ax.legend(loc="lower right", fontsize=14, framealpha=1.0, edgecolor="0.7")

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
