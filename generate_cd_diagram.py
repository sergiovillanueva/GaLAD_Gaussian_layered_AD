"""
Generate the Critical Difference (CD) diagram for the 5-method Demsar analysis.

Recomputes ranks from the raw CSVs so the figure is fully reproducible, then
draws the canonical Nemenyi CD diagram (Demsar 2006) in the project's
publication style.

Output: paper/figs/cd_diagram.pdf (and .png)
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range

OUTPUT_DIR = Path("results")
FIGS_DIR = Path("results/figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

KEEP = ("mvtec_AD/", "mvtec_loco_AD/", "VisA/", "AutoVI/", "btad/", "GoodsAD/")
METHODS = ["galad", "patchcore", "reverse_distillation", "dinomaly", "efficient_ad"]
NAME_MAP = {
    "galad": "GaLAD",
    "patchcore": "PatchCore",
    "reverse_distillation": "Reverse Distillation",
    "dinomaly": "Dinomaly",
    "efficient_ad": "EfficientAD",
}
ALPHA = 0.05


def load():
    galad = pd.read_csv(OUTPUT_DIR / "galad_results.csv", sep=";")
    galad["model"] = "galad"
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_results.csv", sep=";")
    cols = ["dataset", "model", "seed", "train_limit", "img_aupr"]
    df = pd.concat([galad[cols], baseline[cols]], ignore_index=True)
    df = df[df["dataset"].str.startswith(KEEP)]
    df = df.drop_duplicates(subset=["dataset", "model", "seed", "train_limit"], keep="last")
    return df


def compute_ranks(df):
    sub = df[df["train_limit"].isin([5, 10, 20])]
    per = (sub.groupby(["dataset", "model", "train_limit"])["img_aupr"].mean()
           .reset_index()
           .groupby(["dataset", "model"])["img_aupr"].mean().reset_index())
    pivot = per.pivot(index="dataset", columns="model", values="img_aupr").dropna()
    pivot = pivot[METHODS]
    ranks = pivot.rank(axis=1, ascending=False)
    chi2, p = friedmanchisquare(*[pivot[m].values for m in METHODS])
    N, k = pivot.shape
    q = studentized_range.ppf(1 - ALPHA, k, df=1e10) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6 * N))
    return {m: float(ranks[m].mean()) for m in METHODS}, cd, chi2, p, N


def _find_cliques(sorted_methods, cd):
    """Maximal groups of consecutive methods within CD of each other."""
    k = len(sorted_methods)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and (sorted_methods[j + 1][1] - sorted_methods[i][1]) <= cd:
            j += 1
        if j > i:
            cliques.append(list(range(i, j + 1)))
    # remove cliques fully contained in another
    filtered = []
    for c in cliques:
        if not any(set(c) < set(o) for o in cliques):
            filtered.append(c)
    return filtered


def plot_cd_diagram(mean_ranks, cd, output_path):
    sorted_methods = sorted(mean_ranks.items(), key=lambda x: x[1])
    k = len(sorted_methods)

    cliques = _find_cliques(sorted_methods, cd)
    n_cliques = len(cliques)

    low_rank = 1
    high_rank = k
    x_pad = 0.6
    y_axis = 0.0
    label_spacing = 0.50
    clique_spacing = 0.30

    n_left = (k + 1) // 2
    left_methods = sorted_methods[:n_left]
    right_methods = sorted_methods[n_left:]

    y_top = y_axis + 1.2
    # clique bars sit just below the axis (canonical Demsar layout)
    clique_y_start = y_axis - 0.16
    y_clique_bottom = clique_y_start - n_cliques * clique_spacing
    y_label_bottom = y_axis - 0.6 - max(n_left, len(right_methods)) * label_spacing
    y_bottom = min(y_clique_bottom, y_label_bottom)

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.6))
    ax.set_xlim(low_rank - x_pad - 3.2, high_rank + x_pad + 3.2)
    ax.set_ylim(y_bottom - 0.3, y_top + 0.2)
    ax.axis("off")

    text_bbox = dict(boxstyle="round,pad=0.08", facecolor="white",
                     edgecolor="none", alpha=1.0)

    # axis line + ticks
    ax.hlines(y_axis, low_rank, high_rank, colors="black", linewidth=2.0)
    for r in range(1, k + 1):
        ax.vlines(r, y_axis - 0.10, y_axis + 0.10, colors="black", linewidth=2.0)
        ax.text(r, y_axis + 0.18, str(r), ha="center", va="bottom", fontsize=15,
                fontweight="bold")

    # CD bar
    cd_y = y_axis + 0.75
    cd_start = low_rank
    cd_end = low_rank + cd
    ax.hlines(cd_y, cd_start, cd_end, colors="#CC0000", linewidth=3.5)
    ax.vlines(cd_start, cd_y - 0.10, cd_y + 0.10, colors="#CC0000", linewidth=3.5)
    ax.vlines(cd_end, cd_y - 0.10, cd_y + 0.10, colors="#CC0000", linewidth=3.5)
    ax.text((cd_start + cd_end) / 2, cd_y + 0.15, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=15, color="#CC0000", fontweight="bold")

    # left side
    x_label_left = low_rank - x_pad - 3.1
    for idx, (method, rank) in enumerate(left_methods):
        y_pos = y_axis - 0.6 - idx * label_spacing
        label = NAME_MAP.get(method, method)
        ax.hlines(y_pos, x_label_left + 0.05, rank, colors="#AAAAAA", linewidth=0.9, zorder=1)
        ax.vlines(rank, y_pos, y_axis, colors="#AAAAAA", linewidth=0.9, zorder=1)
        ax.plot(rank, y_axis, "ko", markersize=7, zorder=5)
        ax.text(x_label_left, y_pos, f"{label} ({rank:.2f})", ha="left", va="center",
                fontsize=15, fontweight="bold" if idx == 0 else "normal",
                bbox=text_bbox, zorder=4)

    # right side
    x_label_right = high_rank + x_pad + 3.1
    for idx, (method, rank) in enumerate(right_methods):
        y_pos = y_axis - 0.6 - idx * label_spacing
        label = NAME_MAP.get(method, method)
        ax.hlines(y_pos, rank, x_label_right - 0.05, colors="#AAAAAA", linewidth=0.9, zorder=1)
        ax.vlines(rank, y_pos, y_axis, colors="#AAAAAA", linewidth=0.9, zorder=1)
        ax.plot(rank, y_axis, "ko", markersize=7, zorder=5)
        ax.text(x_label_right, y_pos, f"({rank:.2f}) {label}", ha="right", va="center",
                fontsize=15, bbox=text_bbox, zorder=4)

    # clique bars: just below the axis, connecting methods not significantly different
    clique_color = "#2C3E50"
    for cidx, clique in enumerate(cliques):
        min_rank = sorted_methods[clique[0]][1]
        max_rank = sorted_methods[clique[-1]][1]
        bar_y = clique_y_start - cidx * 0.13
        ax.plot([min_rank, max_rank], [bar_y, bar_y], color=clique_color,
                linewidth=4.5, solid_capstyle="round", zorder=6)

    plt.tight_layout()
    fmt = "pdf" if str(output_path).endswith(".pdf") else "png"
    plt.savefig(output_path, format=fmt, dpi=300, bbox_inches="tight")
    # also save a PNG preview
    plt.savefig(str(output_path).replace(".pdf", ".png"), format="png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"  CD diagram saved to {output_path}")


def main():
    df = load()
    ranks, cd, chi2, p, N = compute_ranks(df)
    print(f"N={N} categories, Friedman chi2={chi2:.2f}, p={p:.2e}, CD={cd:.3f}")
    for m in sorted(ranks, key=ranks.get):
        print(f"  {NAME_MAP[m]:<22} {ranks[m]:.3f}")
    plot_cd_diagram(ranks, cd, str(FIGS_DIR / "cd_diagram.pdf"))


if __name__ == "__main__":
    main()
