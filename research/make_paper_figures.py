"""Per-category paired AP differences (GaLAD - AnomalyDINO on DINOv3-L, mean over N) for the manuscript.

Reads paper/electronics/tables/per_category_diff.csv (written by make_paper_tables.py).
Writes paper/electronics/figs/per_category.pdf.
Usage: uv run make_paper_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIG = Path("paper/electronics/figs")
BENCH = ["mvtec_AD", "VisA", "mvtec_loco_AD", "AutoVI", "btad", "GoodsAD"]
NAMES = {"mvtec_AD": "MVTec AD", "VisA": "VisA", "mvtec_loco_AD": "LOCO", "AutoVI": "AutoVI", "btad": "BTAD",
         "GoodsAD": "GoodsAD"}

d = pd.read_csv(Path("paper/electronics/tables/per_category_diff.csv"), index_col=0).diff_ap
fig, ax = plt.subplots(figsize=(7.2, 2.5))
x, ticks = 0, []
for b in BENCH:
    s = d[d.index.str.startswith(b + "/")].sort_values()
    held_out = b not in ("mvtec_AD", "VisA")
    xs = range(x, x + len(s))
    ax.bar(xs, s.values, color="#d95f02" if held_out else "#7570b3", width=0.8)
    for xi, (cat, v) in zip(xs, s.items()):
        if cat in ("mvtec_loco_AD/screw_bag", "VisA/pipe_fryum"):  # the two losses discussed in the text
            ax.annotate(cat.split("/")[1].replace("_", " "), (xi, v), xytext=(4, -2), textcoords="offset points",
                        fontsize=6, va="top")
    ticks.append((x + (len(s) - 1) / 2, NAMES[b]))
    x += len(s) + 1
    ax.axvline(x - 1, color="0.85", lw=0.6)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks([t for t, _ in ticks])
ax.set_xticklabels([n for _, n in ticks], fontsize=8)
ax.set_xlim(-1, x - 1)
ax.set_ylabel("AP difference (points)", fontsize=8)
ax.tick_params(axis="y", labelsize=7)
ax.legend(handles=[Patch(color="#7570b3", label="Development"), Patch(color="#d95f02", label="Held-out")],
          fontsize=7, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.3)
plt.savefig(FIG / "per_category.pdf")
print(f"wrote {FIG / 'per_category.pdf'}; range {d.min():+.1f} to {d.max():+.1f}")

# Figure: image AP on the held-out categories and AU-PRO over all categories against N.
R = Path("output/server_run")
L = pd.read_csv(R / "l_square.csv")
O = pd.read_csv("output/adino_official.csv").assign(method="official").rename(columns={"aupro_std": "aupro_30"})
D = pd.concat([L[["dataset", "method", "n_train", "seed", "img_ap", "aupro_30"]],
               O[["dataset", "method", "n_train", "seed", "img_ap", "aupro_30"]]])
D = D[D.dataset.isin(L.dataset.unique())]
held = ~D.dataset.str.startswith(("mvtec_AD", "VisA"))
STYLE = {"official": ("AnomalyDINO (official)", "#999999", "s", "--"),
         "adino": ("AnomalyDINO (DINOv3-L)", "#7570b3", "o", "--"),
         "galad": ("GaLAD", "#d95f02", "D", "-")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, data, metric, title in ((axes[0], D[held], "img_ap", "(a) Image AP, held-out categories"),
                                (axes[1], D, "aupro_30", "(b) AU-PRO (FPR $\\leq$ 0.3), all categories")):
    for m, (lab, col, mk, ls) in STYLE.items():
        y = data[data.method == m].groupby(["n_train", "dataset"])[metric].mean().groupby("n_train").mean() * 100
        ax.plot(y.index, y.values, ls, color=col, marker=mk, ms=4, lw=1.5 if m == "galad" else 1.1, label=lab)
    ax.set_xticks([1, 2, 4, 5])
    ax.set_xlabel("Reference images $N$", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("%", fontsize=8)
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(pad=0.4)
plt.savefig(FIG / "vs_n.pdf")
print(f"wrote {FIG / 'vs_n.pdf'}")

# Thumbnails for the pipeline figure (Figure 1): a reference image, a test image and the GaLAD map of
# the MVTec AD capsule example of Figure 3 (maps cached by make_paper_qualitative.py).
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

z = np.load("output/qualitative_maps.npz")
Image.open("data/mvtec_AD/capsule/train/good/000.png").convert("RGB").resize((224, 224)).save(FIG / "fig1_reference.png")
Image.fromarray(z["0_img"]).resize((224, 224)).save(FIG / "fig1_test.png")
rgb = (plt.get_cmap("inferno")(np.clip(z["0_galad"] / 2, 0, 1))[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb).resize((224, 224)).save(FIG / "fig1_map.png")
print("wrote fig1 thumbnails")
