"""Figures of the Electronics manuscript (version of 9 Oct 2026): results against N and the pipeline thumbnails.

vs_n.pdf: AU-PRO up to FPR 0.3 and up to FPR 0.05 against the number of reference images, mean over the 55 paper
categories and 5 draws, for AnomalyDINO (official), AnomalyDINO (DINOv3-L), PatchCore (when its runs cover
every category and N) and GaLAD (final runs, output/server/final_square.csv).
Usage: uv run make_paper_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = Path("paper/electronics/figs")
NS = [1, 2, 4, 5]
BENCH = ("mvtec_AD", "VisA", "AutoVI", "btad", "GoodsAD", "mvtec_loco_AD", "mvtec_ad_2")
cols = ["dataset", "method", "n_train", "seed", "img_ap", "aupro_30", "aupro_05"]
Lsq, NEW = pd.read_csv("output/server_run/l_square.csv"), pd.read_csv("output/server/new_square.csv")
F = pd.read_csv("output/server/final_square.csv")
A = pd.concat([Lsq[Lsq.method == "adino"], NEW[NEW.method == "adino"]])
O = pd.read_csv("output/adino_official.csv").assign(method="official").rename(columns={"aupro_std": "aupro_30"})
O["aupro_05"] = float("nan")
if Path("output/adino_official_v2.csv").exists():  # rerun with AU-PRO up to FPR 0.05 (drawn once it is complete)
    V2 = pd.read_csv("output/adino_official_v2.csv").drop_duplicates(["dataset", "n_train", "seed"], keep="last")
    O = O.drop(columns="aupro_05").merge(V2[["dataset", "n_train", "seed", "aupro_05"]], how="left",
                                          on=["dataset", "n_train", "seed"])
pcs = [pd.read_csv(f, sep=";") for f in ["output/server/patchcore_new.csv"] + sorted(map(str, Path("output/server_run").glob("patchcore_results_*.csv")))
       + ["output/patchcore_laptop.csv"] if Path(f).exists()]
P = pd.concat(pcs).rename(columns={"img_aupr": "img_ap"}).assign(method="patchcore")
P = P.drop_duplicates(["dataset", "model", "n_train", "seed"], keep="last")
D = pd.concat([F[F.method == "galad"][cols], A[cols], O[cols], P[P.model == "patchcore"][cols]])
CATS = sorted(c for c in A.dataset.unique() if c.split("/")[0] in BENCH)
D = D[D.dataset.isin(CATS)]
STYLE = {"official": ("AnomalyDINO (official)", "#2a78d6", "s", "--"),
         "adino": ("AnomalyDINO (DINOv3-L)", "#4a3aa7", "o", "--"),
         "patchcore": ("PatchCore", "#1baf7a", "^", ":"),
         "galad": ("GaLAD", "#d95f02", "D", "-")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, metric, title in ((axes[0], "aupro_30", "(a) AU-PRO (FPR $\\leq$ 0.3)"), (axes[1], "aupro_05", "(b) AU-PRO (FPR $\\leq$ 0.05)")):
    for m, (lab, col, mk, ls) in STYLE.items():
        t = D[D.method == m].dropna(subset=[metric]).groupby(["n_train", "dataset"])[metric].mean().unstack("dataset")
        t = t.reindex(index=NS, columns=CATS)
        if t.isna().any().any():  # incomplete coverage: not drawn
            continue
        y = t.mean(axis=1) * 100
        ax.plot(y.index, y.values, ls, color=col, marker=mk, ms=4, lw=1.5 if m == "galad" else 1.1, label=lab)
    ax.set_xticks(NS)
    ax.set_xlabel("Reference images $N$", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("AU-PRO (%)", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, 0))
plt.tight_layout(pad=0.4, rect=(0, 0.08, 1, 1))
plt.savefig(FIG / "vs_n.pdf")
print(f"wrote {FIG / 'vs_n.pdf'} ({len(CATS)} categories)")

# Per-category localization (scatter.pdf) and gain by defect size (size.pdf): AU-PRO up to FPR 0.3, mean over
# N = 1, 2, 4, 5 and the draws; a comparator is drawn only when it covers every category and N.
cat = {m: D[D.method == m].groupby(["dataset", "n_train"]).aupro_30.mean().unstack().reindex(index=CATS, columns=NS)
       for m in STYLE}
cat = {m: t.mean(axis=1) * 100 for m, t in cat.items() if not t.isna().any().any()}
comps = [m for m in ("official", "adino", "patchcore") if m in cat]
DEVB = ("mvtec_AD", "VisA")
dev = np.array([c.split("/")[0] in DEVB for c in CATS])
fig, axes = plt.subplots(1, len(comps), figsize=(2.45 * len(comps), 2.95))
lo = 0
for i, (ax, m) in enumerate(zip(np.atleast_1d(axes), comps)):
    x, y = cat[m].values, cat["galad"].values
    ax.fill_between([lo, 100], [lo, 100], 100, color=STYLE["galad"][1], alpha=0.08, lw=0, zorder=0)  # GaLAD better
    ax.plot([lo, 100], [lo, 100], color="#888888", lw=0.8, ls="--", zorder=1)
    ax.scatter(x[dev], y[dev], s=20, marker="o", color="#333333", lw=0, zorder=3, label="Development category")
    ax.scatter(x[~dev], y[~dev], s=22, marker="^", facecolor="white", edgecolor="#333333", lw=0.8, zorder=3,
               label="External category")
    d = y - x
    ax.set_title(f"({'abc'[i]}) GaLAD vs. {STYLE[m][0]}\nGaLAD better in {(d > .5).sum()}, worse in {(d < -.5).sum()}",
                 fontsize=7.5, loc="left")
    ax.set_xlim(lo, 101), ax.set_ylim(lo, 101), ax.set_aspect("equal")
    ax.set_xlabel(f"{STYLE[m][0]}, AU-PRO (%)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("GaLAD, AU-PRO (%)", fontsize=7.5)
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
h.append(plt.Rectangle((0, 0), 1, 1, color=STYLE["galad"][1], alpha=0.15, lw=0))
l.append("Area where GaLAD localizes better")
fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
plt.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1))
plt.savefig(FIG / "scatter.pdf")
print(f"wrote {FIG / 'scatter.pdf'} ({', '.join(comps)})")

area = pd.read_csv("output/defect_area.csv").set_index("dataset").median_area_pct.reindex(CATS).dropna()
q = area.quantile([1 / 3, 2 / 3]).values
groups = [("Small", area <= q[0], f"$\\leq${q[0]:.1f}%"), ("Medium", (area > q[0]) & (area <= q[1]), f"{q[0]:.1f}–{q[1]:.1f}%"),
          ("Large", area > q[1], f"$>${q[1]:.1f}%")]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(4.3, 2.9))
w = 0.8 / len(comps)
for j, m in enumerate(comps):
    for i, (_, sel, _) in enumerate(groups):
        dd = (cat["galad"] - cat[m])[area[sel].index].values
        bs = dd[rng.integers(0, len(dd), (10000, len(dd)))].mean(1)
        xpos = i + (j - (len(comps) - 1) / 2) * w
        ax.bar(xpos, dd.mean(), w * 0.92, color=STYLE[m][1], label=STYLE[m][0] if i == 0 else None, zorder=2)
        ax.errorbar(xpos, dd.mean(), yerr=[[dd.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - dd.mean()]],
                    color="#333333", lw=0.8, capsize=2, zorder=3)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g} defects\n{r} of the image\n{s.sum()} categories" for g, s, r in groups], fontsize=7.5)
ax.set_ylabel("AU-PRO gain of GaLAD (points)", fontsize=8)
ax.tick_params(axis="y", labelsize=7.5)
ax.grid(axis="y", alpha=0.3, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, title="Gain of GaLAD over:", title_fontsize=7.5, loc="upper right")
ax.set_ylim(top=ax.get_ylim()[1] * 1.45)  # room for the legend above the bars
plt.tight_layout(pad=0.4)
plt.savefig(FIG / "size.pdf")
print(f"wrote {FIG / 'size.pdf'}")

# Thumbnails for the pipeline figure (Figure 1): a reference image, a test image and the GaLAD map of
# the MVTec AD capsule example (maps cached by make_paper_qualitative.py).
from PIL import Image  # noqa: E402
import cv2  # noqa: E402

z = np.load("output/qualitative_maps.npz")
Image.open("data/mvtec_AD/capsule/train/good/000.png").convert("RGB").resize((224, 224)).save(FIG / "fig1_reference.png")
Image.fromarray(z["0_img"]).resize((224, 224)).save(FIG / "fig1_test.png")
rgb = (plt.get_cmap("inferno")(np.clip(z["0_galad"] / 2, 0, 1))[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb).resize((224, 224)).save(FIG / "fig1_map.png")
# Reference thumbnail as a 2x2 grid of four of its eight rotations (0, 45, 90 and 135 degrees).
ref = np.asarray(Image.open("data/mvtec_AD/capsule/train/good/000.png").convert("RGB").resize((224, 224)))
tiles = []
for angle in (0, 45, 90, 135):
    m = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    v = cv2.warpAffine(ref, m, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    tiles.append(np.pad(v, ((4, 4), (4, 4), (0, 0)), constant_values=255))
Image.fromarray(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])).save(FIG / "fig1_reference_rot.png")
print("wrote fig1 thumbnails")
