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
