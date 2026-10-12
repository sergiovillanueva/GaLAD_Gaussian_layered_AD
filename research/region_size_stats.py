"""Statistics of analyze_region_size.py: does the Gaussian gain over 1-NN grow as defect regions get smaller
within each category, and does the interior-only AU-PRO keep the gain?
Usage: uv run region_size_stats.py
"""

import numpy as np
import pandas as pd

R = pd.read_csv("output/region_size.csv")
R["diff"] = (R.galad - R.knn) * 100
R["la"] = np.log10(R.area_pct)
q = R.drop_duplicates(["dataset", "region"]).area_pct.quantile([1 / 3, 2 / 3]).values
R["size"] = np.where(R.area_pct <= q[0], "small", np.where(R.area_pct <= q[1], "medium", "large"))
print(f"{R.drop_duplicates(['dataset', 'region']).shape[0]} regions in {R.dataset.nunique()} categories; "
      f"region terciles at {q[0]:.3f}% and {q[1]:.3f}% of the image")
rng = np.random.default_rng(0)


def slope(d):
    """Within-category slope of diff on log10 area (category-demeaned)."""
    x = d.la - d.groupby("dataset").la.transform("mean")
    y = d["diff"] - d.groupby("dataset")["diff"].transform("mean")
    return (x * y).sum() / (x * x).sum()


def cat_boot(d, stat, b=2000):
    cats = d.dataset.unique()
    g = {c: d[d.dataset == c] for c in cats}
    vals = [stat(pd.concat([g[c] for c in rng.choice(cats, len(cats))])) for _ in range(b)]
    return stat(d), np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def small_gain(d):
    return d[d["size"] == "small"].groupby("dataset")["diff"].mean().mean()


for n in (1, 5):
    for fpr in (0.05, 0.01):
        d = R[(R.n_train == n) & (R.fpr == fpr)]
        s = cat_boot(d, slope)
        by = d.groupby(["size", "dataset"])["diff"].mean().groupby("size").mean()
        sub = d[~d.dataset.str.startswith("mvtec_ad_2") & ~d.edge]
        g = cat_boot(sub, small_gain)
        print(f"N={n} FPR={fpr}: slope per decade {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}]; "
              f"gain small/medium/large {by['small']:+.1f} / {by['medium']:+.1f} / {by['large']:+.1f}; "
              f"small, without AD 2 and border {g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]")

I = pd.read_csv("output/interior_aupro.csv")
F = pd.read_csv("output/server/final_square.csv")
F = F[F.n_train.isin([1, 5]) & F.method.isin(["galad", "knn"])]
for name, D in (("full image", F), ("interior", I)):
    t = D.groupby(["dataset", "method"])[["aupro_30", "aupro_05"]].mean().unstack("method")
    t = t.loc[t.index.isin(I.dataset.unique())]
    for m in ("aupro_30", "aupro_05"):
        x = ((t[m].galad - t[m].knn) * 100).values
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
        print(f"{name} {m}: GaLAD - 1-NN {x.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
