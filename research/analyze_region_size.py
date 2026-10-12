"""Region-level checks of the Gaussian model against the same-feature 1-NN, from the saved final maps.

1. Defect size within categories: for every ground-truth region (connected component, 448 px), the fraction of
   its pixels above the threshold at which 5% (and 1%) of all normal pixels of the category are flagged, for
   GaLAD and 1-NN on the same N and draw. Output rows: one per region, method difference averaged over draws.
2. Border check: AU-PRO@0.3/0.05 on the interior only (outer 32 px of the 448 px image removed from maps and
   masks), to see whether the gain sits at the image border (reflected padding of the rotated references).
Maps: output/server/scores/final_square/*/maps_N{1,5}_s{0-4}_{galad,knn}.npz.
Usage: uv run analyze_region_size.py   -> output/region_size.csv, output/interior_aupro.csv
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned

_argv, sys.argv = sys.argv, sys.argv[:1]
import galad_train_all as G  # noqa: E402  (parses its own arguments on import)
sys.argv = _argv
RES, BORDER, NS, SEEDS = 448, 32, (1, 5), range(5)
root = Path("output/server/scores/final_square")
reg_rows, int_rows = [], []
for d in sorted(root.iterdir()):
    ds = d.name.replace("__", "/")
    _, info = G.load_dataset(ds)
    masks = []
    for e in info:
        m = G.load_mask(e)
        g = np.zeros((RES, RES), np.uint8) if m is None else (m > 0).astype(np.uint8)
        if g.shape != (RES, RES):  # same resizing as server_eval's ground truth
            g = (zoom(g.astype(float), (RES / g.shape[0], RES / g.shape[1]), order=0) > 0.5).astype(np.uint8)
        masks.append(g)
    regions = []  # (image index, pixel indices, area, touches border band)
    for i, g in enumerate(masks):
        n, lab = cv2.connectedComponents(g, connectivity=8)
        for r in range(1, n):
            ys, xs = np.nonzero(lab == r)
            edge = (ys.min() < BORDER) or (xs.min() < BORDER) or (ys.max() >= RES - BORDER) or (xs.max() >= RES - BORDER)
            regions.append((i, (ys, xs), len(ys), edge))
    if not regions:
        continue
    inner = [g[BORDER:-BORDER, BORDER:-BORDER] for g in masks]
    inner_regions = mask_regions(inner)
    ov = {}
    for n in NS:
        for s in SEEDS:
            for method in ("galad", "knn"):
                f = d / f"maps_N{n}_s{s}_{method}.npz"
                if not f.exists():
                    continue
                maps = [zoom(x.astype(np.float32), RES / x.shape[0], order=1) for x in np.load(f)["maps"]]
                normal = np.concatenate([mp[::4, ::4][g[::4, ::4] == 0] for mp, g in zip(maps, masks)])
                thr = {fpr: np.quantile(normal, 1 - fpr) for fpr in (0.05, 0.01)}
                for j, (i, (ys, xs), area, edge) in enumerate(regions):
                    v = maps[i][ys, xs]
                    for fpr, t in thr.items():
                        ov.setdefault((j, n, fpr, method), []).append((v > t).mean())
                pro = std_pixel_metrics_binned([mp[BORDER:-BORDER, BORDER:-BORDER] for mp in maps], inner,
                                               inner_regions, limit=(0.3, 0.05))[0]
                if pro is not None:
                    int_rows.append(dict(dataset=ds, method=method, n_train=n, seed=s, aupro_30=pro[0], aupro_05=pro[1]))
    for j, (i, _, area, edge) in enumerate(regions):
        for n in NS:
            for fpr in (0.05, 0.01):
                gl, kn = ov.get((j, n, fpr, "galad")), ov.get((j, n, fpr, "knn"))
                if gl and kn:
                    reg_rows.append(dict(dataset=ds, region=j, image=i, area_pct=area / RES ** 2 * 100, edge=edge,
                                         n_train=n, fpr=fpr, galad=np.mean(gl), knn=np.mean(kn)))
    print(ds, len(regions), "regions", flush=True)
    pd.DataFrame(reg_rows).to_csv("output/region_size.csv", index=False)
    pd.DataFrame(int_rows).to_csv("output/interior_aupro.csv", index=False)
