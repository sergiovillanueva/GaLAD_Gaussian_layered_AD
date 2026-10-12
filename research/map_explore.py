"""Pixel metrics from maps saved by server_eval.py --save-maps, with alternative map post-processing and an
optional split of the test set by a condition read from the file name (MVTec AD 2: regular, overexposed,
underexposed, shift).

Saved grids: galad/knn = fused smoothed patch map (sigma 1 on the grid), adino/subspace = raw patch distances.
Post-processing variants (pixel maps at 448 x 448):
  own       each method's own pipeline (galad/knn: bilinear zoom of the saved grid; adino/subspace:
            bilinear to 448 then Gaussian sigma = 4 px)
  up_s4     bilinear to 448, then Gaussian sigma = 4 px (applied to every method's saved grid)
  up_s8     bilinear to 448, then Gaussian sigma = 8 px
Writes output/map_explore_<tag>.csv (dataset, method, post, cond, n_train, seed, aupro_30, aupro_05, aupro_01).

Usage: uv run map_explore.py --tag ad2_maps --conditions
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--conditions", action="store_true", help="Also split by the MVTec AD 2 condition in the file name.")
p.add_argument("--prefix", default="", help="Only datasets whose name starts with this (e.g. mvtec_ad_2).")
p.add_argument("--own", action="store_true", help="Only each method's own post-processing.")
a = p.parse_args()
_argv, sys.argv = sys.argv, sys.argv[:1]
import galad_train_all as G  # noqa: E402  (parses its own arguments on import)
sys.argv = _argv
RES = 448


def cond_of(path):
    s = Path(path).stem
    return "shift" if "shift" in s else re.sub(r"^\d+_", "", s)


def post(grid, how, method):
    g = np.asarray(grid, np.float32)
    up = cv2.resize(g, (RES, RES), interpolation=cv2.INTER_LINEAR)
    if how == "own":
        return zoom(g, RES / g.shape[0], order=1) if method in ("galad", "knn") else gaussian_filter(up, sigma=4)
    return gaussian_filter(up, sigma=int(how[4:]))


rows = []
root = Path("output/server/scores") / a.tag
for d in sorted(root.iterdir()):
    ds = d.name.replace("__", "/")
    if not ds.startswith(a.prefix):
        continue
    _, info = G.load_dataset(ds)
    masks = []
    for e in info:
        m = G.load_mask(e)
        g = np.zeros((RES, RES), np.uint8) if m is None else (m > 0).astype(np.uint8)
        if g.shape != (RES, RES):  # same resizing as improve_experiments.masks_448 (server_eval's ground truth)
            g = (zoom(g.astype(float), (RES / g.shape[0], RES / g.shape[1]), order=0) > 0.5).astype(np.uint8)
        masks.append(g)
    conds = np.array([cond_of(e["path"]) for e in info]) if a.conditions else np.array(["all"] * len(info))
    groups = {"all": np.ones(len(info), bool)} | ({k: conds == k for k in np.unique(conds)} if a.conditions else {})
    regions = {k: mask_regions([masks[i] for i in np.flatnonzero(sel)]) for k, sel in groups.items()}
    for f in sorted(d.glob("maps_N*_s*_*.npz")):
        n, seed, method = re.fullmatch(r"maps_N(\d+)_s(\d+)_(\w+)\.npz", f.name).groups()
        grids = np.load(f)["maps"]
        for how in ("own",) if a.own else ("own", "up_s4", "up_s8"):
            maps = [post(g, how, method) for g in grids]
            for k, sel in groups.items():
                idx = np.flatnonzero(sel)
                pro = std_pixel_metrics_binned([maps[i] for i in idx], [masks[i] for i in idx], regions[k],
                                               limit=(0.3, 0.05, 0.01))[0]
                if pro is not None:
                    rows.append(dict(dataset=ds, method=method, post=how, cond=k, n_train=int(n), seed=int(seed),
                                     aupro_30=pro[0], aupro_05=pro[1], aupro_01=pro[2]))
    print(ds, len(rows), flush=True)
pd.DataFrame(rows).to_csv(f"output/map_explore_{a.tag}{'_' + a.prefix if a.prefix else ''}.csv", index=False)
