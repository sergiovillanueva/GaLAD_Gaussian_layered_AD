"""AU-PRO of the ablation variants, from the raw chi-square patch maps saved by gauss_screen.py --save-maps.

Post-processing as in server_eval.py: each layer's grid smoothed (sigma in patches), mean of the two layers,
bilinear zoom to 448 x 448; masks resized as server_eval's ground truth. Variants: every fit with the final
post-processing; for the final fit (k12s2000) also sigma 0 and 2 and each layer alone.
Usage: uv run ablation_aupro.py   -> output/tune/ablation_aupro.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned

_argv, sys.argv = sys.argv, sys.argv[:1]
import galad_train_all as G  # noqa: E402  (parses its own arguments on import)
sys.argv = _argv
RES, PATCH = 448, 16
rows = []
for d in sorted(Path("output/tune/maps").iterdir()):
    ds = d.name.replace("__", "/")
    _, info = G.load_dataset(ds)
    masks = []
    for e in info:
        m = G.load_mask(e)
        g = np.zeros((RES, RES), np.uint8) if m is None else (m > 0).astype(np.uint8)
        if g.shape != (RES, RES):
            g = (zoom(g.astype(float), (RES / g.shape[0], RES / g.shape[1]), order=0) > 0.5).astype(np.uint8)
        masks.append(g)
    regions = mask_regions(masks)
    for f in sorted(d.glob("N*_s*_*.npz")):
        n, seed, fit = re.fullmatch(r"N(\d+)_s(\d+)_(\w+)\.npz", f.name).groups()
        z = np.load(f)
        t6, t12 = z["t6"], z["t12"]
        assert len(t6) == len(info), (ds, len(t6), len(info))
        variants = {"final": ((t6, t12), 1)}
        if fit == "k12s2000":
            variants.update({"sig0": ((t6, t12), 0), "sig2": ((t6, t12), 2), "L6": ((t6,), 1), "L12": ((t12,), 1)})
        for name, (layers, sg) in variants.items():
            maps = [zoom(np.mean([gaussian_filter(l[i].astype(np.float64), sg) if sg else l[i] for l in layers], 0),
                         PATCH, order=1) for i in range(len(info))]
            pro = std_pixel_metrics_binned(maps, masks, regions, limit=(0.3, 0.05))[0]
            rows.append(dict(dataset=ds, fit=fit, variant=name, n_train=int(n), seed=int(seed),
                             aupro_30=pro[0], aupro_05=pro[1]))
    print(ds, len(rows), flush=True)
    pd.DataFrame(rows).to_csv("output/tune/ablation_aupro.csv", index=False)
