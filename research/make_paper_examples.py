"""Anomaly maps where GaLAD localizes better (Figure "examples" of the manuscript), against the three comparators.

Selection rule (stated in the caption):
  categories: the six with the largest AU-PRO gain of GaLAD over the better of the two AnomalyDINO versions
  (official and DINOv3-L; mean over N and draws, 55 paper categories), at most one per benchmark;
  image: N = 2, draw 0; among the structural anomalies whose GaLAD map peaks inside the defect, the one with the
  largest per-image pixel AP of GaLAD minus the best per-image pixel AP of the three comparators.
Maps: GaLAD and AnomalyDINO (DINOv3-L) from server_eval.py, AnomalyDINO (official) from adino_official.py, PatchCore
from Anomalib (server/patchcore_eval.py --ns 2 --seeds 0 --dump-maps output/fig_maps/patchcore). Each map is divided
by the 99th percentile of the same method's maps on the normal test images of the category.
Typical cases, including failures, are in figs/qualitative.pdf (Appendix).

Writes paper/electronics/figs/examples.pdf; maps cached in output/example_maps_v3.npz.
Usage: GAUSS_MAX_ITER=30 GAUSS_SKIP=0 uv run make_paper_examples.py
"""

import os
import sys

sys.argv = [sys.argv[0], "--tag", "examples", "--datasets", "none", "--threads", "4"]
import server_eval as SE  # noqa: E402

sys.argv = [sys.argv[0], "--datasets", "none"]
import adino_official as AO  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

E, G = SE.E, SE.G
BENCH = ["mvtec_AD", "VisA", "AutoVI", "btad", "GoodsAD", "mvtec_loco_AD", "mvtec_ad_2"]
NAMES = {"mvtec_AD": "MVTec AD", "VisA": "VisA", "mvtec_loco_AD": "MVTec LOCO", "AutoVI": "AutoVI", "btad": "BTAD",
         "GoodsAD": "GoodsAD", "mvtec_ad_2": "MVTec AD 2"}
METHODS = ["patchcore", "official", "adino", "galad"]
TITLES = ["Image and defect", "PatchCore", "AnomalyDINO", "AnomalyDINO (DINOv3-L)", "GaLAD"]
N, SEED = 2, 0
CACHE = SE.Path("output/example_maps_v3.npz")
PCDIR = SE.Path("output/fig_maps/patchcore")

Fin, Lsq, New = (pd.read_csv(f) for f in ("output/server/final_square.csv", "output/server_run/l_square.csv",
                                          "output/server/new_square.csv"))
Off = pd.read_csv("output/adino_official.csv").rename(columns={"aupro_std": "aupro_30"})
g = Fin[Fin.method == "galad"].groupby("dataset").aupro_30.mean()
a = pd.concat([Lsq[Lsq.method == "adino"], New[New.method == "adino"]]).groupby("dataset").aupro_30.mean()
o = Off.groupby("dataset").aupro_30.mean()
gain = (g - pd.concat([a, o], axis=1).max(axis=1)).dropna()
gain = gain[[c.split("/")[0] in BENCH for c in gain.index]].sort_values(ascending=False)
CATS = list(gain.groupby(lambda c: c.split("/")[0]).head(1).head(6).index)
SE.log(f"categories: {CATS}")


def up(grids, patch, kind):
    """Pixel maps at 448 x 448: GaLAD bilinear zoom of the smoothed grid; AnomalyDINO bilinear + Gaussian 4 px."""
    if kind == "galad":
        return [SE.zoom(x, patch, order=1) for x in grids]
    return [SE.gaussian_filter(SE.cv2.resize(x.astype(np.float32), (448, 448), interpolation=SE.cv2.INTER_LINEAR),
                               sigma=4) for x in grids]


def official_maps(ds, info, train_all, net2):
    bank = [net2.tokens(r)[0] for img in E.open_rgb(E.sample_train(train_all, N, SEED)) for r in AO.rotations(img)]
    bank = F.normalize(torch.from_numpy(np.concatenate(bank)).to(AO.DEV), dim=-1)
    out = []
    for e in info:
        tok, grid = net2.tokens(Image.open(e["path"]).convert("RGB"))
        keep = AO.background_mask(tok, grid) if AO.masking_for(ds) else np.ones(tok.shape[0], bool)
        q = F.normalize(torch.from_numpy(tok).to(AO.DEV), dim=-1)
        d = (1 - (q @ bank.T).max(1).values).cpu().numpy()
        d[~keep] = 0.0
        out.append(SE.gaussian_filter(SE.cv2.resize(d.reshape(grid).astype(np.float32), (448, 448),
                                                    interpolation=SE.cv2.INTER_LINEAR), sigma=4))
    return out


def patchcore_maps(ds, info):
    z = np.load(PCDIR / f"{ds.replace('/', '__')}_N{N}_s{SEED}.npz")
    by = {os.path.normcase(os.path.abspath(p)): m for p, m in zip(z["paths"], z["maps"])}
    return [by[os.path.normcase(os.path.abspath(e["path"]))].astype(np.float32) for e in info]


rows = []
if CACHE.exists():
    z = np.load(CACHE)
    rows = [(ds, z[f"{i}_img"], z[f"{i}_gt"], [z[f"{i}_{m}"] for m in METHODS]) for i, ds in enumerate(CATS)]
else:
    net, net2 = SE.Net("dinov3l"), AO.DinoV2S()
    for ds in CATS:
        train_all, info = G.load_dataset(ds)
        masks = E.masks_448(info)
        labels = np.array([e["label"] for e in info])
        test = net.extract([net.prep(Image.open(e["path"]).convert("RGB")) for e in info])
        g_views, a_views = SE.refs_for(net, E.sample_train(train_all, N, SEED))
        models, _ = SE.galad_fit(net.extract(g_views, dtype=torch.float32), net, SEED)
        _, g_grids, _ = SE.galad_score(models, test, net)
        feats = net.extract(a_views, dtype=torch.float32)
        bank = F.normalize(torch.cat([f["last"] for f, _ in feats]).to(SE.DEVICE), dim=-1)
        _, a_grids = SE.adino_score(bank, test)
        px = {"galad": up(g_grids, net.patch, "galad"), "adino": up(a_grids, net.patch, "adino"),
              "official": official_maps(ds, info, train_all, net2), "patchcore": patchcore_maps(ds, info)}
        normals = np.flatnonzero(labels == 0)
        ref = {k: np.percentile(np.concatenate([v[i].ravel() for i in normals]), 99) for k, v in px.items()}
        anom = [i for i in range(len(info)) if labels[i] == 1 and masks[i].any()
                and SE.Path(info[i]["path"]).parent.name != "logical_anomalies"]
        best, best_gain = None, -np.inf
        for i in anom:
            gt = masks[i] > 0
            peak = np.unravel_index(np.argmax(px["galad"][i]), gt.shape)
            if not binary_dilation(gt, iterations=8)[peak]:
                continue
            ap = {k: average_precision_score(gt.ravel(), px[k][i].ravel()) for k in px}
            d = ap["galad"] - max(ap[k] for k in METHODS[:3])
            if d > best_gain:
                best, best_gain = i, d
        img = np.asarray(net.prep(Image.open(info[best]["path"]).convert("RGB")))
        rows.append((ds, img, masks[best], [np.clip(px[k][best] / ref[k], 0, 2) for k in METHODS]))
        SE.log(f"{ds}: {info[best]['path']}, pixel AP gain over the best comparator {best_gain:+.3f}")
        del models, bank, feats, test
        torch.cuda.empty_cache()
    np.savez_compressed(CACHE, **{f"{i}_{k}": v for i, (_, img, gt, maps) in enumerate(rows)
                                  for k, v in zip(["img", "gt"] + METHODS, [img, gt] + maps)})

fig, ax = plt.subplots(len(rows), 5, figsize=(7.2, 1.5 * len(rows) + 0.35))
for r, (ds, img, gt, maps) in enumerate(rows):
    n, lab, stats, cent = SE.cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    tiny = [cent[j] for j in range(1, n) if stats[j, SE.cv2.CC_STAT_AREA] < 400]  # circled so they stay visible
    for c, p in enumerate([img] + maps):
        x = ax[r, c]
        if c == 0:
            x.imshow(p)
            x.contour(gt, levels=[0.5], colors="#00e5ff", linewidths=0.8)
        else:
            im = x.imshow(p, cmap="inferno", vmin=0, vmax=2)
            x.contour(gt, levels=[0.5], colors="#00e5ff", linewidths=0.6)
        for cx, cy in tiny:
            x.add_patch(plt.Circle((cx, cy), 26, fill=False, color="#00e5ff", lw=0.7, ls="--"))
        x.set_xticks([])
        x.set_yticks([])
        if r == 0:
            x.set_title(TITLES[c], fontsize=7.5, fontweight="bold" if c == 4 else "normal")
        if c == 0:
            x.set_ylabel(f"{NAMES[ds.split('/')[0]]}\n{ds.split('/')[1].replace('_', ' ')}", fontsize=7)
plt.tight_layout(pad=0.3, rect=(0, 0.04, 1, 1))
cax = fig.add_axes([0.42, 0.014, 0.55, 0.009])
fig.colorbar(im, cax=cax, orientation="horizontal")
cax.tick_params(labelsize=6)
cax.set_xlabel("map / 99th percentile of the same method's maps on normal images", fontsize=6, labelpad=1)
plt.savefig("paper/electronics/figs/examples.pdf", dpi=200, bbox_inches="tight", pad_inches=0.03)
SE.log("wrote paper/electronics/figs/examples.pdf")
