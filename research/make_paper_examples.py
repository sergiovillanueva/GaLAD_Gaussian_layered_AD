"""Illustrative anomaly maps where GaLAD localizes better (Figure 4 of the manuscript).

Selection rule (stated in the caption): the six categories with the largest AU-PRO gain of GaLAD over
AnomalyDINO (DINOv3-L) in output/server_run/l_square.csv; in each category (N = 2, draw 0),
the structural anomaly with the largest gain in per-image pixel AUROC. Maps are normalized as in
make_paper_qualitative.py (99th percentile of the method's maps on the normal test images).
Typical cases, including failures, are in figs/qualitative.pdf (Appendix).

Writes paper/electronics/figs/examples.pdf; maps cached in output/example_maps.npz.
Usage: uv run make_paper_examples.py
"""

import sys

sys.argv = [sys.argv[0], "--tag", "examples", "--datasets", "none", "--threads", "4"]
import server_eval as SE  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

E, G = SE.E, SE.G
BENCH = ["mvtec_AD", "VisA", "mvtec_loco_AD", "AutoVI", "btad", "GoodsAD"]
NAMES = {"mvtec_AD": "MVTec AD", "VisA": "VisA", "mvtec_loco_AD": "MVTec LOCO", "AutoVI": "AutoVI", "btad": "BTAD",
         "GoodsAD": "GoodsAD"}
N, SEED = 2, 0
CACHE = SE.Path("output/example_maps.npz")

L = pd.read_csv("output/server_run/l_square.csv")
pro = L.pivot_table(index="dataset", columns="method", values="aupro_30")
gain = (pro.galad - pro.adino).dropna()
CATS = list(gain.nlargest(6).index)
SE.log(f"categories: {CATS}")


def to_pixels(grids, kind, patch):
    if kind == "galad":
        return [SE.zoom(x, patch, order=1) for x in grids]
    return [SE.gaussian_filter(SE.cv2.resize(x.astype(np.float32), (x.shape[1] * patch, x.shape[0] * patch),
                                             interpolation=SE.cv2.INTER_LINEAR), sigma=4) for x in grids]


rows = []
if CACHE.exists():
    z = np.load(CACHE)
    rows = [(ds, z[f"{i}_img"], z[f"{i}_gt"], z[f"{i}_adino"], z[f"{i}_galad"]) for i, ds in enumerate(CATS)]
else:
    net = SE.Net("dinov3l")
    for ds in CATS:
        train_all, info = G.load_dataset(ds)
        masks = E.masks_448(info)
        labels = np.array([e["label"] for e in info])
        anom = [i for i in range(len(info)) if labels[i] == 1 and masks[i].any()
                and SE.Path(info[i]["path"]).parent.name != "logical_anomalies"]
        normals = [i for i in range(len(info)) if labels[i] == 0]
        idx = normals + anom
        test = net.extract([net.prep(Image.open(info[i]["path"]).convert("RGB")) for i in idx])
        g_views, a_views = SE.refs_for(net, E.sample_train(train_all, N, SEED))
        models, _ = SE.galad_fit(net.extract(g_views, dtype=torch.float32), net, SEED)
        _, g_grids = SE.galad_score(models, test, net)
        feats = net.extract(a_views, dtype=torch.float32)
        bank = F.normalize(torch.cat([f["last"] for f, _ in feats]).to(SE.DEVICE), dim=-1)
        _, a_grids = SE.adino_score(bank, test)
        px = {k: to_pixels(gr, k, net.patch) for k, gr in (("galad", g_grids), ("adino", a_grids))}
        ref = {k: np.percentile(np.concatenate([p.ravel() for p in v[:len(normals)]]), 99) for k, v in px.items()}
        best, best_gain = None, -np.inf
        for j, i in enumerate(anom):
            gt = masks[i].ravel() > 0
            au = {k: roc_auc_score(gt, px[k][len(normals) + j].ravel()) for k in px}
            if au["galad"] - au["adino"] > best_gain:
                best, best_gain = j, au["galad"] - au["adino"]
        i = anom[best]
        img = np.asarray(net.prep(Image.open(info[i]["path"]).convert("RGB")))
        out = {k: np.clip(px[k][len(normals) + best] / ref[k], 0, 2) for k in px}
        rows.append((ds, img, masks[i], out["adino"], out["galad"]))
        SE.log(f"{ds}: {info[i]['path']}, pixel AUROC gain {best_gain:+.3f}")
        del models, bank, feats, test
        torch.cuda.empty_cache()
    np.savez_compressed(CACHE, **{f"{i}_{k}": v for i, r in enumerate(rows) for k, v in zip(("img", "gt", "adino", "galad"), r[1:])})

fig, ax = plt.subplots(len(rows), 4, figsize=(7.2, 1.85 * len(rows)))
titles = ["Image", "Ground truth", "AnomalyDINO (DINOv3-L)", "GaLAD"]
for r, (ds, img, gt, am, gm) in enumerate(rows):
    for c, p in enumerate([img, gt, am, gm]):
        a = ax[r, c]
        if c == 0:
            a.imshow(p)
        elif c == 1:
            a.imshow(p, cmap="gray", vmin=0, vmax=1)
        else:
            im = a.imshow(p, cmap="inferno", vmin=0, vmax=2)
            a.contour(gt, levels=[0.5], colors="cyan", linewidths=0.6)
        a.set_xticks([])
        a.set_yticks([])
        if r == 0:
            a.set_title(titles[c], fontsize=8)
        if c == 0:
            a.set_ylabel(f"{NAMES[ds.split('/')[0]]}\n{ds.split('/')[1].replace('_', ' ')}", fontsize=7)
plt.tight_layout(pad=0.3, rect=(0, 0.035, 1, 1))
cax = fig.add_axes([0.52, 0.012, 0.45, 0.009])
fig.colorbar(im, cax=cax, orientation="horizontal")
cax.tick_params(labelsize=6)
cax.set_xlabel("map / 99th percentile of the maps of normal images", fontsize=6, labelpad=1)
plt.savefig("paper/electronics/figs/examples.pdf", dpi=200)
SE.log("wrote paper/electronics/figs/examples.pdf")
