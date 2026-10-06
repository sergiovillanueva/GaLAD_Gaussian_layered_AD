"""Qualitative anomaly maps for the manuscript (one GPU job, a few minutes).

Selection rule fixed before looking at any map (typical cases, not successes): in each benchmark,
the category whose GaLAD AU-PRO (mean over N and draws, output/server_run/l_square.csv) is closest
to the median over the categories of that benchmark; in that category, the structural anomaly
(logical anomalies of MVTec LOCO excluded) whose defect area is the median of the category. N = 2 references, draw 0, DINOv3-L. Each method's map is divided by the 99th
percentile of its own maps on the normal test images of the category, so colours are comparable
within a column and a normal image is mostly dark. Maps are cached in output/qualitative_maps.npz;
delete it to recompute.

Writes paper/electronics/figs/qualitative.pdf.
Usage: uv run make_paper_qualitative.py
"""

import sys

sys.argv = [sys.argv[0], "--tag", "qualitative", "--datasets", "none", "--threads", "4"]
import server_eval as SE  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

E, G = SE.E, SE.G
BENCH = ["mvtec_AD", "VisA", "mvtec_loco_AD", "AutoVI", "btad", "GoodsAD"]
_res = SE.csv.DictReader(open("output/server_run/l_square.csv"))
_pro = {}
for r in _res:
    if r["method"] == "galad" and r["aupro_30"]:
        _pro.setdefault(r["dataset"], []).append(float(r["aupro_30"]))
_pro = {c: np.mean(v) for c, v in _pro.items()}
CATS = []
for b in BENCH:
    vals = {c: v for c, v in _pro.items() if c.startswith(b + "/")}
    med = np.median(list(vals.values()))
    CATS.append(min(vals, key=lambda c: abs(vals[c] - med)))
NAMES = {"mvtec_AD": "MVTec AD", "VisA": "VisA", "mvtec_loco_AD": "MVTec LOCO", "AutoVI": "AutoVI", "btad": "BTAD",
         "GoodsAD": "GoodsAD"}
N, SEED = 2, 0


def to_pixels(grids, kind, patch):
    if kind == "galad":
        return [SE.zoom(x, patch, order=1) for x in grids]
    return [SE.gaussian_filter(SE.cv2.resize(x.astype(np.float32), (x.shape[1] * patch, x.shape[0] * patch),
                                             interpolation=SE.cv2.INTER_LINEAR), sigma=4) for x in grids]


CACHE = SE.Path("output/qualitative_maps.npz")
SE.log(f"selected categories: {CATS}")
rows = []
if CACHE.exists():
    z = np.load(CACHE)
    rows = [(ds, z[f"{i}_img"], z[f"{i}_gt"], z[f"{i}_adino"], z[f"{i}_galad"]) for i, ds in enumerate(CATS)]
net = SE.Net("dinov3l") if not rows else None
for ds in CATS if not rows else []:
    train_all, info = G.load_dataset(ds)
    masks = E.masks_448(info)
    labels = np.array([e["label"] for e in info])
    anom = [i for i in range(len(info)) if labels[i] == 1 and masks[i].any()
            and SE.Path(info[i]["path"]).parent.name != "logical_anomalies"]
    area = np.array([masks[i].mean() for i in anom])
    pick = anom[int(np.argsort(area)[len(area) // 2])]
    normals = [i for i in range(len(info)) if labels[i] == 0]
    idx = normals + [pick]
    test = net.extract([net.prep(Image.open(info[i]["path"]).convert("RGB")) for i in idx])
    g_views, a_views = SE.refs_for(net, E.sample_train(train_all, N, SEED))
    models, _ = SE.galad_fit(net.extract(g_views, dtype=torch.float32), net, SEED)
    _, g_grids = SE.galad_score(models, test, net)
    feats = net.extract(a_views, dtype=torch.float32)
    bank = F.normalize(torch.cat([f["last"] for f, _ in feats]).to(SE.DEVICE), dim=-1)
    _, a_grids = SE.adino_score(bank, test)
    out = {}
    for kind, grids in (("adino", a_grids), ("galad", g_grids)):
        px = to_pixels(grids, kind, net.patch)
        ref = np.percentile(np.concatenate([p.ravel() for p in px[:-1]]), 99)
        out[kind] = np.clip(px[-1] / ref, 0, 2)
    img = np.asarray(net.prep(Image.open(info[pick]["path"]).convert("RGB")))
    rows.append((ds, img, masks[pick], out["adino"], out["galad"]))
    SE.log(f"{ds}: image {info[pick]['path']}, defect area {100 * masks[pick].mean():.2f}%")
    del models, bank, feats, test
    torch.cuda.empty_cache()

if not CACHE.exists():
    np.savez_compressed(CACHE, **{f"{i}_{k}": v for i, r in enumerate(rows) for k, v in zip(("img", "gt", "adino", "galad"), r[1:])})
fig, ax = plt.subplots(len(rows), 4, figsize=(7.2, 1.85 * len(rows)))
titles = ["Image", "Ground truth", "AnomalyDINO (DINOv3-L)", "GaLAD"]
for r, (ds, img, gt, am, gm) in enumerate(rows):
    panels = [img, gt, am, gm]
    for c, p in enumerate(panels):
        a = ax[r, c]
        if c == 0:
            a.imshow(p)
        elif c == 1:
            a.imshow(p, cmap="gray", vmin=0, vmax=1)
        else:
            a.imshow(p, cmap="inferno", vmin=0, vmax=2)
            a.contour(gt, levels=[0.5], colors="cyan", linewidths=0.6)
        a.set_xticks([])
        a.set_yticks([])
        if r == 0:
            a.set_title(titles[c], fontsize=8)
        if c == 0:
            a.set_ylabel(f"{NAMES[ds.split('/')[0]]}\n{ds.split('/')[1].replace('_', ' ')}", fontsize=7)
plt.tight_layout(pad=0.3)
out = SE.Path("paper/electronics/figs/qualitative.pdf")
plt.savefig(out, dpi=200)
SE.log(f"wrote {out}")
