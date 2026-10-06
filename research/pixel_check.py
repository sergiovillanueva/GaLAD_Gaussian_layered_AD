"""Controlled localization (and detection) comparison, IMPROVEMENT_PLAN.md section 18.

Per (category, N, seed), same reference draw for every method:
  galad_none   GaLAD, no augmentation
  galad_rot8   GaLAD with the locked 45-degree rotations (resize to 448, then rotate)
  knn_galad    nearest neighbour on GaLAD's own representation: same layers, same PCA (from the
               rot8 fit), same rot8 references, same smoothing and pooling; only the scorer differs
  adino        AnomalyDINO as re-implemented (rotate original image, then the backbone resizes)
  adino_sq     AnomalyDINO with GaLAD's reference transform (resize to 448, then rotate)
Map post-processing (fixed before results): 'own' (each method's original pipeline) and the same
for all methods on the raw patch grid: bilinear resize to 448, then Gaussian sigma in {0, 4, 16}.
Pixel metrics: paper's custom AUsPRO ('own' only), standard AU-PRO@0.3, pixel AUROC, pixel AP.
Image metrics (each method's own scoring): image AP and AUROC. Raw grids saved to output/maps/.

Usage: uv run pixel_check.py --datasets VisA/candle --ns 2 5
"""

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--datasets", nargs="+", required=True)
p.add_argument("--ns", nargs="+", type=int, default=[2, 5])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()

_argv, sys.argv = sys.argv, [sys.argv[0], "--phase", "screen", "--threads", str(a.threads), "--no-tta"]
import tune_experiments as TE  # noqa: E402
sys.argv = _argv
E, G = TE.E, TE.G

import csv  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned  # noqa: E402

OUT = Path("output/pixel_check.csv")
MAPS = Path("output/maps")
FIELDS = ["timestamp", "dataset", "n_train", "seed", "method", "post", "img_ap", "img_auroc",
          "auspro_custom", "aupro_std", "pix_auroc", "pix_ap"]
RES, SIGMAS = E.RES, (0, 4, 16)


def done():
    if not OUT.exists():
        return set()
    with open(OUT, newline="") as f:
        return {(r["dataset"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}


def save(rows):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def bilinear(grid):
    return cv2.resize(grid.astype(np.float32), (RES, RES), interpolation=cv2.INTER_LINEAR)


def post_maps(layers, post, method):
    """layers: [L, n, g, g] raw per-layer grids (L=1 for AnomalyDINO). Returns list of RES maps."""
    if post == "own" and method.startswith("adino"):
        return [gaussian_filter(bilinear(x), sigma=4) for x in layers[0]]          # dists2map
    if post == "own":
        fused = np.mean([np.stack([gaussian_filter(x, sigma=1) for x in l]) for l in layers], axis=0)
        return [zoom(x, RES / x.shape[0], order=1) for x in fused]                 # paper's GaLAD maps
    sigma = int(post[1:])
    fused = np.mean(layers, axis=0)
    return [gaussian_filter(bilinear(x), sigma=sigma) if sigma else bilinear(x) for x in fused]


def image_score(layers, method):
    if method.startswith("adino"):
        return E.pool(layers[0], "top1pct")
    return np.mean([E.pool(np.stack([gaussian_filter(x, sigma=1) for x in l]), "p95") for l in layers], axis=0)


def galad_layers(bb, refs, fit, cache, seed, want_knn=False):
    cfg = TE.cfg_of(fit)
    tr = bb.extract(TE.ref_images(refs, cfg["aug"]), list(TE.LAYERS))
    gl, kl = [], []
    for l in cfg["layers"]:
        X = TE.preprocess(tr[l].to(TE.DEV_T), cfg, bb.grid)
        m = TE.LayerModel(X, cfg, seed)
        bank = m.project(X).float() if want_knn else None
        del X
        x = cache[l]
        nll, nn = [], []
        for i in range(0, x.shape[0], 32):
            q = TE.preprocess(torch.from_numpy(np.ascontiguousarray(x[i:i + 32])).to(TE.DEV_T), cfg, bb.grid)
            nll.append(m.nll(q))
            if want_knn:
                z = m.project(q).float()
                nn.append(torch.cat([torch.cdist(z[j:j + 4096], bank).min(1).values
                                     for j in range(0, z.shape[0], 4096)]))
        n = x.shape[0]
        gl.append(torch.cat(nll).reshape(n, bb.grid, bb.grid).float().cpu().numpy())
        if want_knn:
            kl.append(torch.cat(nn).reshape(n, bb.grid, bb.grid).cpu().numpy())
        del m, bank
        torch.cuda.empty_cache()
    return np.stack(gl), (np.stack(kl) if want_knn else None)


def main():
    bb = E.Backbone("dinov3l")
    finished = done()
    for ds in a.datasets:
        try:
            todo = [(n, s) for n in a.ns for s in a.seeds if (ds, n, s) not in finished]
            if not todo:
                continue
            train_all, test_info = G.load_dataset(ds)
            if not any(e.get("mask_path") or e.get("mask_dir") for e in test_info):
                E.log(f"skip {ds}: no pixel ground truth")
                continue
            masks = E.masks_448(test_info)
            regions = mask_regions(masks)
            labels = np.array([int(t.get("label", 0)) for t in test_info])
            cache = {l: np.load(TE.cdir(ds) / f"test_orig_L{l}.npy", mmap_mode="r") for l in TE.LAYERS}
            last = bb.extract([t["path"] for t in test_info], ["last"])["last"]
            (MAPS / ds.replace("/", "__")).mkdir(parents=True, exist_ok=True)
            for n, seed in todo:
                refs = E.sample_train(train_all, n, seed)
                layers = {}
                layers["galad_none"], _ = galad_layers(bb, refs, "none", cache, seed)
                layers["galad_rot8"], layers["knn_galad"] = galad_layers(bb, refs, "rot8", cache, seed, want_knn=True)
                for name, imgs in (("adino", E.augment(E.open_rgb(refs), E.rot45)),
                                   ("adino_sq", TE.ref_images(refs, "rot8"))):
                    ref = bb.extract(imgs, ["last"])["last"].reshape(-1, last.shape[-1])
                    layers[name] = E.knn_scores(ref, last).reshape(1, -1, bb.grid, bb.grid)
                np.savez_compressed(MAPS / ds.replace("/", "__") / f"N{n}_s{seed}.npz",
                                    **{k: v.astype(np.float32) for k, v in layers.items()}, labels=labels)
                rows = []
                for method, lay in layers.items():
                    iau, iap = E.metrics(labels, image_score(lay, method))
                    posts = ["own", "s0", "s4", "s16"] if method in ("galad_rot8", "adino") else ["own", "s4"]
                    for post in posts:
                        maps = post_maps(lay, post, method)
                        aupro, pau, pap = std_pixel_metrics_binned(maps, masks, regions)
                        custom = E.fast_auspro(maps, masks, labels)[1] if post == "own" else None
                        rows.append(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, n_train=n,
                                         seed=seed, method=method, post=post, img_ap=f"{iap:.4f}",
                                         img_auroc=f"{iau:.4f}", auspro_custom=E.fmt(custom), aupro_std=E.fmt(aupro),
                                         pix_auroc=E.fmt(pau), pix_ap=E.fmt(pap)))
                save(rows)
                own = {r["method"]: r["aupro_std"] for r in rows if r["post"] == "own"}
                E.log(f"  pixel_check {ds} N={n} seed {seed}: std AU-PRO own " + ", ".join(f"{k} {v}" for k, v in own.items()))
            del last
            torch.cuda.empty_cache()

        except Exception as e:  # log and continue; rerunning fills the gap
            E.log(f"ERROR {ds}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
