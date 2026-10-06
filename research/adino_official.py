"""AnomalyDINO exactly as published (IMPROVEMENT_PLAN.md, section 24), under our protocol.

Follows the official code (github.com/dammsi/AnomalyDINO, read, not executed):
  backbone DINOv2 ViT-S/14, last layer (normalized), smaller image edge resized to 448 (bicubic,
  aspect ratio kept) and cropped to a multiple of 14; references augmented with eight 45-degree
  rotations (agnostic preprocessing, rotated at original resolution, reflected borders);
  test-image background masking for the categories listed in its utils.py (VisA: all; MVTec AD:
  capsule, hazelnut, pill, screw, toothbrush): PCA(1) on the image's own tokens, first PC > 10,
  adaptive flip if the central crop keeps <= 35%, 3x3 dilation + closing; masked patches get
  distance 0; k=1 cosine NN; image score = mean of the top 1% (floor); maps = bilinear resize to
  the evaluation size (448x448, as for every method here) + Gaussian sigma=4.
Same reference draws (seeds) as all other experiments.

Usage: uv run adino_official.py --datasets mvtec_AD/screw --ns 1 2 4 5
       --no-mask: same pipeline without masking (separates the mask from backbone/resize effects).
"""

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--datasets", nargs="+", required=True)
p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 5])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--threads", type=int, default=4)
p.add_argument("--no-mask", action="store_true", help="Disable background masking everywhere (masking=0 rows, separate CSV).")
a = p.parse_args()

_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "adino", "--backbone", "dinov2s", "--threads", str(a.threads)]
import improve_experiments as E  # noqa: E402
sys.argv = _argv
G = E.G

import csv  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from transformers import AutoModel  # noqa: E402

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned  # noqa: E402

OUT = Path("output/adino_official_nomask.csv" if a.no_mask else "output/adino_official.csv")
FIELDS = ["timestamp", "dataset", "n_train", "seed", "masking", "img_ap", "img_auroc",
          "aupro_std", "pix_auroc", "pix_ap"]
MASKED_MVTEC = {"capsule", "hazelnut", "pill", "screw", "toothbrush"}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
DEV = torch.device("cuda")


def masking_for(ds):
    if a.no_mask:
        return False
    bench, cat = ds.split("/")
    return bench == "VisA" or (bench == "mvtec_AD" and cat in MASKED_MVTEC)


class DinoV2S:
    def __init__(self):
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
        self.patch = 14

    def tokens(self, img):
        """img: PIL RGB -> (tokens [h*w, 384] numpy float32, grid (h, w))."""
        w, h = img.size
        s = 448 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        H, W = x.shape[0] - x.shape[0] % self.patch, x.shape[1] - x.shape[1] % self.patch
        x = torch.from_numpy(x[:H, :W].transpose(2, 0, 1).copy()).float().unsqueeze(0).to(DEV)
        with torch.inference_mode():
            t = self.model(pixel_values=x).last_hidden_state[0, 1:]
        return t.float().cpu().numpy(), (H // self.patch, W // self.patch)


def background_mask(tok, grid, threshold=10, kernel=3, border=0.2):
    pc = PCA(n_components=1, svd_solver="randomized").fit_transform(tok.astype(np.float32))
    mask = pc > threshold
    m = mask.reshape(grid)[int(grid[0] * border):int(grid[0] * (1 - border)), int(grid[1] * border):int(grid[1] * (1 - border))]
    if m.sum() <= m.size * 0.35:
        mask = -pc > threshold
    k = np.ones((kernel, kernel), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8).reshape(grid), k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k).astype(bool)
    return mask.ravel()


def rotations(img):
    a = np.asarray(img)
    h, w = a.shape[:2]
    out = []
    for angle in range(0, 360, 45):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(Image.fromarray(cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)))
    return out


def done():
    if not OUT.exists():
        return set()
    with open(OUT, newline="") as f:
        return {(r["dataset"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}


def main():
    net = DinoV2S()
    fin = done()
    for ds in a.datasets:
        try:
            todo = [(n, s) for n in a.ns for s in a.seeds if (ds, n, s) not in fin]
            if not todo:
                continue
            train_all, test_info = G.load_dataset(ds)
            labels = np.array([t["label"] for t in test_info])
            masks = E.masks_448(test_info)
            regions = mask_regions(masks)
            use_mask = masking_for(ds)
            test = []
            for t in test_info:
                tok, grid = net.tokens(Image.open(t["path"]).convert("RGB"))
                keep = background_mask(tok, grid) if use_mask else np.ones(tok.shape[0], bool)
                test.append((torch.nn.functional.normalize(torch.from_numpy(tok).to(DEV), dim=-1), grid, keep))
            for n, seed in todo:
                bank = [net.tokens(r)[0] for img in E.open_rgb(E.sample_train(train_all, n, seed)) for r in rotations(img)]
                bank = torch.nn.functional.normalize(torch.from_numpy(np.concatenate(bank)).to(DEV), dim=-1)
                scores, maps = [], []
                for q, grid, keep in test:
                    d = (1 - (q @ bank.T).max(1).values).cpu().numpy()
                    d[~keep] = 0.0
                    k = max(1, int(0.01 * d.size))
                    scores.append(np.sort(d)[-k:].mean())
                    m = cv2.resize(d.reshape(grid).astype(np.float32), (E.RES, E.RES), interpolation=cv2.INTER_LINEAR)
                    maps.append(gaussian_filter(m, sigma=4))
                au, ap = E.metrics(labels, np.array(scores))
                aupro, pau, pap = std_pixel_metrics_binned(maps, masks, regions)
                row = dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, n_train=n, seed=seed,
                           masking=int(use_mask), img_ap=f"{ap:.4f}", img_auroc=f"{au:.4f}", aupro_std=E.fmt(aupro),
                           pix_auroc=E.fmt(pau), pix_ap=E.fmt(pap))
                new = not OUT.exists()
                with open(OUT, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                E.log(f"  official {ds} N={n} seed {seed}: AP {row['img_ap']} AU-PRO {row['aupro_std']}")
                del bank
                torch.cuda.empty_cache()
        except Exception as e:
            E.log(f"ERROR {ds}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
