"""Evaluate GaLAD and AnomalyDINO on the same DINOv3 backbone, as in the paper (Tables 2-4, 6).

For every category, N and draw: image AUROC and AP, AU-PRO up to FPR 0.3 and 0.05, pixel AUROC and AP.
Rows are appended to results/reproduced/<tag>.csv (finished units are skipped, so runs can be resumed).

Usage:
  python scripts/run_eval.py --datasets VisA/pcb1 btad/02 --ns 1 2 4 5 --seeds 0 1 2 3 4
  python scripts/run_eval.py --all            # the 47 categories of the paper
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galad import AnomalyDINO, GaLAD  # noqa: E402
from galad.data import load_dataset, masks_448, sample_references  # noqa: E402
from galad.features import Backbone  # noqa: E402
from galad.metrics import image_metrics, mask_regions, std_pixel_metrics_binned  # noqa: E402
from PIL import Image  # noqa: E402

CATEGORIES = {
    "mvtec_AD": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
                 "tile", "toothbrush", "transistor", "wood", "zipper"],
    "VisA": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3",
             "pcb4", "pipe_fryum"],
    "mvtec_loco_AD": ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"],
    "AutoVI": ["engine_wiring", "pipe_clip", "pipe_staple", "tank_screw", "underbody_pipes", "underbody_screw"],
    "btad": ["01", "02", "03"],
    "GoodsAD": ["cigarette_box", "drink_bottle", "drink_can", "food_bottle", "food_box", "food_package"],
}
FIELDS = ["dataset", "method", "n_train", "seed", "img_ap", "img_auroc", "aupro_30", "aupro_05", "pix_auroc", "pix_ap"]

p = argparse.ArgumentParser()
p.add_argument("--datasets", nargs="*", default=[])
p.add_argument("--all", action="store_true")
p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 5])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--backbone", default="dinov3l")
p.add_argument("--data", default="data")
p.add_argument("--tag", default="l_square")
a = p.parse_args()
datasets = [f"{b}/{c}" for b, cs in CATEGORIES.items() for c in cs] if a.all else a.datasets

out = Path("results/reproduced") / f"{a.tag}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    with open(out, newline="") as f:
        done = {(r["dataset"], r["method"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}

net = Backbone(a.backbone)
for ds in datasets:
    todo = [(n, s) for n in a.ns for s in a.seeds if any((ds, m, n, s) not in done for m in ("galad", "adino"))]
    if not todo:
        continue
    train, info = load_dataset(ds, a.data)
    labels = np.array([e["label"] for e in info])
    test = net.extract([net.prep(Image.open(e["path"]).convert("RGB")) for e in info])
    gts = masks_448(info)
    regions = mask_regions(gts)
    for n, seed in todo:
        g_refs, a_refs = net.references(sample_references(train, n, seed))
        for name, model, refs in (("galad", GaLAD(net.layers, seed=seed), g_refs), ("adino", AnomalyDINO(), a_refs)):
            scores, maps = model.fit(refs).score(test, net.patch)
            auroc, ap = image_metrics(labels, scores)
            (pro30, pro05), pau, pap = std_pixel_metrics_binned(maps, gts, regions, limit=(0.3, 0.05))
            row = dict(dataset=ds, method=name, n_train=n, seed=seed, img_ap=f"{ap:.4f}", img_auroc=f"{auroc:.4f}",
                       aupro_30=f"{pro30:.4f}", aupro_05=f"{pro05:.4f}", pix_auroc=f"{pau:.4f}", pix_ap=f"{pap:.4f}")
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"{ds} N={n} s{seed} {name}: AP {ap:.4f} AUROC {auroc:.4f} AU-PRO {pro30:.4f}", flush=True)
            torch.cuda.empty_cache()
