"""Per-image scores of AnomalyDINO on DINOv3 for threshold-transfer analyses (IMPROVEMENT_PLAN.md
section 27, step 4). Same method as improve_experiments.py (adino_rot): references rotated by
45-degree steps at original resolution, last DINOv3 layer, cosine 1-NN, mean of the top 1%.

Saved per (category, N, seed) in output/tune/scores_adino/<ds>/N{n}_s{seed}.npz:
  test    image scores of the test set, labels  test labels
  ref_loo leave-one-out scores of the unrotated references (each scored against the bank of the
          other references); for N=1 the image is scored against its own seven other rotations.
Also appends image AP/AUROC to output/tune/scores_adino.csv to check against adino_dinov3l_pix.csv.

Usage: uv run adino_scores.py --datasets mvtec_AD/screw --ns 1 2 4 5
"""

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--datasets", nargs="+", required=True)
p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 5])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()

_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "adino", "--threads", str(a.threads)]
import improve_experiments as E  # noqa: E402
sys.argv = _argv
G = E.G

import csv  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = Path("output/tune/scores_adino")
CSV = Path("output/tune/scores_adino.csv")
FIELDS = ["timestamp", "dataset", "n_train", "seed", "img_ap", "img_auroc"]


def score(bank, queries):
    return E.pool(E.knn_scores(bank, queries), "top1pct")


def main():
    bb = E.Backbone("dinov3l")
    for ds in a.datasets:
        try:
            d = OUT / ds.replace("/", "__")
            todo = [(n, s) for n in a.ns for s in a.seeds if not (d / f"N{n}_s{s}.npz").exists()]
            if not todo:
                continue
            d.mkdir(parents=True, exist_ok=True)
            train_all, test_info = G.load_dataset(ds)
            labels = np.array([int(t["label"]) for t in test_info])
            test = bb.extract([t["path"] for t in test_info], ["last"])["last"]
            for n, seed in todo:
                imgs = E.open_rgb(E.sample_train(train_all, n, seed))
                feats = bb.extract(E.augment(imgs, E.rot45), ["last"])["last"]  # [8n, P, D], 8 views per image
                s_test = score(feats.reshape(-1, feats.shape[-1]), test)
                loo = []
                for i in range(n):
                    own = feats[8 * i:8 * i + 8]
                    other = own[1:] if n == 1 else torch.cat([feats[:8 * i], feats[8 * i + 8:]])
                    loo.append(score(other.reshape(-1, feats.shape[-1]), own[:1])[0])
                np.savez_compressed(d / f"N{n}_s{seed}.npz", test=s_test.astype(np.float32), labels=labels,
                                    ref_loo=np.array(loo, np.float32))
                au, ap = E.metrics(labels, s_test)
                new = not CSV.exists()
                with open(CSV, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, n_train=n,
                                    seed=seed, img_ap=f"{ap:.4f}", img_auroc=f"{au:.4f}"))
                del feats
                torch.cuda.empty_cache()
            E.log(f"  adino scores {ds}: {len(todo)} units")
            del test
            torch.cuda.empty_cache()
        except Exception as e:  # log and continue; rerunning fills the gap
            E.log(f"ERROR {ds}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
