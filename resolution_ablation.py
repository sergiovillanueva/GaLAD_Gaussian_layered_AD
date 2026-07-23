"""
Resolution ablation for GaLAD (DINOv3 ViT-L/16).

Runs the full GaLAD pipeline at several input resolutions to test sensitivity
to small defects (higher resolution -> finer patch grid). Everything else is
kept at the default configuration (layers [-6,-12], PCA d=256 skip=2, GMM K=3,
sigma=1, percentile 95).

Resolutions: 336 (grid 21x21), 448 (grid 28x28, default), 672 (grid 42x42).

Design: resumable, robust (per-experiment error logging), same seeds and
train-sampling as the rest of the project.

Usage:
    uv run resolution_ablation.py --test     # 1 category
    uv run resolution_ablation.py            # 8 ablation categories
    uv run resolution_ablation.py --device 0
"""

import argparse
import os
import csv
import gc
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--test", action="store_true")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
import transformers
from transformers import AutoModel, AutoImageProcessor
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
RESOLUTIONS = [336, 448, 672]
LAYERS = [-6, -12]
PCA_DIM = 256
PCA_SKIP = 2
N_COMPONENTS = 3
SMOOTH_SIGMA = 1.0
BATCH_SIZE = 8

ABLATION_DATASETS = [
    "mvtec_AD/bottle", "mvtec_AD/carpet", "mvtec_AD/hazelnut", "mvtec_AD/leather",
    "VisA/capsules", "VisA/pcb1",
    "GoodsAD/drink_bottle", "GoodsAD/food_box",
]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_LIMIT = 10

CSV_PATH = Path("results/resolution_ablation.csv")
LOG_PATH = Path("results/resolution_ablation_log.txt")
ERR_PATH = Path("results/resolution_ablation_errors.txt")
CSV_FIELDS = ["resolution", "dataset", "seed", "train_limit", "grid", "img_auroc", "img_aupr"]


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(resolution, dataset, seed, exc):
    ERR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] res={resolution} | {dataset} | seed={seed}\n")
        f.write(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}" + "=" * 70 + "\n")


class ResGaLAD:
    """GaLAD with DINOv3 at a configurable resolution. Backbone cached once."""
    _model = None
    _processor = None

    def __init__(self, resolution: int, seed: int = 42):
        self.resolution = resolution
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if ResGaLAD._model is None:
            log(f"Loading DINOv3 ({MODEL_NAME})...")
            ResGaLAD._processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            ResGaLAD._model = AutoModel.from_pretrained(
                MODEL_NAME, output_hidden_states=True).to(self.device).eval()
        self.processor, self.model = ResGaLAD._processor, ResGaLAD._model
        self.patch_size = self.model.config.patch_size
        self.num_reg = getattr(self.model.config, "num_register_tokens", 0)
        self.grid_size = resolution // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.start_idx = 1 + self.num_reg
        self.pca_models, self.gmm_models = {}, {}

    def _extract(self, paths: List[str]):
        feats = []
        for i in range(0, len(paths), BATCH_SIZE):
            batch = paths[i:i + BATCH_SIZE]
            imgs = []
            for p in batch:
                im = Image.open(p)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                imgs.append(im.resize((self.resolution, self.resolution), Image.BICUBIC))
            inputs = self.processor(images=imgs, return_tensors="pt",
                                    do_resize=False, do_center_crop=False).to(self.device)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    out = self.model(**inputs)
                hs = out.hidden_states
                for b in range(len(batch)):
                    feats.append({li: hs[li][b][self.start_idx:self.start_idx + self.num_patches].float().cpu().numpy()
                                  for li in LAYERS})
        return feats

    def fit(self, train_paths: List[str]):
        feats = self._extract(train_paths)
        for li in LAYERS:
            data = np.concatenate([f[li] for f in feats], axis=0)
            total = min(PCA_DIM + PCA_SKIP, data.shape[0], data.shape[1])
            pca = PCA(n_components=total, random_state=self.seed)
            proj = pca.fit_transform(data.astype(np.float32))[:, PCA_SKIP:]
            self.pca_models[li] = pca
            ns, nf = proj.shape
            n_comp = min(N_COMPONENTS, max(2, ns // (nf + 10)))
            gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                                  reg_covar=1e-2, random_state=self.seed, n_init=1, max_iter=200)
            gmm.fit(proj)
            self.gmm_models[li] = gmm
        return self

    def score(self, paths: List[str]) -> List[float]:
        feats = self._extract(paths)
        out = []
        for f in feats:
            ls = []
            for li in LAYERS:
                proj = self.pca_models[li].transform(f[li].astype(np.float32))[:, PCA_SKIP:]
                nll = -self.gmm_models[li].score_samples(proj)
                amap = nll.reshape(self.grid_size, self.grid_size)
                if SMOOTH_SIGMA > 0:
                    amap = gaussian_filter(amap, sigma=SMOOTH_SIGMA)
                ls.append(float(np.percentile(amap, 95)))
            out.append(float(np.mean(ls)))
        return out


def load_dataset(data_dir: str):
    if data_dir.startswith("VisA/"):
        import csv as _csv
        category = data_dir.split("/", 1)[1]
        visa_root = Path("data/VisA")
        split_csv = visa_root / "split_csv" / "1cls.csv"
        if not split_csv.exists():
            return [], []
        train, test = [], []
        with open(split_csv, newline="") as f:
            for row in _csv.DictReader(f):
                if row["object"] != category:
                    continue
                path = str(visa_root / row["image"])
                if row["split"] == "train":
                    train.append(path)
                elif row["split"] == "test":
                    test.append({"path": path, "label": 0 if row["label"] == "normal" else 1})
        return sorted(train), test
    data_path = Path("data") / data_dir
    train_dir, test_dir = data_path / "train" / "good", data_path / "test"
    if not train_dir.exists():
        return [], []
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    train = sorted(str(p) for p in train_dir.glob("*.*") if p.suffix.lower() in exts)
    test = []
    for cat in sorted(test_dir.iterdir()):
        if cat.is_dir():
            label = 0 if cat.name == "good" else 1
            for p in sorted(cat.glob("*.*")):
                if p.suffix.lower() in exts:
                    test.append({"path": str(p), "label": label})
    return train, test


def sample_train(train, n, seed):
    if len(train) <= n:
        return train
    return sorted(random.Random(seed).sample(sorted(train), n))


def load_completed():
    if not CSV_PATH.exists():
        return set()
    done = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            done.add((int(row["resolution"]), row["dataset"], int(row["seed"]), int(row["train_limit"])))
    return done


def save_row(row):
    header = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        if header:
            w.writeheader()
        w.writerow(row)


def main():
    datasets = ["mvtec_AD/bottle"] if args.test else ABLATION_DATASETS
    seeds = [0, 1] if args.test else SEEDS
    completed = load_completed()
    total = len(RESOLUTIONS) * len(datasets) * len(seeds)
    log(f"Resolution ablation: {len(RESOLUTIONS)} res x {len(datasets)} cat x {len(seeds)} seeds "
        f"= {total} ({len(completed)} done)")
    current = 0
    for res in RESOLUTIONS:
        for dataset in datasets:
            train_all, test_info = load_dataset(dataset)
            if not train_all or not test_info:
                log(f"SKIP {dataset}: not found")
                continue
            tp = [t["path"] for t in test_info]
            tl = [t["label"] for t in test_info]
            for seed in seeds:
                current += 1
                key = (res, dataset, seed, TRAIN_LIMIT)
                if key in completed:
                    continue
                try:
                    m = ResGaLAD(res, seed=seed).fit(sample_train(train_all, TRAIN_LIMIT, seed))
                    sc = m.score(tp)
                    auroc = roc_auc_score(tl, sc)
                    aupr = average_precision_score(tl, sc)
                    save_row({"resolution": res, "dataset": dataset, "seed": seed,
                              "train_limit": TRAIN_LIMIT, "grid": m.grid_size,
                              "img_auroc": round(auroc, 6), "img_aupr": round(aupr, 6)})
                    log(f"[{current}/{total}] res={res} | {dataset} | seed={seed} | "
                        f"AUROC={auroc:.3f} AUPR={aupr:.3f}")
                    del m
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    log(f"[{current}/{total}] ERROR res={res} | {dataset} | seed={seed}: "
                        f"{type(exc).__name__}: {exc}")
                    log_error(res, dataset, seed, exc)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    log("Resolution ablation finished.")


if __name__ == "__main__":
    main()
