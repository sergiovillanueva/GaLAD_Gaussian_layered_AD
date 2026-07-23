"""
Backbone-generality ablation for GaLAD.

Runs the exact GaLAD pipeline (intermediate-layer features -> PCA with skip ->
per-layer GMM -> NLL scoring -> smoothing -> percentile pooling) on top of
several frozen foundation-model backbones, to show that the framework is not
tied to DINOv3:

    - DINOv3 ViT-L/16  (self-supervised, default)
    - DINOv2 ViT-L/14  (self-supervised)
    - CLIP   ViT-L/14  (vision-language)
    - SigLIP ViT-L/16  (vision-language, modern)

All backbones use the same configuration (layers [-6, -12], PCA d=256 skip=2,
GMM K=3, sigma=1, percentile 95); only the feature extractor changes. Each
backbone runs at its native resolution.

Design:
    - Resumable: skips (backbone, dataset, seed, train_limit) already in the CSV.
    - Robust: a failure in one experiment is logged and the rest continue.
    - Reproducible: same seeds and train-sampling as galad_train_all.py.

Usage:
    uv run backbone_ablation.py --test          # 1 category per backbone (smoke test)
    uv run backbone_ablation.py                 # full ablation set
    uv run backbone_ablation.py --device 1      # pick GPU
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
parser.add_argument("--device", type=int, default=0, help="GPU device id")
parser.add_argument("--test", action="store_true", help="smoke test: 1 category, 2 seeds")
parser.add_argument("--full", action="store_true", help="run on all 46 categories instead of the 8 ablation ones")
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
from transformers import AutoModel, AutoImageProcessor, CLIPVisionModel, SiglipVisionModel
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# Configuration
# ============================================================================

# backbone_id -> (hf_model_name, type, native_resolution)
# type controls how patch tokens are extracted from the hidden states.
BACKBONES = {
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dino", 448),
    "dinov2_vitl14": ("facebook/dinov2-large", "dino", 448),
    "clip_vitl14":   ("openai/clip-vit-large-patch14", "clip", 224),
    "siglip_vitl16": ("google/siglip-large-patch16-256", "siglip", 256),
}

LAYERS = [-6, -12]
PCA_DIM = 256
PCA_SKIP = 2
N_COMPONENTS = 3
SMOOTH_SIGMA = 1.0
BATCH_SIZE = 8

# Ablation category subset (same 8 used elsewhere in the paper).
ABLATION_DATASETS = [
    "mvtec_AD/bottle", "mvtec_AD/carpet", "mvtec_AD/hazelnut", "mvtec_AD/leather",
    "VisA/capsules", "VisA/pcb1",
    "GoodsAD/drink_bottle", "GoodsAD/food_box",
]

# Full 46-category evaluation set (six benchmarks).
FULL_DATASETS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
    "mvtec_loco_AD/breakfast_box", "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins", "mvtec_loco_AD/screw_bag", "mvtec_loco_AD/splicing_connectors",
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum",
    "VisA/fryum", "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1",
    "VisA/pcb2", "VisA/pcb3", "VisA/pcb4", "VisA/pipe_fryum",
    "AutoVI/engine_wiring", "AutoVI/pipe_clip", "AutoVI/tank_screw",
    "AutoVI/underbody_pipes", "AutoVI/underbody_screw",
    "btad/01", "btad/02", "btad/03",
    "GoodsAD/cigarette_box", "GoodsAD/drink_bottle", "GoodsAD/drink_can",
    "GoodsAD/food_bottle", "GoodsAD/food_box", "GoodsAD/food_package",
]
SEEDS = [0, 1, 2, 3, 4]
TRAIN_LIMIT = 10

CSV_PATH = Path("results/backbone_ablation.csv")
LOG_PATH = Path("results/backbone_ablation_log.txt")
ERR_PATH = Path("results/backbone_ablation_errors.txt")
CSV_FIELDS = ["backbone", "dataset", "seed", "train_limit", "resolution",
              "img_auroc", "img_aupr"]


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(backbone, dataset, seed, exc):
    ERR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {backbone} | {dataset} | seed={seed}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("=" * 70 + "\n")


# ============================================================================
# Multi-backbone feature extractor (reuses GaLAD's PCA+GMM scoring)
# ============================================================================

class BackboneGaLAD:
    """GaLAD pipeline on top of an arbitrary frozen backbone.

    Only the feature-extraction step depends on the backbone type; the PCA,
    GMM, smoothing, and pooling steps are identical to galad_model.GaLAD.
    """

    # Cache loaded backbones so repeated category runs do not reload weights.
    _cache = {}

    def __init__(self, backbone_id: str, seed: int = 42):
        self.backbone_id = backbone_id
        model_name, btype, resolution = BACKBONES[backbone_id]
        self.btype = btype
        self.resolution = resolution
        self.layers = LAYERS
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if backbone_id not in BackboneGaLAD._cache:
            log(f"Loading backbone {backbone_id} ({model_name})...")
            processor = AutoImageProcessor.from_pretrained(model_name)
            if btype == "clip":
                model = CLIPVisionModel.from_pretrained(model_name, output_hidden_states=True)
            elif btype == "siglip":
                model = SiglipVisionModel.from_pretrained(model_name, output_hidden_states=True)
            else:  # dino (DINOv2 / DINOv3 via AutoModel)
                model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            model = model.to(self.device).eval()
            patch_size = model.config.patch_size
            num_reg = getattr(model.config, "num_register_tokens", 0)
            BackboneGaLAD._cache[backbone_id] = (processor, model, patch_size, num_reg)

        self.processor, self.model, self.patch_size, self.num_reg = BackboneGaLAD._cache[backbone_id]
        self.grid_size = self.resolution // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # start index of patch tokens in the token sequence
        if btype == "dino":
            self.start_idx = 1 + self.num_reg          # [CLS] (+ registers)
        elif btype == "clip":
            self.start_idx = 1                          # [CLS], no registers
        else:  # siglip has no CLS token
            self.start_idx = 0

        self.pca_models = {}
        self.gmm_models = {}

    def _extract(self, image_paths: List[str]):
        feats = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch = image_paths[i:i + BATCH_SIZE]
            images = []
            for p in batch:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img.resize((self.resolution, self.resolution), Image.BICUBIC))
            inputs = self.processor(images=images, return_tensors="pt",
                                    do_resize=False, do_center_crop=False).to(self.device)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    out = self.model(**inputs)
                hs = out.hidden_states
                for b in range(len(batch)):
                    d = {}
                    for li in self.layers:
                        tok = hs[li][b][self.start_idx:self.start_idx + self.num_patches]
                        d[li] = tok.float().cpu().numpy()
                    feats.append(d)
        return feats

    def fit(self, train_paths: List[str]):
        feats = self._extract(train_paths)
        for li in self.layers:
            data = np.concatenate([f[li] for f in feats], axis=0)
            total = PCA_DIM + PCA_SKIP
            # guard: PCA needs at least `total` samples and features
            total = min(total, data.shape[0], data.shape[1])
            pca = PCA(n_components=total, random_state=self.seed)
            proj = pca.fit_transform(data.astype(np.float32))[:, PCA_SKIP:]
            self.pca_models[li] = pca
            n_samples, n_features = proj.shape
            max_comp = max(2, n_samples // (n_features + 10))
            n_comp = min(N_COMPONENTS, max_comp)
            gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                                  reg_covar=1e-2, random_state=self.seed,
                                  n_init=1, max_iter=200)
            gmm.fit(proj)
            self.gmm_models[li] = gmm
        return self

    def score(self, test_paths: List[str]) -> List[float]:
        feats = self._extract(test_paths)
        scores = []
        for f in feats:
            layer_scores = []
            for li in self.layers:
                proj = self.pca_models[li].transform(f[li].astype(np.float32))[:, PCA_SKIP:]
                nll = -self.gmm_models[li].score_samples(proj)
                amap = nll.reshape(self.grid_size, self.grid_size)
                if SMOOTH_SIGMA > 0:
                    amap = gaussian_filter(amap, sigma=SMOOTH_SIGMA)
                layer_scores.append(float(np.percentile(amap, 95)))
            scores.append(float(np.mean(layer_scores)))
        return scores


# ============================================================================
# Data
# ============================================================================

def load_dataset(data_dir: str):
    """Load train/test. Handles VisA's official split CSV."""
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
    train_dir = data_path / "train" / "good"
    test_dir = data_path / "test"
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


def sample_train(train_paths: List[str], n: int, seed: int) -> List[str]:
    if len(train_paths) <= n:
        return train_paths
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(train_paths), n))


def load_completed():
    if not CSV_PATH.exists():
        return set()
    done = set()
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            done.add((row["backbone"], row["dataset"], int(row["seed"]), int(row["train_limit"])))
    return done


def save_row(row: dict):
    write_header = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, delimiter=";")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ============================================================================
# Main
# ============================================================================

def main():
    datasets = FULL_DATASETS if args.full else ABLATION_DATASETS
    seeds = SEEDS
    backbones = list(BACKBONES.keys())

    if args.full:
        log("FULL MODE: 46 categories, 5 seeds, all backbones")
    if args.test:
        datasets = ["mvtec_AD/bottle"]
        seeds = [0, 1]
        log("TEST MODE: 1 category, 2 seeds, all backbones")

    completed = load_completed()
    total = len(backbones) * len(datasets) * len(seeds)
    log(f"Backbone ablation: {len(backbones)} backbones x {len(datasets)} datasets "
        f"x {len(seeds)} seeds = {total} experiments ({len(completed)} already done)")

    current = 0
    for backbone_id in backbones:
        for dataset in datasets:
            train_all, test_info = load_dataset(dataset)
            if not train_all or not test_info:
                log(f"SKIP {dataset}: not found")
                continue
            test_paths = [t["path"] for t in test_info]
            test_labels = [t["label"] for t in test_info]

            for seed in seeds:
                current += 1
                key = (backbone_id, dataset, seed, TRAIN_LIMIT)
                if key in completed:
                    continue

                try:
                    sampled = sample_train(train_all, TRAIN_LIMIT, seed)
                    model = BackboneGaLAD(backbone_id, seed=seed)
                    model.fit(sampled)
                    scores = model.score(test_paths)
                    auroc = roc_auc_score(test_labels, scores)
                    aupr = average_precision_score(test_labels, scores)
                    save_row({
                        "backbone": backbone_id, "dataset": dataset, "seed": seed,
                        "train_limit": TRAIN_LIMIT, "resolution": model.resolution,
                        "img_auroc": round(auroc, 6), "img_aupr": round(aupr, 6),
                    })
                    log(f"[{current}/{total}] {backbone_id} | {dataset} | seed={seed} | "
                        f"AUROC={auroc:.3f} AUPR={aupr:.3f}")
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    log(f"[{current}/{total}] ERROR {backbone_id} | {dataset} | seed={seed}: "
                        f"{type(exc).__name__}: {exc}")
                    log_error(backbone_id, dataset, seed, exc)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    log("Backbone ablation finished.")


if __name__ == "__main__":
    main()
