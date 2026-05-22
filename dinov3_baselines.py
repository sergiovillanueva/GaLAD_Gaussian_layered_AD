"""
Same-backbone baselines for fair comparison (all 46 categories).

Implements:
- DINOv3-kNN: k-Nearest Neighbors on DINOv3 features (PatchCore-style)
- DINOv3-K1: Single Gaussian density (PaDiM-style)

Resumable: checks CSV before running each experiment.
Sampling matches galad_train_all.py (sorted + random.sample).

Usage: uv run dinov3_baselines.py
"""

import os
import csv
import random
import gc
import numpy as np
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, AutoModel
import transformers
import warnings

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================================
# Configuration
# ============================================================================

DATASETS = [
    # MVTec AD (15)
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
    # MVTec LOCO (5)
    "mvtec_loco_AD/breakfast_box", "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins", "mvtec_loco_AD/screw_bag", "mvtec_loco_AD/splicing_connectors",
    # VisA (12)
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum",
    "VisA/fryum", "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1",
    "VisA/pcb2", "VisA/pcb3", "VisA/pcb4", "VisA/pipe_fryum",
    # AutoVI (5)
    "AutoVI/engine_wiring", "AutoVI/pipe_clip", "AutoVI/tank_screw",
    "AutoVI/underbody_pipes", "AutoVI/underbody_screw",
    # BTAD (3)
    "btad/01", "btad/02", "btad/03",
    # GoodsAD (6)
    "GoodsAD/cigarette_box", "GoodsAD/drink_bottle", "GoodsAD/drink_can",
    "GoodsAD/food_bottle", "GoodsAD/food_box", "GoodsAD/food_package",
]

SEEDS = [0, 1, 2, 3, 4]
TRAIN_SIZES = [5, 10, 20]

RESOLUTION = 448
BATCH_SIZE = 56
LAYERS = [-6, -12]
PCA_DIM = 256
PCA_SKIP = 2

CSV_PATH = Path("results/dinov3_baselines_all.csv")
CSV_FIELDS = ["model", "dataset", "seed", "train_limit", "img_auroc", "img_aupr"]


# ============================================================================
# Resume logic
# ============================================================================

def load_completed():
    if not CSV_PATH.exists():
        return set()
    done = set()
    with open(CSV_PATH, "r", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            key = (row["model"], row["dataset"], int(row["seed"]), int(row["train_limit"]))
            done.add(key)
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
# DINOv3 Feature Extractor (shared singleton)
# ============================================================================

class DINOv3Extractor:
    _model = None
    _processor = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DINOv3Extractor._model is None:
            print("Loading DINOv3 ViT-L/16...")
            name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            DINOv3Extractor._processor = AutoImageProcessor.from_pretrained(name)
            DINOv3Extractor._model = AutoModel.from_pretrained(
                name, output_hidden_states=True
            ).to(self.device)
            DINOv3Extractor._model.eval()
        self.model = DINOv3Extractor._model
        self.processor = DINOv3Extractor._processor
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.grid_size = RESOLUTION // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    def extract(self, image_paths: List[str]) -> np.ndarray:
        all_features = []
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            images = []
            for path in batch_paths:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)
                images.append(img)

            inputs = self.processor(
                images=images, return_tensors="pt",
                do_resize=False, do_center_crop=False
            ).to(self.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states
                start_idx = 1 + self.num_register_tokens
                for b in range(len(batch_paths)):
                    layer_feats = []
                    for layer_idx in LAYERS:
                        tokens = hidden_states[layer_idx][b]
                        patch_tokens = tokens[start_idx:start_idx + self.num_patches].cpu().numpy()
                        layer_feats.append(patch_tokens)
                    combined = np.concatenate(layer_feats, axis=1)
                    all_features.append(combined)
        return np.array(all_features)


# ============================================================================
# DINOv3-kNN
# ============================================================================

class DINOv3kNN:
    def __init__(self, k: int = 9, seed: int = 42):
        self.k = k
        self.seed = seed
        self.extractor = DINOv3Extractor()
        self.pca = None
        self.nn = None

    def fit(self, train_paths: List[str]):
        features = self.extractor.extract(train_paths)
        N, P, D = features.shape
        features_flat = features.reshape(-1, D)

        total_dim = PCA_DIM + PCA_SKIP
        self.pca = PCA(n_components=total_dim, random_state=self.seed)
        features_pca = self.pca.fit_transform(features_flat.astype(np.float32))
        features_pca = features_pca[:, PCA_SKIP:]

        max_samples = 10000
        if features_pca.shape[0] > max_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(features_pca.shape[0], max_samples, replace=False)
            memory_bank = features_pca[idx]
        else:
            memory_bank = features_pca

        self.nn = NearestNeighbors(n_neighbors=self.k, algorithm="auto", metric="euclidean")
        self.nn.fit(memory_bank)

    def score(self, test_paths: List[str]) -> List[float]:
        features = self.extractor.extract(test_paths)
        N, P, D = features.shape
        scores = []
        for i in range(N):
            patches_pca = self.pca.transform(features[i].astype(np.float32))[:, PCA_SKIP:]
            distances, _ = self.nn.kneighbors(patches_pca)
            patch_scores = distances.mean(axis=1)
            scores.append(float(np.max(patch_scores)))
        return scores


# ============================================================================
# DINOv3-K1 (Single Gaussian)
# ============================================================================

class DINOv3K1:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.extractor = DINOv3Extractor()
        self.pca = None
        self.gmm = None

    def fit(self, train_paths: List[str]):
        features = self.extractor.extract(train_paths)
        N, P, D = features.shape
        features_flat = features.reshape(-1, D)

        total_dim = PCA_DIM + PCA_SKIP
        self.pca = PCA(n_components=total_dim, random_state=self.seed)
        features_pca = self.pca.fit_transform(features_flat.astype(np.float32))
        features_pca = features_pca[:, PCA_SKIP:]

        self.gmm = GaussianMixture(
            n_components=1, covariance_type="full",
            reg_covar=1e-2, random_state=self.seed,
        )
        self.gmm.fit(features_pca)

    def score(self, test_paths: List[str]) -> List[float]:
        features = self.extractor.extract(test_paths)
        N, P, D = features.shape
        grid_size = self.extractor.grid_size
        scores = []
        for i in range(N):
            patches_pca = self.pca.transform(features[i].astype(np.float32))[:, PCA_SKIP:]
            patch_nll = -self.gmm.score_samples(patches_pca)
            amap = patch_nll.reshape(grid_size, grid_size)
            amap = gaussian_filter(amap, sigma=1.0)
            scores.append(float(np.percentile(amap, 95)))
        return scores


# ============================================================================
# Dataset Loading (matches galad_train_all.py sampling)
# ============================================================================

def load_visa_dataset(data_dir: str) -> Tuple[List[str], List[dict]]:
    """Load VisA category using the official 1cls split CSV."""
    import csv as _csv
    category = data_dir.split("/", 1)[1]
    visa_root = Path("data/VisA")
    split_csv = visa_root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
        return [], []
    train_paths, test_info = [], []
    with open(split_csv, "r", newline="") as f:
        for row in _csv.DictReader(f):
            if row["object"] != category:
                continue
            img_path = str(visa_root / row["image"])
            if row["split"] == "train":
                train_paths.append(img_path)
            elif row["split"] == "test":
                label = 0 if row["label"] == "normal" else 1
                test_info.append({"path": img_path, "label": label})
    return sorted(train_paths), test_info


def load_dataset(data_dir: str) -> Tuple[List[str], List[dict]]:
    # VisA uses an official split CSV, not the train/good + test/ structure
    if data_dir.startswith("VisA/"):
        return load_visa_dataset(data_dir)

    data_path = Path("data") / data_dir
    train_dir = data_path / "train" / "good"
    test_dir = data_path / "test"
    if not train_dir.exists():
        return [], []

    train_paths = sorted([str(p) for p in train_dir.glob("*.*")
                         if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
    test_info = []
    for cat_dir in sorted(test_dir.iterdir()):
        if cat_dir.is_dir():
            label = 0 if cat_dir.name == "good" else 1
            for img_path in sorted(cat_dir.glob("*.*")):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    test_info.append({"path": str(img_path), "label": label})
    return train_paths, test_info


def sample_train(train_paths: List[str], n: int, seed: int) -> List[str]:
    """Sample training images. Matches galad_train_all.py: sorted + random.sample."""
    if len(train_paths) <= n:
        return train_paths
    paths_sorted = sorted(train_paths)
    rng = random.Random(seed)
    chosen = rng.sample(paths_sorted, n)
    return sorted(chosen)


# ============================================================================
# Main
# ============================================================================

def main():
    completed = load_completed()
    methods = {"dinov3_knn": DINOv3kNN, "dinov3_k1": DINOv3K1}

    total = len(DATASETS) * len(SEEDS) * len(TRAIN_SIZES) * len(methods)
    done = len(completed)
    print(f"Same-backbone baselines: {total} experiments, {done} already done, {total - done} remaining")

    for dataset in DATASETS:
        train_paths, test_info = load_dataset(dataset)
        if not train_paths or not test_info:
            print(f"  {dataset}: SKIPPED (not found)")
            continue

        test_paths = [t["path"] for t in test_info]
        test_labels = [t["label"] for t in test_info]

        for train_size in TRAIN_SIZES:
            for seed in SEEDS:
                for method_name, MethodClass in methods.items():
                    key = (method_name, dataset, seed, train_size)
                    if key in completed:
                        continue

                    sampled = sample_train(train_paths, train_size, seed)
                    model = MethodClass(seed=seed)
                    model.fit(sampled)
                    scores = model.score(test_paths)

                    auroc = roc_auc_score(test_labels, scores)
                    aupr = average_precision_score(test_labels, scores)

                    save_row({
                        "model": method_name,
                        "dataset": dataset,
                        "seed": seed,
                        "train_limit": train_size,
                        "img_auroc": round(auroc, 6),
                        "img_aupr": round(aupr, 6),
                    })

                    print(f"  {method_name} | {dataset} | N={train_size} seed={seed} | AUROC={auroc:.3f} AUPR={aupr:.3f}")

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
