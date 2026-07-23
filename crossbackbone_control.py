"""
Cross-backbone same-backbone control.

For each (backbone, dataset, seed) we extract features ONCE with the frozen
backbone, fit the same PCA, and then evaluate three scoring layers on the
same feature space:

    - GMM K=3   (this is the GaLAD default; already in backbone_ablation.csv)
    - GMM K=1   (single Gaussian per layer)
    - kNN       (mean Euclidean distance to the k nearest neighbours in the
                 patch memory bank, with a 10^4 patch cap)

This generalises the DINOv3-only same-backbone control of the paper to
DINOv2, CLIP and SigLIP, so that the scoring-layer claim (GMM beats a single
Gaussian and is competitive with dense kNN) can be made backbone-agnostic.

GMM K=3 is RE-ran here (instead of read from backbone_ablation.csv) only when
--include-k3 is passed; by default we skip it to save compute.

Design:
    - Resumable: skips (backbone, dataset, seed, method) already in the CSV.
    - Robust: per-experiment try/except with error log.
    - Cached: backbone weights and feature extraction are shared across methods.

Usage:
    uv run crossbackbone_control.py --test
    uv run crossbackbone_control.py            # 3 backbones x 46 cat x 5 seeds
    uv run crossbackbone_control.py --include-k3
"""

import argparse
import os
import csv
import gc
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--test", action="store_true")
parser.add_argument("--include-k3", action="store_true",
                    help="also re-run GMM K=3 (default: skip, use backbone_ablation.csv)")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
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

# All four backbones, including DINOv3, so that Table 12 uses an identical
# kNN protocol (k=5, 10^4 cap) across every encoder. The DINOv3 same-backbone
# control in Section 6.3 uses k=9 (standard PatchCore setting); re-running
# DINOv3 here at k=5 shows the conclusion is robust to that choice.
BACKBONES = {
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "dino", 448),
    "dinov2_vitl14": ("facebook/dinov2-large", "dino", 448),
    "clip_vitl14":   ("openai/clip-vit-large-patch14", "clip", 224),
    "siglip_vitl16": ("google/siglip-large-patch16-256", "siglip", 256),
}

LAYERS = [-6, -12]
PCA_DIM = 256
PCA_SKIP = 2
N_COMPONENTS_K3 = 3
SMOOTH_SIGMA = 1.0
BATCH_SIZE = 8
KNN_K = 5
KNN_BANK_CAP = 10_000

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

METHODS = ["k1", "knn", "pcares"]  # K=1 GMM + kNN + PCA-residual; K=3 added if --include-k3

CSV_PATH = Path("results/crossbackbone_control.csv")
LOG_PATH = Path("results/crossbackbone_control_log.txt")
ERR_PATH = Path("results/crossbackbone_control_errors.txt")
CSV_FIELDS = ["backbone", "dataset", "seed", "train_limit", "method",
              "resolution", "img_auroc", "img_aupr"]


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(backbone, dataset, seed, method, exc):
    ERR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {backbone} | {dataset} | seed={seed} | {method}\n")
        f.write(f"{type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("=" * 70 + "\n")


# ============================================================================
# Backbone feature extractor (shared across methods)
# ============================================================================

_BACKBONE_CACHE: Dict[str, tuple] = {}


def load_backbone(backbone_id: str):
    if backbone_id in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[backbone_id]
    model_name, btype, resolution = BACKBONES[backbone_id]
    log(f"Loading backbone {backbone_id} ({model_name})...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    if btype == "clip":
        model = CLIPVisionModel.from_pretrained(model_name, output_hidden_states=True)
    elif btype == "siglip":
        model = SiglipVisionModel.from_pretrained(model_name, output_hidden_states=True)
    else:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    patch_size = model.config.patch_size
    num_reg = getattr(model.config, "num_register_tokens", 0)
    if btype == "dino":
        start_idx = 1 + num_reg
    elif btype == "clip":
        start_idx = 1
    else:
        start_idx = 0
    grid = resolution // patch_size
    _BACKBONE_CACHE[backbone_id] = (processor, model, btype, resolution,
                                    start_idx, grid, device)
    return _BACKBONE_CACHE[backbone_id]


def extract_features(backbone_id: str, image_paths: List[str]):
    processor, model, btype, resolution, start_idx, grid, device = load_backbone(backbone_id)
    num_patches = grid * grid
    feats = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i:i + BATCH_SIZE]
        images = []
        for p in batch:
            img = Image.open(p)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img.resize((resolution, resolution), Image.BICUBIC))
        inputs = processor(images=images, return_tensors="pt",
                           do_resize=False, do_center_crop=False).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                out = model(**inputs)
            hs = out.hidden_states
            for b in range(len(batch)):
                d = {}
                for li in LAYERS:
                    tok = hs[li][b][start_idx:start_idx + num_patches]
                    d[li] = tok.float().cpu().numpy()
                feats.append(d)
    return feats, grid


def fit_pca(train_feats, seed: int):
    """Fit one PCA per layer over all train patches; return PCA models and
    the projected per-image, per-layer arrays."""
    pcas = {}
    projected = []  # list of dicts {li: array(num_patches, dim)}
    for li in LAYERS:
        data = np.concatenate([f[li] for f in train_feats], axis=0)
        total = PCA_DIM + PCA_SKIP
        total = min(total, data.shape[0], data.shape[1])
        pca = PCA(n_components=total, random_state=seed)
        pca.fit(data.astype(np.float32))
        pcas[li] = pca
    for f in train_feats:
        d = {}
        for li in LAYERS:
            d[li] = pcas[li].transform(f[li].astype(np.float32))[:, PCA_SKIP:]
        projected.append(d)
    return pcas, projected


def project_test(test_feats, pcas):
    out = []
    for f in test_feats:
        d = {}
        for li in LAYERS:
            d[li] = pcas[li].transform(f[li].astype(np.float32))[:, PCA_SKIP:]
        out.append(d)
    return out


def percentile_pool(scores_per_layer_per_image, grid):
    """scores_per_layer_per_image: list (per image) of dict {li: array(num_patches,)}"""
    out = []
    for d in scores_per_layer_per_image:
        layer_scores = []
        for li in LAYERS:
            amap = d[li].reshape(grid, grid)
            if SMOOTH_SIGMA > 0:
                amap = gaussian_filter(amap, sigma=SMOOTH_SIGMA)
            layer_scores.append(float(np.percentile(amap, 95)))
        out.append(float(np.mean(layer_scores)))
    return out


# ============================================================================
# Scoring methods (all on the same PCA-projected features)
# ============================================================================

def score_gmm(train_proj, test_proj, n_components: int, seed: int):
    gmms = {}
    for li in LAYERS:
        data = np.concatenate([f[li] for f in train_proj], axis=0)
        n_samples, n_features = data.shape
        max_comp = max(2, n_samples // (n_features + 10))
        n_comp = min(n_components, max_comp)
        if n_components == 1:
            n_comp = 1
        gmm = GaussianMixture(n_components=n_comp, covariance_type="full",
                              reg_covar=1e-2, random_state=seed,
                              n_init=1, max_iter=200)
        gmm.fit(data)
        gmms[li] = gmm
    out = []
    for f in test_proj:
        d = {}
        for li in LAYERS:
            d[li] = -gmms[li].score_samples(f[li])
        out.append(d)
    return out


def score_pcares(test_feats, pcas):
    """Subspace-reconstruction baseline (SubspaceAD-style): score each patch by
    its reconstruction error under the per-layer PCA fitted on normal patches.
    A patch that does not lie in the normal subspace is poorly reconstructed and
    scores high. Uses the same per-layer PCA and percentile pooling as GaLAD."""
    out = []
    for f in test_feats:
        d = {}
        for li in LAYERS:
            x = f[li].astype(np.float32)
            coords = pcas[li].transform(x)
            recon = pcas[li].inverse_transform(coords)
            d[li] = ((x - recon) ** 2).sum(axis=1)
        out.append(d)
    return out


def score_knn(train_proj, test_proj, seed: int):
    """Mean Euclidean distance to k nearest neighbours in the patch bank,
    with a 10^4 patch cap (sub-sampled deterministically)."""
    nns = {}
    for li in LAYERS:
        bank = np.concatenate([f[li] for f in train_proj], axis=0)
        if bank.shape[0] > KNN_BANK_CAP:
            rng = np.random.default_rng(seed)
            idx = rng.choice(bank.shape[0], size=KNN_BANK_CAP, replace=False)
            bank = bank[idx]
        k_eff = min(KNN_K, bank.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
        nn.fit(bank)
        nns[li] = nn
    out = []
    for f in test_proj:
        d = {}
        for li in LAYERS:
            dist, _ = nns[li].kneighbors(f[li])
            d[li] = dist.mean(axis=1)
        out.append(d)
    return out


# ============================================================================
# Data
# ============================================================================

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
            done.add((row["backbone"], row["dataset"], int(row["seed"]),
                      int(row["train_limit"]), row["method"]))
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
    methods = list(METHODS)
    if args.include_k3:
        methods = ["k3"] + methods

    datasets = FULL_DATASETS
    seeds = SEEDS
    backbones = list(BACKBONES.keys())

    if args.test:
        datasets = ["mvtec_AD/bottle"]
        seeds = [0]
        log("TEST MODE: 1 category, 1 seed, all backbones")

    completed = load_completed()
    total_configs = len(backbones) * len(datasets) * len(seeds)
    total_methods = total_configs * len(methods)
    log(f"Cross-backbone control: {len(backbones)} backbones x {len(datasets)} "
        f"datasets x {len(seeds)} seeds x {len(methods)} methods = "
        f"{total_methods} experiments ({len(completed)} already done)")

    current = 0
    for backbone_id in backbones:
        resolution = BACKBONES[backbone_id][2]
        for dataset in datasets:
            train_all, test_info = load_dataset(dataset)
            if not train_all or not test_info:
                log(f"SKIP {dataset}: not found")
                continue
            test_paths = [t["path"] for t in test_info]
            test_labels = [t["label"] for t in test_info]

            for seed in seeds:
                current += 1
                pending = [m for m in methods if (backbone_id, dataset, seed, TRAIN_LIMIT, m) not in completed]
                if not pending:
                    continue

                try:
                    sampled = sample_train(train_all, TRAIN_LIMIT, seed)
                    # Shared feature extraction + PCA
                    train_feats, grid = extract_features(backbone_id, sampled)
                    test_feats, _ = extract_features(backbone_id, test_paths)
                    pcas, train_proj = fit_pca(train_feats, seed)
                    test_proj = project_test(test_feats, pcas)

                    for method in pending:
                        try:
                            if method == "k3":
                                patch_scores = score_gmm(train_proj, test_proj, N_COMPONENTS_K3, seed)
                            elif method == "k1":
                                patch_scores = score_gmm(train_proj, test_proj, 1, seed)
                            elif method == "knn":
                                patch_scores = score_knn(train_proj, test_proj, seed)
                            elif method == "pcares":
                                patch_scores = score_pcares(test_feats, pcas)
                            else:
                                continue

                            img_scores = percentile_pool(patch_scores, grid)
                            auroc = roc_auc_score(test_labels, img_scores)
                            aupr = average_precision_score(test_labels, img_scores)
                            save_row({
                                "backbone": backbone_id, "dataset": dataset, "seed": seed,
                                "train_limit": TRAIN_LIMIT, "method": method,
                                "resolution": resolution,
                                "img_auroc": round(auroc, 6), "img_aupr": round(aupr, 6),
                            })
                            log(f"[cfg {current}/{total_configs}] {backbone_id} | {dataset} | "
                                f"seed={seed} | {method} | AUROC={auroc:.3f} AUPR={aupr:.3f}")
                        except Exception as exc:
                            log(f"ERROR {backbone_id} | {dataset} | seed={seed} | {method}: "
                                f"{type(exc).__name__}: {exc}")
                            log_error(backbone_id, dataset, seed, method, exc)

                    del train_feats, test_feats, train_proj, test_proj, pcas
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    log(f"ERROR (config) {backbone_id} | {dataset} | seed={seed}: "
                        f"{type(exc).__name__}: {exc}")
                    log_error(backbone_id, dataset, seed, "config", exc)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    log("Cross-backbone control finished.")


if __name__ == "__main__":
    main()
