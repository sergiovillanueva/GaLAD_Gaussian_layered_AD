"""GALAD batch training script - Paper Version (Variant A only).

This script extracts DINOv3 embeddings once per dataset and evaluates the final
GaLAD configuration (Variant A) across seeds and train sizes.

Variant A (final paper):
- layers=[-6, -12]
- pca_skip=2
- pca_dim=256 (kept *after* skipping; i.e., fit PCA with 256 + pca_skip comps)
- n_comp=3

Usage: uv run train_galad.py [--device 0]
"""

import os
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='GPU device ID')
parser.add_argument(
    '--data-root',
    type=str,
    default=os.environ.get('GALAD_DATA_ROOT', 'data'),
    help='Datasets root directory (contains mvtec_AD, VisA, AutoVI, ...)',
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
DATA_ROOT_STR = args.data_root

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import torch
import gc
import csv
import json
import traceback
import random
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.ndimage import zoom, label as cc_label, gaussian_filter
import matplotlib.cm as cm

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import transformers
from transformers import AutoImageProcessor, AutoModel
transformers.logging.set_verbosity_error()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = [
    "mvtec_AD/bottle",
    "mvtec_AD/cable",
    "mvtec_AD/capsule",
    "mvtec_AD/carpet",
    "mvtec_AD/grid",
    "mvtec_AD/hazelnut",
    "mvtec_AD/leather",
    "mvtec_AD/metal_nut",
    "mvtec_AD/pill",
    "mvtec_AD/screw",
    "mvtec_AD/tile",
    "mvtec_AD/toothbrush",
    "mvtec_AD/transistor",
    "mvtec_AD/wood",
    "mvtec_AD/zipper",
    "AutoVI/underbody_pipes",
    "AutoVI/underbody_screw",
    "AutoVI/engine_wiring",
    "AutoVI/pipe_clip",
    "AutoVI/tank_screw",
    "mvtec_loco_AD/breakfast_box",
    "mvtec_loco_AD/juice_bottle",
    "mvtec_loco_AD/pushpins",
    "mvtec_loco_AD/screw_bag",
    "mvtec_loco_AD/splicing_connectors",
    "VisA/candle",
    "VisA/capsules",
    "VisA/cashew",
    "VisA/chewinggum",
    "VisA/fryum",
    "VisA/macaroni1",
    "VisA/macaroni2",
    "VisA/pcb1",
    "VisA/pcb2",
    "VisA/pcb3",
    "VisA/pcb4",
    "VisA/pipe_fryum",
    "btad/01",
    "btad/02",
    "btad/03",
    "GoodsAD/cigarette_box",
    "GoodsAD/drink_bottle",
    "GoodsAD/drink_can",
    "GoodsAD/food_bottle",
    "GoodsAD/food_box",
    "GoodsAD/food_package",
]

SEEDS = [0,1,2,3,4]
TRAIN_SIZES = [2, 5, 10, 20, 50]  
CANDIDATES = {
    "galad": {
        "layers": [-6, -12],
        "pca_skip": 2,
        "n_comp": 3,
        "reduction": "pca",
        "dim": 256,
    },
}

RESOLUTION = 448
BATCH_SIZE = 56
SMOOTH_SIGMA = 1.0
MAX_VIS_IMAGES = 0

CSV_PATH = Path("results/galad_results.csv")
LOG_PATH = Path("results/galad_log.txt")
ERROR_LOG_PATH = Path("results/galad_errors.txt")
TEST_OUTPUT_DIR = Path("results/galad_test")

# =============================================================================
# UTILITIES
# =============================================================================

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error(dataset, model, seed, train_size, exception):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"ERROR: {timestamp}\n")
        f.write(f"Dataset: {dataset}, Model: {model}, Seed: {seed}, Train: {train_size}\n")
        f.write(f"Exception: {type(exception).__name__}: {exception}\n")
        f.write(traceback.format_exc())
        f.write("=" * 80 + "\n\n")


def get_completed():
    completed = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (row["dataset"], row["model"], int(row["seed"]), str(row["train_limit"]))
                completed.add(key)
    return completed


def cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# DATASET LOADING (same as baseline)
# =============================================================================

def load_mvtec_dataset(data_root):
    train_dir = data_root / "train" / "good"
    test_dir = data_root / "test"
    gt_dir = data_root / "ground_truth"

    train_paths = sorted([str(p) for p in train_dir.glob("*.*")
                          if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])

    test_info = []
    for cat_dir in sorted(test_dir.iterdir()):
        if cat_dir.is_dir():
            is_normal = cat_dir.name == "good"
            label = 0 if is_normal else 1
            for img_path in sorted(cat_dir.glob("*.*")):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
                    continue
                entry = {'path': str(img_path), 'label': label, 'mask_path': None}
                if not is_normal and gt_dir.exists():
                    mask_dir = gt_dir / cat_dir.name
                    if mask_dir.exists():
                        stem = img_path.stem
                        for ext in ['.png', '_mask.png']:
                            mp = mask_dir / f"{stem}{ext}"
                            if mp.exists():
                                entry['mask_path'] = str(mp)
                                break
                test_info.append(entry)
    return train_paths, test_info


def load_loco_dataset(data_root):
    train_dir = data_root / "train" / "good"
    test_dir = data_root / "test"
    gt_dir = data_root / "ground_truth"

    train_paths = sorted([str(p) for p in train_dir.glob("*.*")
                          if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])

    test_info = []
    for cat_dir in sorted(test_dir.iterdir()):
        if cat_dir.is_dir():
            is_normal = cat_dir.name == "good"
            label = 0 if is_normal else 1
            for img_path in sorted(cat_dir.glob("*.*")):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
                    continue
                entry = {'path': str(img_path), 'label': label, 'mask_path': None, 'mask_dir': None}
                if not is_normal and gt_dir.exists():
                    mask_subdir = gt_dir / cat_dir.name / img_path.stem
                    if mask_subdir.exists():
                        masks = list(mask_subdir.glob("*.png"))
                        if masks:
                            entry['mask_path'] = str(masks[0])
                            entry['mask_dir'] = str(mask_subdir)
                test_info.append(entry)
    return train_paths, test_info


def load_visa_dataset(data_root):
    import csv as csv_mod
    category = data_root.name
    split_csv = data_root.parent / "split_csv" / "1cls.csv"
    visa_root = data_root.parent

    train_paths = []
    test_info = []

    if not split_csv.exists():
        raise FileNotFoundError(f"VisA split CSV not found: {split_csv}")

    with open(split_csv, 'r') as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            if row['object'] != category:
                continue
            img_path = visa_root / row['image']
            if not img_path.exists():
                continue
            if row['split'] == 'train':
                train_paths.append(str(img_path))
            elif row['split'] == 'test':
                label = 0 if row['label'] == 'normal' else 1
                mask_path = None
                if row.get('mask') and row['mask'].strip():
                    mf = visa_root / row['mask']
                    if mf.exists():
                        mask_path = str(mf)
                test_info.append({'path': str(img_path), 'label': label, 'mask_path': mask_path})
    return train_paths, test_info


def load_dataset(dataset):
    data_root = Path(DATA_ROOT_STR) / dataset
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found: {data_root}")
    if dataset.startswith("VisA/"):
        return load_visa_dataset(data_root)
    elif dataset.startswith("mvtec_loco_AD/"):
        return load_loco_dataset(data_root)
    else:
        return load_mvtec_dataset(data_root)


def load_mask(entry):
    if not entry.get('mask_path'):
        return None
    if entry.get('mask_dir'):
        mask_dir = Path(entry['mask_dir'])
        combined = None
        for mp in mask_dir.glob("*.png"):
            m = np.array(Image.open(mp).convert('L'))
            combined = m if combined is None else np.maximum(combined, m)
        return combined
    return np.array(Image.open(entry['mask_path']).convert('L'))


# =============================================================================
# METRICS (same as baseline)
# =============================================================================

def compute_tpr_at_fpr(labels, scores, target_fpr):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(fpr <= target_fpr)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0


def compute_f1_max(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
    return float(np.max(f1))


def compute_auspro(anomaly_maps, gt_masks, image_labels, alpha=0.3):
    normal_idx = np.where(np.array(image_labels) == 0)[0]
    abnormal_idx = np.where(np.array(image_labels) == 1)[0]
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return None, None

    all_normal_pixels = []
    for idx in normal_idx:
        if idx < len(anomaly_maps):
            all_normal_pixels.append(anomaly_maps[idx].ravel())
    if not all_normal_pixels:
        return None, None
    all_normal_pixels = np.concatenate(all_normal_pixels)

    thresholds = np.linspace(np.max(all_normal_pixels), np.min(all_normal_pixels), 100)
    total_normal = len(all_normal_pixels)
    fprs, pros = [], []

    for thr in thresholds:
        fpr = (all_normal_pixels >= thr).sum() / total_normal
        fprs.append(fpr)
        pro_vals = []
        for idx in abnormal_idx:
            if idx >= len(anomaly_maps):
                continue
            amap, gtm = anomaly_maps[idx], gt_masks[idx]
            labeled, num_cc = cc_label(gtm)
            for cc_id in range(1, num_cc + 1):
                cc_mask = (labeled == cc_id)
                cc_size = cc_mask.sum()
                if cc_size > 0:
                    pro_vals.append(((amap >= thr) & cc_mask).sum() / cc_size)
        pros.append(np.mean(pro_vals) if pro_vals else 0.0)

    fprs, pros = np.array(fprs), np.array(pros)
    valid = fprs <= alpha
    if not np.any(valid):
        return None, None
    order = np.argsort(fprs[valid])
    xf, yf = fprs[valid][order], pros[valid][order]
    auspro = np.trapezoid(yf, xf) / alpha
    return float(yf[-1]), auspro


# =============================================================================
# FEATURE EXTRACTOR (shared across all configs)
# =============================================================================

class FeatureExtractor:
    """DINOv3 feature extractor - shared across all candidate configs."""

    def __init__(self, resolution=448, device=None):
        self.resolution = resolution
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()

        self.patch_size = self.model.config.patch_size
        self.num_register = getattr(self.model.config, "num_register_tokens", 0)
        self.grid_size = resolution // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    def extract_batch(self, image_paths, batch_size=32, layer_indices=[-6, -12, -9]):
        """Extract features for all required layers at once."""
        unique_layers = list(set(layer_indices))
        all_features = {layer: [] for layer in unique_layers}

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB").resize((self.resolution, self.resolution), Image.BICUBIC)
                images.append(img)

            inputs = self.processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
            inputs = inputs.to(self.device)

            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                outputs = self.model(**inputs)

            hidden = outputs.hidden_states
            start_idx = 1 + self.num_register

            for b in range(len(batch_paths)):
                for layer in unique_layers:
                    patches = hidden[layer][b][start_idx:start_idx + self.num_patches].cpu()
                    all_features[layer].append(patches)

        # Stack per layer: [n_images, num_patches, hidden_dim]
        return {layer: torch.stack(feats) for layer, feats in all_features.items()}


# =============================================================================
# GALAD MODEL (optimized for batch processing)
# =============================================================================

class GaLADConfig:
    """Single GaLAD configuration."""

    def __init__(self, name, layers, pca_skip, n_comp, reduction, dim, grid_size, random_state):
        self.name = name
        self.layers = layers
        self.pca_skip = pca_skip
        self.n_comp = n_comp
        self.reduction = reduction
        self.dim = dim
        self.grid_size = grid_size
        self.random_state = random_state

        self.reducers = {}  # layer -> PCA/SVD
        self.gmms = {}  # layer -> GMM

    def fit(self, train_features):
        """Fit PCA/SVD and GMM for each layer."""
        for layer in self.layers:
            feats = train_features[layer]  # [n_images, n_patches, hidden_dim]
            feats_flat = feats.reshape(-1, feats.shape[-1]).numpy().astype(np.float32)

            # Subsample if too many patches
            max_patches = 100000
            if feats_flat.shape[0] > max_patches:
                np.random.seed(self.random_state)
                idx = np.random.choice(feats_flat.shape[0], max_patches, replace=False)
                feats_sub = feats_flat[idx]
            else:
                feats_sub = feats_flat

            # Dimensionality reduction
            # Keep `dim` components *after* skipping the first `pca_skip` components.
            total_dim = self.dim + self.pca_skip if self.pca_skip > 0 else self.dim
            if self.reduction == "pca":
                reducer = PCA(n_components=total_dim, random_state=self.random_state)
            else:
                reducer = TruncatedSVD(n_components=total_dim, random_state=self.random_state)

            feats_reduced = reducer.fit_transform(feats_sub)
            self.reducers[layer] = reducer

            # Skip first N components (pca_skip)
            if self.pca_skip > 0:
                feats_reduced = feats_reduced[:, self.pca_skip:]

            # GMM - adjust components based on available samples
            n_samples = feats_reduced.shape[0]
            n_features = feats_reduced.shape[1]
            max_comp = max(2, n_samples // (n_features + 10))
            n_comp = min(self.n_comp, max_comp)

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                reg_covar=1e-2,
                random_state=self.random_state,
                n_init=1,
                max_iter=200
            )
            gmm.fit(feats_reduced)
            self.gmms[layer] = gmm

    def score_batch(self, test_features, smooth_sigma=2.0):
        """Score test images, return (scores, anomaly_maps)."""
        n_images = test_features[self.layers[0]].shape[0]
        all_layer_scores = []
        all_layer_maps = []

        for layer in self.layers:
            feats = test_features[layer]  # [n_images, n_patches, hidden_dim]
            layer_scores = []
            layer_maps = []

            for i in range(n_images):
                patches = feats[i].numpy().astype(np.float32)
                reduced = self.reducers[layer].transform(patches)
                if self.pca_skip > 0:
                    reduced = reduced[:, self.pca_skip:]

                nll = -self.gmms[layer].score_samples(reduced)
                amap = nll.reshape(self.grid_size, self.grid_size)
                if smooth_sigma > 0:
                    amap = gaussian_filter(amap, sigma=smooth_sigma)
                layer_maps.append(amap)
                layer_scores.append(np.percentile(amap, 95))

            all_layer_scores.append(layer_scores)
            all_layer_maps.append(layer_maps)

        # Ensemble: mean across layers
        scores = np.mean(all_layer_scores, axis=0)
        anomaly_maps = [np.mean([all_layer_maps[l][i] for l in range(len(self.layers))], axis=0)
                        for i in range(n_images)]

        return scores.tolist(), anomaly_maps


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_dataset(dataset, extractor, completed, train_size):
    """Train all candidates on a single dataset."""
    results = []

    # Load dataset
    train_paths_all, test_info = load_dataset(dataset)
    if len(train_paths_all) == 0:
        raise ValueError(f"No training images for {dataset}")

    # Get all required layers across all candidates
    all_layers = set()
    for cfg in CANDIDATES.values():
        all_layers.update(cfg["layers"])
    all_layers = list(all_layers)

    # Extract test features once (same for all seeds/configs)
    test_paths = [t['path'] for t in test_info]
    test_labels = np.array([t['label'] for t in test_info])

    log(f"  Extracting test features ({len(test_paths)} images, layers={all_layers})...")
    test_features = extractor.extract_batch(test_paths, batch_size=BATCH_SIZE, layer_indices=all_layers)

    # Process each seed
    for seed in SEEDS:
        # Subsample training images (deterministic per seed)
        rng = random.Random(seed)
        if train_size and len(train_paths_all) > train_size:
            train_paths = rng.sample(sorted(train_paths_all), train_size)
            # Match baseline behavior: stable order after sampling.
            train_paths = sorted(train_paths)
        else:
            train_paths = train_paths_all

        n_train = len(train_paths)
        train_limit_str = str(train_size) if train_size else "all"

        # Extract train features for this seed's subset
        train_features = extractor.extract_batch(train_paths, batch_size=BATCH_SIZE, layer_indices=all_layers)

        # Train and evaluate each candidate
        for model_name, cfg in CANDIDATES.items():
            key = (dataset, model_name, seed, train_limit_str)
            if key in completed:
                log(f"  SKIP {model_name} seed={seed} train={train_limit_str}")
                continue

            start_time = datetime.now()

            # Set seeds
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Filter features to only required layers
            train_feats = {l: train_features[l] for l in cfg["layers"]}
            test_feats = {l: test_features[l] for l in cfg["layers"]}

            # Create and fit model
            model = GaLADConfig(
                name=model_name,
                layers=cfg["layers"],
                pca_skip=cfg["pca_skip"],
                n_comp=cfg["n_comp"],
                reduction=cfg["reduction"],
                dim=cfg["dim"],
                grid_size=extractor.grid_size,
                random_state=seed
            )
            model.fit(train_feats)

            # Score test images
            scores, anomaly_maps = model.score_batch(test_feats, smooth_sigma=SMOOTH_SIGMA)
            scores = np.array(scores)

            train_time = (datetime.now() - start_time).total_seconds()

            # Compute metrics
            img_auroc = roc_auc_score(test_labels, scores)
            img_aupr = average_precision_score(test_labels, scores)
            img_f1max = compute_f1_max(test_labels, scores)
            tpr_tnr95 = compute_tpr_at_fpr(test_labels, scores, 0.05)
            tpr_tnr90 = compute_tpr_at_fpr(test_labels, scores, 0.10)

            # Pixel metrics
            pix_auroc, pix_aupr, spro, auspro = None, None, None, None
            all_pix_scores, all_pix_labels = [], []
            maps_2d, masks_2d, labels_pro = [], [], []

            # Match baseline behavior: only compute pixel metrics if GT masks exist in this dataset.
            # Baseline (Anomalib) evaluates at image_size (448x448), so we do the same.
            has_pixel_gt = any((e.get('mask_path') is not None) or (e.get('mask_dir') is not None) for e in test_info)
            eval_shape = (RESOLUTION, RESOLUTION)  # Same as baseline

            if has_pixel_gt:
                for i, entry in enumerate(test_info):
                    am = anomaly_maps[i]
                    mask = load_mask(entry)

                    if mask is not None:
                        gt = (mask > 0).astype(np.uint8)
                        # Scale GT mask to eval_shape (baseline does this via Anomalib transforms)
                        if gt.shape != eval_shape:
                            zy, zx = eval_shape[0] / gt.shape[0], eval_shape[1] / gt.shape[1]
                            gt = zoom(gt.astype(float), (zy, zx), order=0) > 0.5
                            gt = gt.astype(np.uint8)
                    else:
                        # Normal images: all-zero mask at eval_shape
                        gt = np.zeros(eval_shape, dtype=np.uint8)

                    # Scale anomaly map to eval_shape
                    if am.shape != eval_shape:
                        zy, zx = eval_shape[0] / am.shape[0], eval_shape[1] / am.shape[1]
                        am = zoom(am, (zy, zx), order=1)

                    all_pix_scores.append(am.ravel())
                    all_pix_labels.append(gt.ravel())

                    maps_2d.append(am)
                    masks_2d.append(gt)
                    labels_pro.append(int(entry.get('label', 0)))

            if all_pix_scores:
                pix_s = np.concatenate(all_pix_scores)
                pix_l = np.concatenate(all_pix_labels)
                if len(np.unique(pix_l)) > 1:
                    pix_auroc = roc_auc_score(pix_l, pix_s)
                    pix_aupr = average_precision_score(pix_l, pix_s)
                if maps_2d:
                    spro, auspro = compute_auspro(maps_2d, masks_2d, labels_pro)

            result = {
                "dataset": dataset,
                "model": model_name,
                "image_size": f"{RESOLUTION}x{RESOLUTION}",
                "train_limit": train_limit_str,
                "n_train": n_train,
                "n_test": len(test_info),
                "seed": seed,
                "epochs": 1,
                "train_time": train_time,
                "gpu_mem_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                "img_auroc": img_auroc,
                "img_aupr": img_aupr,
                "img_f1max": img_f1max,
                "tpr_tnr95": tpr_tnr95,
                "tpr_tnr90": tpr_tnr90,
                "pix_auroc": pix_auroc,
                "pix_aupr": pix_aupr,
                "spro": spro,
                "auspro": auspro,
            }
            results.append(result)

            pix_str = f"pix={pix_auroc:.4f}" if pix_auroc is not None else "pix=N/A"
            log(f"  {model_name} s{seed} t{train_limit_str}: img={img_auroc:.4f} {pix_str} {train_time:.1f}s")

            del model

    # Cleanup
    del train_features, test_features
    cuda_cleanup()

    return results


def save_result(result):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists()

    def fmt(v, d=4):
        return "" if v is None else f"{v:.{d}f}"

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        if not file_exists:
            writer.writerow([
                "timestamp", "dataset", "model", "image_size",
                "train_limit", "n_train", "n_test", "seed", "epochs",
                "img_auroc", "img_aupr", "img_f1max", "tpr_tnr95", "tpr_tnr90",
                "pix_auroc", "pix_aupr", "spro", "auspro",
                "train_time_s", "gpu_mem_mb"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["dataset"], result["model"], result["image_size"],
            result["train_limit"], result["n_train"], result["n_test"],
            result["seed"], result["epochs"],
            fmt(result["img_auroc"]), fmt(result["img_aupr"]),
            fmt(result.get("img_f1max")), fmt(result.get("tpr_tnr95")),
            fmt(result.get("tpr_tnr90")), fmt(result.get("pix_auroc")),
            fmt(result.get("pix_aupr")), fmt(result.get("spro")),
            fmt(result.get("auspro")), f"{result['train_time']:.1f}",
            f"{result['gpu_mem_mb']:.0f}",
        ])


# =============================================================================
# MAIN
# =============================================================================

def main():
    log("=" * 70)
    log("GALAD BATCH TRAINING - FINAL PAPER VERSION")
    log("=" * 70)
    log(f"Datasets: {len(DATASETS)}")
    log(f"Candidates: {list(CANDIDATES.keys())}")
    log(f"Seeds: {SEEDS}")
    log(f"Train sizes: {TRAIN_SIZES}")
    log(f"Resolution: {RESOLUTION}")

    total = len(DATASETS) * len(CANDIDATES) * len(SEEDS) * len(TRAIN_SIZES)
    log(f"Total experiments: {total}")

    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # Initialize feature extractor (shared)
    log("Loading DINOv3 model...")
    extractor = FeatureExtractor(resolution=RESOLUTION)

    completed = get_completed()
    log(f"Already completed: {len(completed)}")

    errors = []

    for train_size in TRAIN_SIZES:
        for i, dataset in enumerate(DATASETS):
            log(f"\n[{i+1}/{len(DATASETS)}] {dataset} (train_size={train_size})")

            try:
                results = train_dataset(dataset, extractor, completed, train_size)
                for r in results:
                    save_result(r)
            except Exception as e:
                error_msg = f"{dataset}: {e}"
                errors.append(error_msg)
                log(f"ERROR: {error_msg}")
                log_error(dataset, "all", -1, train_size, e)
                cuda_cleanup()

    log("\n" + "=" * 70)
    log("BATCH TRAINING COMPLETED")
    log(f"Total: {total}, Errors: {len(errors)}")
    if errors:
        for e in errors:
            log(f"  - {e}")
    log(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
