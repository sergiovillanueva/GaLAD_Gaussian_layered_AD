"""
Ablation Study for GaLAD Paper

Generates Tables T4 (Layer Selection) and T5 (Pooling Strategy)
plus hyperparameter sensitivity analysis.

Datasets: MVTec AD (15 categories), VisA (12 categories), GoodsAD (6 categories)
Protocol: N=5 training images, 5 seeds (0-4)
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

DATA_ROOT_STR = os.environ.get("GALAD_DATA_ROOT", "data")

_parser = argparse.ArgumentParser(add_help=True)
_parser.add_argument(
    "--device",
    type=int,
    default=None,
    help="GPU device ID (sets CUDA_VISIBLE_DEVICES). If omitted, uses current env.",
)
_parser.add_argument(
    "--data-root",
    type=str,
    default=DATA_ROOT_STR,
    help="Datasets root directory (contains mvtec_AD, VisA, GoodsAD, ...)",
)
_args, _unknown = _parser.parse_known_args()
if _args.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_args.device)
DATA_ROOT_STR = _args.data_root

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
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

ABLATION_DATASETS = [
    # MVTec AD (15 categories)
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
    # VisA (12 categories)
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum",
    "VisA/fryum", "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1",
    "VisA/pcb2", "VisA/pcb3", "VisA/pcb4", "VisA/pipe_fryum",
    # GoodsAD (6 categories)
    "GoodsAD/cigarette_box", "GoodsAD/drink_bottle", "GoodsAD/drink_can",
    "GoodsAD/food_bottle", "GoodsAD/food_box", "GoodsAD/food_package",
]

SEEDS = [0, 1, 2, 3, 4]
TRAIN_SIZE = 5
RESOLUTION = 448
BATCH_SIZE = 56

# Default configuration (baseline)
DEFAULT_CONFIG = {
    "layers": [-6, -12],
    "pca_dim": 256,
    "pca_skip": 2,
    "n_components": 3,
    "smooth_sigma": 1.0,
    "percentile": 95,
}

# Ablation configurations
ABLATIONS = {
    # T4: Layer Selection
    "layers": [
        {"name": "L-1 (final)", "layers": [-1]},
        {"name": "L-6 only", "layers": [-6]},
        {"name": "L-12 only", "layers": [-12]},
        {"name": "L-6, L-12 (ours)", "layers": [-6, -12]},
    ],
    # T5: Pooling Strategy
    "pooling": [
        {"name": "Max", "percentile": 100},
        {"name": "Mean", "percentile": None},  # Special case
        {"name": "p90", "percentile": 90},
        {"name": "p95 (ours)", "percentile": 95},
        {"name": "p99", "percentile": 99},
    ],
    # Hyperparameter: GMM Components
    "gmm_components": [
        {"name": "K=1", "n_components": 1},
        {"name": "K=3 (ours)", "n_components": 3},
        {"name": "K=5", "n_components": 5},
        {"name": "K=10", "n_components": 10},
    ],
    # Hyperparameter: Smoothing Sigma
    "smooth_sigma": [
        {"name": "σ=0", "smooth_sigma": 0.0},
        {"name": "σ=1 (ours)", "smooth_sigma": 1.0},
        {"name": "σ=2", "smooth_sigma": 2.0},
        {"name": "σ=3", "smooth_sigma": 3.0},
    ],
    # Hyperparameter: PCA Skip
    "pca_skip": [
        {"name": "k=0", "pca_skip": 0},
        {"name": "k=1", "pca_skip": 1},
        {"name": "k=2 (ours)", "pca_skip": 2},
        {"name": "k=3", "pca_skip": 3},
    ],
}

OUTPUT_DIR = Path("results/ablations")
OUTPUT_CSV = OUTPUT_DIR / "ablation_results.csv"


# ============================================================================
# GaLAD Model (simplified for ablations)
# ============================================================================

class GaLADAblation:
    """GaLAD model with configurable parameters for ablation studies."""

    _model = None
    _processor = None

    def __init__(
        self,
        layers: List[int] = [-6, -12],
        pca_dim: int = 256,
        pca_skip: int = 2,
        n_components: int = 3,
        smooth_sigma: float = 1.0,
        percentile: int = 95,
        seed: int = 42,
    ):
        self.layers = layers
        self.pca_dim = pca_dim
        self.pca_skip = pca_skip
        self.n_components = n_components
        self.smooth_sigma = smooth_sigma
        self.percentile = percentile
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy load model (shared across instances)
        if GaLADAblation._model is None:
            print("Loading DINOv3 model...")
            model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            GaLADAblation._processor = AutoImageProcessor.from_pretrained(model_name)
            GaLADAblation._model = AutoModel.from_pretrained(
                model_name, output_hidden_states=True
            ).to(self.device)
            GaLADAblation._model.eval()

        self.model = GaLADAblation._model
        self.processor = GaLADAblation._processor
        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.grid_size = RESOLUTION // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.pca_models = {}
        self.gmm_models = {}

    def extract_features(self, image_paths: List[str]) -> List[dict]:
        """Extract features from images."""
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
                with torch.autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                    outputs = self.model(**inputs)

                hidden_states = outputs.hidden_states
                start_idx = 1 + self.num_register_tokens

                for b in range(len(batch_paths)):
                    features = {}
                    for layer_idx in self.layers:
                        layer_hidden = hidden_states[layer_idx][b]
                        patch_tokens = layer_hidden[start_idx:start_idx + self.num_patches].cpu()
                        features[layer_idx] = patch_tokens
                    all_features.append(features)

        return all_features

    def fit(self, train_paths: List[str]):
        """Fit PCA and GMM on training images."""
        features_list = self.extract_features(train_paths)

        all_features = {idx: [] for idx in self.layers}
        for features in features_list:
            for layer_idx in self.layers:
                all_features[layer_idx].append(features[layer_idx])

        for layer_idx in self.layers:
            train_data = torch.cat(all_features[layer_idx], dim=0).numpy()

            # Subsample if needed
            if train_data.shape[0] > 100000:
                np.random.seed(self.seed)
                indices = np.random.choice(train_data.shape[0], 100000, replace=False)
                train_data = train_data[indices]

            # PCA
            total_pca_dim = self.pca_dim + self.pca_skip
            pca = PCA(n_components=total_pca_dim, random_state=self.seed)
            train_pca_full = pca.fit_transform(train_data.astype(np.float32))
            train_pca = train_pca_full[:, self.pca_skip:]
            self.pca_models[layer_idx] = pca

            # GMM
            n_samples = train_pca.shape[0]
            n_features = train_pca.shape[1]
            max_comp = max(2, n_samples // (n_features + 10))
            n_comp = min(self.n_components, max_comp)

            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                reg_covar=1e-2,
                random_state=self.seed,
                n_init=1,
                max_iter=200
            )
            gmm.fit(train_pca)
            self.gmm_models[layer_idx] = gmm

    def score(self, test_paths: List[str]) -> Tuple[List[float], List[np.ndarray]]:
        """Score test images."""
        features_list = self.extract_features(test_paths)
        scores = []
        anomaly_maps = []

        for features in features_list:
            layer_maps = []

            for layer_idx in self.layers:
                patches = features[layer_idx].numpy()
                patches_pca_full = self.pca_models[layer_idx].transform(patches.astype(np.float32))
                patches_pca = patches_pca_full[:, self.pca_skip:]
                patch_nll = -self.gmm_models[layer_idx].score_samples(patches_pca)

                amap = patch_nll.reshape(self.grid_size, self.grid_size)
                if self.smooth_sigma > 0:
                    amap = gaussian_filter(amap, sigma=self.smooth_sigma)
                layer_maps.append(amap)

            # Fuse layers
            fused_map = np.mean(layer_maps, axis=0)
            anomaly_maps.append(fused_map)

            # Compute score
            if self.percentile is None:  # Mean pooling
                score = float(np.mean(fused_map))
            else:
                score = float(np.percentile(fused_map, self.percentile))
            scores.append(score)

        return scores, anomaly_maps


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(data_dir: str) -> Tuple[List[str], List[dict]]:
    """Load train and test images from dataset."""
    data_path = Path(DATA_ROOT_STR) / data_dir
    train_dir = data_path / "train" / "good"
    test_dir = data_path / "test"

    if not train_dir.exists():
        return [], []

    train_paths = sorted([str(p) for p in train_dir.glob("*.*")
                         if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])

    test_info = []
    for category_dir in sorted(test_dir.iterdir()):
        if category_dir.is_dir():
            label = 0 if category_dir.name == "good" else 1
            for img_path in sorted(category_dir.glob("*.*")):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    test_info.append({'path': str(img_path), 'label': label})

    return train_paths, test_info


def sample_train_images(train_paths: List[str], n: int, seed: int) -> List[str]:
    """Sample n training images with given seed."""
    np.random.seed(seed)
    if len(train_paths) <= n:
        return train_paths
    indices = np.random.choice(len(train_paths), n, replace=False)
    return [train_paths[i] for i in sorted(indices)]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_config(
    config: dict,
    datasets: List[str],
    seeds: List[int],
    train_size: int,
) -> Dict[str, List[float]]:
    """Evaluate a configuration across datasets and seeds."""
    all_aurocs = []
    all_auprs = []
    all_auspros = []

    for dataset in datasets:
        train_paths, test_info = load_dataset(dataset)
        if not train_paths or not test_info:
            print(f"  Skipping {dataset} (not found)")
            continue

        test_paths = [t['path'] for t in test_info]
        test_labels = [t['label'] for t in test_info]

        for seed in seeds:
            # Sample training images
            sampled_train = sample_train_images(train_paths, train_size, seed)

            # Create and fit model
            model = GaLADAblation(
                layers=config.get("layers", DEFAULT_CONFIG["layers"]),
                pca_dim=config.get("pca_dim", DEFAULT_CONFIG["pca_dim"]),
                pca_skip=config.get("pca_skip", DEFAULT_CONFIG["pca_skip"]),
                n_components=config.get("n_components", DEFAULT_CONFIG["n_components"]),
                smooth_sigma=config.get("smooth_sigma", DEFAULT_CONFIG["smooth_sigma"]),
                percentile=config.get("percentile", DEFAULT_CONFIG["percentile"]),
                seed=seed,
            )
            model.fit(sampled_train)

            # Score test images
            scores, _ = model.score(test_paths)

            # Compute metrics
            auroc = roc_auc_score(test_labels, scores)
            aupr = average_precision_score(test_labels, scores)

            all_aurocs.append(auroc)
            all_auprs.append(aupr)

    return {
        "auroc": all_aurocs,
        "aupr": all_auprs,
    }


# ============================================================================
# Main
# ============================================================================

def run_ablation(ablation_name: str, configs: List[dict], datasets: List[str]):
    """Run ablation study for a specific parameter."""
    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_name}")
    print(f"{'='*60}")

    results = []

    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n  Config: {name}")

        # Merge with default config
        full_config = DEFAULT_CONFIG.copy()
        full_config.update(cfg)

        # Evaluate
        metrics = evaluate_config(full_config, datasets, SEEDS, TRAIN_SIZE)

        auroc_mean = np.mean(metrics["auroc"])
        auroc_std = np.std(metrics["auroc"])
        aupr_mean = np.mean(metrics["aupr"])
        aupr_std = np.std(metrics["aupr"])

        print(f"    AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
        print(f"    AUPR:  {aupr_mean:.4f} ± {aupr_std:.4f}")

        results.append({
            "ablation": ablation_name,
            "config": name,
            "auroc_mean": auroc_mean,
            "auroc_std": auroc_std,
            "aupr_mean": aupr_mean,
            "aupr_std": aupr_std,
            "n_experiments": len(metrics["auroc"]),
        })

        # Restore name for next iteration
        cfg["name"] = name

    return results


def main():
    """Run all ablation studies."""
    print("GaLAD Ablation Study")
    print(f"Datasets: {len(ABLATION_DATASETS)} categories")
    print(f"Seeds: {SEEDS}")
    print(f"Train size: N={TRAIN_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run each ablation
    for ablation_name, configs in ABLATIONS.items():
        results = run_ablation(ablation_name, configs, ABLATION_DATASETS)
        all_results.extend(results)

        # Save intermediate results
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ablation", "config", "auroc_mean", "auroc_std",
                "aupr_mean", "aupr_std", "n_experiments"
            ])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n  Results saved to {OUTPUT_CSV}")

    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)

    for ablation_name in ABLATIONS.keys():
        print(f"\n{ablation_name}:")
        for r in all_results:
            if r["ablation"] == ablation_name:
                print(f"  {r['config']:20s} AUPR: {r['aupr_mean']:.4f} ± {r['aupr_std']:.4f}")


if __name__ == "__main__":
    main()
