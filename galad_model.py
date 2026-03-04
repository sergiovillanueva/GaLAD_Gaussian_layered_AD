"""
GaLAD: Gaussian Layered Anomaly Detector

Few-shot anomaly detection using DINOv3 intermediate features and GMM density estimation.

Default configuration (GaLAD A - best overall):
- layers: [-6, -12]
- pca_skip: 2 (skip first 2 PCA components)
- n_components: 3 (GMM components)
- pca_dim: 256
- resolution: 448
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Tuple
import warnings
import transformers
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GaLAD:
    """
    Gaussian Layered Anomaly Detector

    Uses DINOv3 features with GMM for few-shot anomaly detection.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        resolution: int = 448,
        layers: List[int] = [-6, -12],
        pca_dim: int = 256,
        pca_skip: int = 2,
        n_components: int = 3,
        smooth_sigma: float = 1.0,
        seed: Optional[int] = 42,
        device: Optional[str] = None
    ):
        """
        Initialize GaLAD model.

        Args:
            model_name: HuggingFace model name for DINOv3 (ViT-L/16 recommended)
            resolution: Input image resolution (448 recommended)
            layers: Which layers to extract features from (default: [-6, -12])
            pca_dim: PCA output dimensions (default: 256)
            pca_skip: Number of top PCA components to skip (default: 2)
            n_components: Number of GMM components (default: 3)
            smooth_sigma: Gaussian smoothing sigma for anomaly maps (default: 1.0)
            seed: Random seed for PCA/GMM
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.resolution = resolution
        self.layers = layers
        self.pca_dim = pca_dim
        self.pca_skip = pca_skip
        self.n_components = n_components
        self.smooth_sigma = smooth_sigma
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"Loading DINOv3 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.model.eval()

        self.patch_size = self.model.config.patch_size
        self.num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        self.grid_size = resolution // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        print(f"  Resolution: {resolution}, Grid: {self.grid_size}x{self.grid_size}")
        print(f"  Layers: {layers}, PCA dim: {pca_dim}, Skip: {pca_skip}")
        print(f"  GMM components: {n_components}, Smooth sigma: {smooth_sigma}")

        self.pca_models = {}
        self.gmm_models = {}
        self.is_fitted = False

    def extract_features_batch(self, image_paths: List[str], batch_size: int = 8) -> List[dict]:
        """Extract patch features from multiple images."""
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                image = Image.open(path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_resized = image.resize((self.resolution, self.resolution), Image.BICUBIC)
                images.append(image_resized)

            inputs = self.processor(
                images=images,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False
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

    def _fit_from_features(self, batch_features: List[dict], verbose: bool = False) -> "GaLAD":
        """Fit PCA + GMM from pre-extracted features (internal method for benchmarking)."""
        all_features = {idx: [] for idx in self.layers}
        for features in batch_features:
            for layer_idx in self.layers:
                all_features[layer_idx].append(features[layer_idx])

        max_patches = 100000

        for layer_idx in self.layers:
            train_data = torch.cat(all_features[layer_idx], dim=0).numpy()

            if train_data.shape[0] > max_patches:
                np.random.seed(self.seed)
                indices = np.random.choice(train_data.shape[0], max_patches, replace=False)
                train_subset = train_data[indices]
            else:
                train_subset = train_data

            # PCA with skip
            total_pca_dim = self.pca_dim + self.pca_skip
            pca = PCA(n_components=total_pca_dim, random_state=self.seed)
            train_pca_full = pca.fit_transform(train_subset.astype(np.float32))
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

        self.is_fitted = True
        return self

    def fit(self, train_image_paths: List[str], batch_size: int = 8, verbose: bool = True) -> "GaLAD":
        """Fit GaLAD on training images."""
        if verbose:
            print(f"Fitting GaLAD on {len(train_image_paths)} images...")

        batch_features = self.extract_features_batch(train_image_paths, batch_size=batch_size)

        all_features = {idx: [] for idx in self.layers}
        for features in batch_features:
            for layer_idx in self.layers:
                all_features[layer_idx].append(features[layer_idx])

        max_patches = 100000

        for layer_idx in self.layers:
            train_data = torch.cat(all_features[layer_idx], dim=0).numpy()

            if verbose:
                print(f"  Layer {layer_idx}: {train_data.shape}")

            if train_data.shape[0] > max_patches:
                np.random.seed(self.seed)
                indices = np.random.choice(train_data.shape[0], max_patches, replace=False)
                train_subset = train_data[indices]
            else:
                train_subset = train_data

            # PCA with skip
            total_pca_dim = self.pca_dim + self.pca_skip
            pca = PCA(n_components=total_pca_dim, random_state=self.seed)
            train_pca_full = pca.fit_transform(train_subset.astype(np.float32))
            train_pca = train_pca_full[:, self.pca_skip:]  # Skip first components
            self.pca_models[layer_idx] = pca

            if verbose:
                print(f"    PCA variance: {pca.explained_variance_ratio_.sum():.4f}")

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

            if verbose:
                print(f"    GMM components: {n_comp}, converged: {gmm.converged_}")

        self.is_fitted = True
        return self

    def score_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        Compute anomaly scores and maps for multiple images.

        Returns:
            (scores, anomaly_maps): Ensemble scores and mean anomaly maps
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        all_features = self.extract_features_batch(image_paths, batch_size=batch_size)
        scores = []
        anomaly_maps = []

        for features in all_features:
            layer_scores = []
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
                layer_scores.append(float(np.percentile(amap, 95)))

            scores.append(float(np.mean(layer_scores)))
            anomaly_maps.append(np.mean(layer_maps, axis=0))

        return scores, anomaly_maps

    def score(self, image_path: str) -> float:
        """Compute anomaly score for a single image."""
        scores, _ = self.score_batch([image_path], batch_size=1)
        return scores[0]

    def get_anomaly_map(self, image_path: str) -> np.ndarray:
        """Get anomaly map for a single image."""
        _, maps = self.score_batch([image_path], batch_size=1)
        return maps[0]


def load_dataset(data_dir: str) -> Tuple[List[str], List[dict]]:
    """Load train and test images from MVTec-style dataset."""
    data_path = Path(data_dir)
    train_dir = data_path / "train" / "good"
    test_dir = data_path / "test"

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


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    print("GaLAD Model Test")
    print("=" * 60)

    model = GaLAD()

    train_paths, test_info = load_dataset("data/mvtec_AD/bottle")
    print(f"\nDataset: bottle")
    print(f"Train: {len(train_paths)}, Test: {len(test_info)}")

    model.fit(train_paths[:20])

    test_paths = [t['path'] for t in test_info]
    test_labels = [t['label'] for t in test_info]

    scores, _ = model.score_batch(test_paths, batch_size=8)
    auroc = roc_auc_score(test_labels, scores)
    print(f"\nAUROC: {auroc:.4f}")
