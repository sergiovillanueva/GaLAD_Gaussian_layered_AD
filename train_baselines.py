"""
Batch training script for baseline experiments.

Runs all combinations of datasets, models, seeds, and training sizes.
Results saved incrementally to CSV. Crashes resume from last checkpoint.

Usage: uv run train_baselines.py
"""
import argparse
import os
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='GPU device ID to use')
parser.add_argument(
    '--data-root',
    type=str,
    default=os.environ.get('GALAD_DATA_ROOT', 'data'),
    help='Datasets root directory (contains mvtec_AD, VisA, AutoVI, ...)',
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
GPU_DEVICE = args.device
DATA_ROOT_STR = args.data_root

import torch
torch.set_float32_matmul_precision('medium')

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_LIGHTNING_DISABLE_WARNINGS"] = "1"
os.environ["TIMM_FUSED_ATTN"] = "0"
os.environ.setdefault("WANDB_DISABLED", "true")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")

import copy
import gc
import csv
import json
import shutil
import traceback
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.ndimage import zoom, label as cc_label
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def cuda_cleanup(reason: str = ""):
    """Best-effort CUDA + Python GC cleanup between experiments.

    This doesn't change results; it just helps avoid memory accumulation and
    fragmentation when running many experiments sequentially.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize kernel selection for speed


def apply_train_limit(dm, *, max_train: int | None, seed: int, dataset: str) -> None:
    """Apply a deterministic train-size limit to an Anomalib datamodule.

    Important: Lightning/Anomalib may call `datamodule.setup("fit")` internally.
    If we sample BEFORE Trainer starts, and then `setup` runs again, the split
    can be regenerated and our sampling can be lost. This helper is paired with
    setting `dm._is_setup = True` after we finalize the split.
    """
    if max_train is None:
        return

    if not hasattr(dm, "train_data") or dm.train_data is None or not hasattr(dm.train_data, "samples"):
        raise RuntimeError(f"Datamodule has no train_data.samples for dataset={dataset}")

    samples = dm.train_data.samples
    if samples is None or len(samples) == 0:
        raise RuntimeError(f"Empty train samples for dataset={dataset}")

    n_before = len(samples)

    if n_before > max_train:
        # Match `train_galad.py`: random.sample on a sorted list.
        # Use a local RNG so we don't depend on global `random` state.
        paths = samples["image_path"].astype(str).tolist()
        paths_sorted = sorted(paths)
        rng = random.Random(seed)
        chosen = set(rng.sample(paths_sorted, max_train))

        dm.train_data.samples = samples[samples["image_path"].astype(str).isin(chosen)].copy(deep=True)
        # Keep a stable order to reduce accidental nondeterminism.
        dm.train_data.samples = dm.train_data.samples.sort_values("image_path").reset_index(drop=True)

    n_after = len(dm.train_data.samples)

    # Hard assert: if this fails, you're not actually running few-shot.
    if n_after != min(n_before, max_train):
        raise RuntimeError(
            f"Train limit mismatch for {dataset}: before={n_before}, after={n_after}, max_train={max_train}"
        )

    try:
        log(f"Train limit applied for {dataset}: {n_before} -> {n_after} samples")
    except Exception:
        print(f"Train limit applied for {dataset}: {n_before} -> {n_after} samples")

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

# Models to evaluate
MODELS = [
    "padim",
    "patchcore",
    "reverse_distillation",
    "dinomaly",
    "efficient_ad",
]

SEEDS = [0, 1, 2, 3, 4]

TRAIN_SIZES = [2, 5, 10, 20, 50]

MAX_IMAGE_SIDE = 448

BATCH_SIZE = 8

NUM_WORKERS = 0
EPOCHS = 300

MAX_VIS_IMAGES = 1

CSV_PATH = Path("results/baseline_results.csv")
LOG_PATH = Path("results/baseline_log.txt")
ERROR_LOG_PATH = Path("results/baseline_errors.txt")
TEST_OUTPUT_DIR = Path("results/baseline_test")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_error_details(dataset, model, seed, train_size, image_size, batch_size, exception):
    """Log detailed error info to separate error log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"ERROR TIMESTAMP: {timestamp}\n")
        f.write(f"DATASET:     {dataset}\n")
        f.write(f"MODEL:       {model}\n")
        f.write(f"SEED:        {seed}\n")
        f.write(f"TRAIN_SIZE:  {train_size}\n")
        f.write(f"IMAGE_SIZE:  {image_size}\n")
        f.write(f"BATCH_SIZE:  {batch_size}\n")
        f.write(f"EXCEPTION:   {type(exception).__name__}: {str(exception)}\n")
        f.write("TRACEBACK:\n")
        f.write(traceback.format_exc())
        f.write("=" * 80 + "\n\n")

def get_completed_experiments():
    """Return set of (dataset, model, seed, train_size) already completed.

    train_size is stored in CSV as 'n_train' for full data or the configured limit.
    We use a 'train_limit' column if present, otherwise fall back to n_train.
    """
    completed = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                # Use train_limit if available, else n_train
                train_limit = row.get("train_limit", row["n_train"])
                key = (row["dataset"], row["model"], int(row["seed"]), str(train_limit))
                completed.add(key)
    return completed

def compute_tpr_at_fpr(labels, scores, target_fpr):
    """Compute TPR at a specific FPR threshold."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def compute_f1_max(labels, scores):
    """Compute maximum F1 score over all thresholds."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
    return float(np.max(f1))


def compute_auspro(anomaly_maps, gt_masks, image_labels, alpha=0.3):
    """Compute sPRO and AUsPRO (simplified PRO metric).

    PRO = Per-Region Overlap: measures detection rate per connected component.
    sPRO@alpha = PRO at FPR=alpha
    AUsPRO = Area under sPRO curve up to FPR=alpha, normalized by alpha.
    """
    normal_idx = np.where(np.array(image_labels) == 0)[0]
    abnormal_idx = np.where(np.array(image_labels) == 1)[0]

    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return None, None

    # Get threshold range from normal images
    all_normal_pixels = []
    for idx in normal_idx:
        if idx < len(anomaly_maps):
            all_normal_pixels.append(anomaly_maps[idx].ravel())
    if not all_normal_pixels:
        return None, None
    all_normal_pixels = np.concatenate(all_normal_pixels)

    # Thresholds from high to low
    thresholds = np.linspace(np.max(all_normal_pixels), np.min(all_normal_pixels), 100)

    # Calculate FPR and PRO at each threshold
    total_normal_pixels = len(all_normal_pixels)
    fprs = []
    pros = []

    for thr in thresholds:
        # FPR on normal images
        fp = (all_normal_pixels >= thr).sum()
        fpr = fp / total_normal_pixels
        fprs.append(fpr)

        # PRO on abnormal images (per connected component)
        pro_vals = []
        for idx in abnormal_idx:
            if idx >= len(anomaly_maps):
                continue
            amap = anomaly_maps[idx]
            gtm = gt_masks[idx]

            # Get connected components
            labeled, num_cc = cc_label(gtm)
            for cc_id in range(1, num_cc + 1):
                cc_mask = (labeled == cc_id)
                cc_size = cc_mask.sum()
                if cc_size == 0:
                    continue
                detected = ((amap >= thr) & cc_mask).sum()
                pro_vals.append(detected / cc_size)

        pros.append(np.mean(pro_vals) if pro_vals else 0.0)

    fprs = np.array(fprs)
    pros = np.array(pros)

    # sPRO at FPR <= alpha
    valid = fprs <= alpha
    if not np.any(valid):
        return None, None

    # Sort by FPR for integration
    order = np.argsort(fprs[valid])
    xf = fprs[valid][order]
    yf = pros[valid][order]

    # AUsPRO normalized by alpha
    auspro = np.trapezoid(yf, xf) / alpha
    # sPRO at the edge (closest to alpha)
    spro = float(yf[-1])

    return spro, auspro


def save_test_images(image_tensor, anomaly_map, gt_mask, output_dir, idx, label, size=(256, 256)):
    """Save original image, ground truth mask, and heatmap as separate PNGs."""
    if isinstance(size, int):
        size = (size, size)
    h, w = size
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{idx:04d}_label{label}"

    # 1. Original image (denormalized, resized to model size)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((w, h), Image.BILINEAR)
    img_pil.save(output_dir / f"{prefix}_img.png")

    # 2. Ground truth mask (binary, resized)
    if gt_mask is not None:
        gt = (gt_mask > 0).astype(np.uint8) * 255
        if gt.shape != (h, w):
            gt_pil = Image.fromarray(gt).resize((w, h), Image.NEAREST)
        else:
            gt_pil = Image.fromarray(gt)
        gt_pil.save(output_dir / f"{prefix}_gt.png")

    # 3. Heatmap (jet colormap, no overlay)
    am = anomaly_map
    if am.shape != (h, w):
        am = zoom(am, (h / am.shape[0], w / am.shape[1]), order=1)
    am_norm = (am - am.min()) / (am.max() - am.min() + 1e-8)
    heatmap = (cm.jet(am_norm)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(output_dir / f"{prefix}_heatmap.png")


def compute_metrics(model, datamodule, device, output_dir=None, max_vis=0, image_size=(256, 256)):
    """Compute image and pixel-level metrics, optionally save visualizations and raw data."""
    model.eval()
    model = model.to(device)

    test_loader = datamodule.test_dataloader()

    all_scores = []
    all_labels = []
    all_pixel_scores = []
    all_pixel_labels = []
    # For sPRO: keep 2D maps per image
    anomaly_maps_2d = []
    gt_masks_2d = []
    image_labels_for_pro = []
    # For raw data export
    test_results = []
    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch.image = batch.image.to(device)
            if hasattr(batch, 'gt_label'):
                batch.gt_label = batch.gt_label.to(device)

            outputs = model.test_step(batch, batch_idx)
            labels = batch.gt_label.detach().cpu().numpy().reshape(-1)

            # Image-level scores: use pred_score (Anomalib's native aggregation)
            scores = None
            if hasattr(outputs, 'pred_score') and outputs.pred_score is not None:
                scores = outputs.pred_score.detach().cpu().numpy().reshape(-1)
                all_scores.extend(scores)
                all_labels.extend(labels)

            # Pixel-level metrics (if anomaly_map and masks available)
            if hasattr(outputs, 'anomaly_map') and outputs.anomaly_map is not None:
                amap = outputs.anomaly_map.detach().cpu()

                if hasattr(batch, 'gt_mask') and batch.gt_mask is not None:
                    amap_np = amap.numpy()
                    mask_np = batch.gt_mask.detach().cpu().numpy()

                    # Handle dimensions
                    if amap_np.ndim == 4:
                        amap_np = amap_np[:, 0] if amap_np.shape[1] == 1 else amap_np.mean(axis=1)
                    if mask_np.ndim == 4:
                        mask_np = mask_np[:, 0] if mask_np.shape[1] == 1 else mask_np.max(axis=1)

                    for i in range(amap_np.shape[0]):
                        am = amap_np[i]
                        gt = (mask_np[i] > 0).astype(np.uint8)
                        # Resize if needed
                        if am.shape != gt.shape:
                            zy, zx = gt.shape[0] / am.shape[0], gt.shape[1] / am.shape[1]
                            am = zoom(am, (zy, zx), order=1)
                        all_pixel_scores.append(am.ravel())
                        all_pixel_labels.append(gt.ravel())
                        # Keep 2D for PRO
                        anomaly_maps_2d.append(am)
                        gt_masks_2d.append(gt)
                        image_labels_for_pro.append(int(labels[i]))

                        # Store raw test result
                        global_idx = len(test_results)
                        test_results.append({
                            "idx": global_idx,
                            "label": int(labels[i]),
                            "score": float(scores[i]) if scores is not None and i < len(scores) else None,
                            "has_mask": int(gt.sum() > 0),
                        })

                        # Save test images (original, gt, heatmap)
                        if output_dir and max_vis > 0 and vis_count < max_vis:
                            save_test_images(
                                batch.image[i], am, gt,
                                output_dir / "images", global_idx, int(labels[i]), size=image_size
                            )
                            vis_count += 1

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # NO score inversion - report raw model performance
    # If AUROC < 0.5, the model is failing and should be reported as such

    # Image-level metrics
    img_auroc = roc_auc_score(all_labels, all_scores)
    img_aupr = average_precision_score(all_labels, all_scores)
    img_f1max = compute_f1_max(all_labels, all_scores)
    tpr_tnr95 = compute_tpr_at_fpr(all_labels, all_scores, 0.05)  # TNR=95% means FPR=5%
    tpr_tnr90 = compute_tpr_at_fpr(all_labels, all_scores, 0.10)  # TNR=90% means FPR=10%

    # Pixel-level metrics
    pix_auroc = None
    pix_aupr = None
    spro = None
    auspro = None

    if len(all_pixel_scores) > 0:
        pix_scores = np.concatenate(all_pixel_scores)
        pix_labels = np.concatenate(all_pixel_labels)
        if len(np.unique(pix_labels)) > 1:
            pix_auroc = roc_auc_score(pix_labels, pix_scores)
            pix_aupr = average_precision_score(pix_labels, pix_scores)

        # Compute sPRO/AUsPRO
        if len(anomaly_maps_2d) > 0:
            spro, auspro = compute_auspro(anomaly_maps_2d, gt_masks_2d, image_labels_for_pro)

    # Save raw test data to JSON
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_data = {
            "scores": all_scores.tolist(),
            "labels": all_labels.tolist(),
            "n_test": len(all_scores),
            "n_anomaly": int(all_labels.sum()),
            "test_results": test_results,
        }
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(raw_data, f, indent=2)

    return {
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

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def train_single(dataset, model_name, max_side, max_train, seed):
    """Train and evaluate a single configuration. Returns dict with results."""
    import random

    import pandas as pd

    try:
        from anomalib.data import Folder, MVTecLOCO
        from anomalib.data.datamodules.base.image import AnomalibDataModule
        from anomalib.data.datasets.base.image import AnomalibDataset
        from anomalib.data.utils import Split
        from anomalib.engine import Engine
        from anomalib.models import Patchcore, Padim, ReverseDistillation, Dinomaly, EfficientAd
        from lightning.pytorch.callbacks import EarlyStopping
    except Exception as e:
        raise SystemExit(
            "Anomalib is required for baseline experiments. Install with: uv sync --extra baselines (or pip install anomalib). "
            f"Original error: {type(e).__name__}: {e}"
        )

    class VisAOfficialSplitDataModule(AnomalibDataModule):
        """VisA datamodule using the official `split_csv/1cls.csv` split.

        This mirrors the split logic used in `train_galad.load_visa_dataset`.
        It intentionally avoids directory-based splitting to prevent leakage.
        """

        def __init__(
            self,
            *,
            category: str,
            visa_root: Path,
            split_csv_path: Path,
            train_batch_size: int,
            eval_batch_size: int,
            num_workers: int,
            seed: int | None,
        ) -> None:
            super().__init__(
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                val_split_mode="none",
                test_split_mode="none",
                seed=seed,
            )
            self._category = category
            self._visa_root = visa_root
            self._split_csv_path = split_csv_path

        @property
        def name(self) -> str:
            return f"VisA_{self._category}"

        def _create_test_split(self) -> None:
            # Do not alter the provided official split.
            return

        def _create_val_split(self) -> None:
            # Validation is created explicitly in _setup.
            return

        def _setup(self, _stage: str | None = None) -> None:
            if not self._split_csv_path.exists():
                raise FileNotFoundError(
                    f"VisA split CSV not found: {self._split_csv_path}. "
                    "Expected data/VisA/split_csv/1cls.csv"
                )

            df = pd.read_csv(self._split_csv_path)
            df = df[df["object"] == self._category]
            if df.empty:
                raise ValueError(f"No rows for VisA category '{self._category}' in {self._split_csv_path}")

            # Build official train (normal-only) and test (normal+anomaly) file lists.
            train_rows = df[df["split"] == "train"]
            test_rows = df[df["split"] == "test"]

            def _row_to_path(rel_path: str) -> str:
                p = (self._visa_root / str(rel_path)).resolve()
                return str(p)

            train_paths = [_row_to_path(p) for p in train_rows["image"].tolist()]
            test_paths = [_row_to_path(p) for p in test_rows["image"].tolist()]

            # Labels
            test_label_names = test_rows["label"].astype(str).tolist()
            test_label_index = [0 if lbl == "normal" else 1 for lbl in test_label_names]
            train_label_index = [0 for _ in train_paths]

            # Masks (optional)
            def _mask_to_path(v) -> str:
                if v is None:
                    return ""
                s = str(v)
                if s.lower() == "nan" or not s.strip():
                    return ""
                mp = (self._visa_root / s).resolve()
                return str(mp) if mp.exists() else ""

            test_masks = [_mask_to_path(v) for v in test_rows["mask"].tolist()]
            train_masks = ["" for _ in train_paths]

            df_train = pd.DataFrame(
                {
                    "image_path": train_paths,
                    "mask_path": train_masks,
                    "label_index": train_label_index,
                    "split": [Split.TRAIN] * len(train_paths),
                }
            )
            df_test = pd.DataFrame(
                {
                    "image_path": test_paths,
                    "mask_path": test_masks,
                    "label_index": test_label_index,
                    "split": [Split.TEST] * len(test_paths),
                }
            )

            # Ensure attrs['task'] exists (required by AnomalibDataset.task)
            task = "segmentation" if any(m for m in df_test["mask_path"].tolist()) else "classification"
            df_train.attrs["task"] = task
            df_test.attrs["task"] = task

            self.train_data = AnomalibDataset()
            self.test_data = AnomalibDataset()
            self.val_data = AnomalibDataset()
            self.train_data.samples = df_train
            self.test_data.samples = df_test

            # Non-leaky validation: normal-only
            df_val = df_train.copy(deep=True)
            df_val.attrs["task"] = task
            self.val_data.samples = df_val

            # Sanity: overlap check
            overlap = len(set(df_train["image_path"]) & set(df_test["image_path"]))
            if overlap > 0:
                logger_msg = f"WARNING: VisA official split overlap detected ({overlap}) for {self._category}."
                try:
                    log(logger_msg)
                except Exception:
                    print(logger_msg)

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    cuda_cleanup("start train_single")

    # Check if this is a VisA dataset (official split CSV)
    is_visa = dataset.startswith("VisA/")

    # Check if this is a LOCO dataset (needs special datamodule)
    is_loco = dataset.startswith("mvtec_loco_AD/")

    if is_visa:
        category = dataset.split("/")[1]
        visa_root = Path(DATA_ROOT_STR) / "VisA"
        split_csv_path = visa_root / "split_csv" / "1cls.csv"
        dm = VisAOfficialSplitDataModule(
            category=category,
            visa_root=visa_root,
            split_csv_path=split_csv_path,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
        )
        dm.setup()

    elif is_loco:
        # Use MVTecLOCO datamodule (handles multi-mask format)
        category = dataset.split("/")[1]
        data_root = Path(DATA_ROOT_STR) / "mvtec_loco_AD" / category
        dm = MVTecLOCO(
            root=str(Path(DATA_ROOT_STR) / "mvtec_loco_AD"),
            category=category,
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            val_split_mode="from_dir",  # LOCO has validation folder
            seed=seed,
        )
        dm.setup()
    else:
        # Standard Folder datamodule for MVTec AD and AutoVI
        data_root = Path(DATA_ROOT_STR) / dataset
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {data_root}")

        # Get abnormal dirs and mask dirs
        test_dir = data_root / "test"
        gt_dir = data_root / "ground_truth"
        abnormal_dirs = [d.name for d in test_dir.iterdir()
                         if d.is_dir() and d.name != "good"]

        if not abnormal_dirs:
            raise ValueError(f"No abnormal dirs found in {test_dir}")

        # Build mask_dir mapping if ground_truth exists
        mask_dir = None
        if gt_dir.exists():
            mask_dir = [f"ground_truth/{d}" for d in abnormal_dirs]
            if len(mask_dir) == 1:
                mask_dir = mask_dir[0]

        # Build abnormal_dir paths
        abnormal_dir = [f"test/{d}" for d in abnormal_dirs]
        if len(abnormal_dir) == 1:
            abnormal_dir = abnormal_dir[0]

        # Create datamodule
        # IMPORTANT: Do NOT use the test set for validation.
        # In Anomalib, validation can be used to calibrate thresholds/normalization
        # (e.g., hooks such as `on_validation_start`), which would leak test.
        # We instead create a validation set from the (possibly few-shot limited)
        # training samples after `setup()`.
        dm = Folder(
            name=dataset.replace("/", "_"),
            root=str(data_root),
            normal_dir="train/good",
            normal_test_dir="test/good",
            abnormal_dir=abnormal_dir,
            mask_dir=mask_dir,
            val_split_mode="none",
            train_batch_size=BATCH_SIZE,
            eval_batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            seed=seed,
        )
        dm.setup()

    image_size = (max_side, max_side)

    # Apply few-shot limit BEFORE Trainer starts.
    # NOTE: If Lightning calls `dm.setup("fit")` later, it can recreate the split
    # and undo this sampling unless we lock the setup flag.
    apply_train_limit(dm, max_train=max_train, seed=seed, dataset=dataset)

    # Create a non-leaky validation set.
    # Keep validation identical to training (normal-only) to trigger Anomalib's
    # validation hooks without exposing test samples.
    dm.val_data = copy.deepcopy(dm.train_data)

    # Critical: prevent AnomalibDataModule.setup from re-running _setup() inside
    # Lightning Trainer (which would restore the full training set and explode VRAM).
    if hasattr(dm, "_is_setup"):
        dm._is_setup = True

    # Optional sanity check: detect obvious train/test overlap.
    # Paths should never overlap across splits; if they do, the dataset on disk
    # is likely contaminated.
    try:
        train_paths = set(map(str, dm.train_data.samples["image_path"].tolist()))
        test_paths = set(map(str, dm.test_data.samples["image_path"].tolist()))
        overlap = len(train_paths & test_paths)
        if overlap > 0:
            log(f"WARNING: Detected {overlap} overlapping image paths between train and test for {dataset}.")
    except Exception:
        pass

    n_train = len(dm.train_data.samples)
    n_test = len(dm.test_data.samples)

    # Create model and configure epochs
    callbacks = []

    if model_name == "patchcore":
        pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
        model = Patchcore(pre_processor=pre_processor)
        max_epochs = 1
        precision = None  # No gradient training
    elif model_name == "padim":
        pre_processor = Padim.configure_pre_processor(image_size=image_size)
        model = Padim(pre_processor=pre_processor)
        max_epochs = 1
        precision = None  # No gradient training
    elif model_name == "reverse_distillation":
        pre_processor = ReverseDistillation.configure_pre_processor(image_size=image_size)
        model = ReverseDistillation(pre_processor=pre_processor)
        max_epochs = EPOCHS
        precision = "16-mixed"
        callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=15, min_delta=1e-5))
    elif model_name == "dinomaly":
        pre_processor = Dinomaly.configure_pre_processor(image_size=image_size)
        model = Dinomaly(pre_processor=pre_processor)
        max_epochs = EPOCHS
        precision = None
        callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=15, min_delta=1e-5))
    elif model_name == "efficient_ad":
        pre_processor = EfficientAd.configure_pre_processor(image_size=image_size)
        model = EfficientAd(pre_processor=pre_processor, batch_size=1)
        max_epochs = EPOCHS
        precision = None
        callbacks.append(EarlyStopping(monitor="train_loss", mode="min", patience=15, min_delta=1e-5))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Temporary dir for model checkpoints (will be deleted after training)
    model_dir = Path(f"results/baseline/{dataset.replace('/', '_')}/{model_name}/seed{seed}")

    # Create engine
    engine = Engine(
        max_epochs=max_epochs,
        max_steps=-1 if max_epochs == 1 else 1500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        default_root_dir=str(model_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
        check_val_every_n_epoch=1 if max_epochs == 1 else 1000,  # Infrequent validation for speed
    )

    # Train
    cuda_cleanup("before engine.fit")
    start_time = datetime.now()

    engine.fit(datamodule=dm, model=model)
    
    train_time = (datetime.now() - start_time).total_seconds()
    gpu_mem = get_gpu_memory_mb()

    # Free training caches before running custom test loop
    cuda_cleanup("after engine.fit")

    # For multi-epoch models (RD), use trainer's count
    epochs_trained = max_epochs if max_epochs == 1 else engine.trainer.current_epoch + 1

    # Output dir for test results and visualizations
    train_label = "all" if max_train is None else str(max_train)
    test_output_dir = TEST_OUTPUT_DIR / dataset.replace("/", "_") / model_name / f"seed{seed}_train{train_label}"

    # Test using custom metrics, save results and heatmaps
    metrics = compute_metrics(
        model, dm, device,
        output_dir=test_output_dir,
        max_vis=MAX_VIS_IMAGES,
        image_size=image_size  # tuple (h,w)
    )

    cuda_cleanup("after compute_metrics")

    # Aggressive cleanup to prevent memory leaks
    del model
    del engine
    del dm
    cuda_cleanup("end train_single")

    # Delete model checkpoints to save disk space
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)
    # Clean empty parent dirs
    for parent in [model_dir.parent, model_dir.parent.parent]:
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    return {
        "dataset": dataset,
        "model": model_name,
        "image_size": f"{image_size[0]}x{image_size[1]}",
        "train_limit": "all" if max_train is None else max_train,
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "epochs": epochs_trained,
        "train_time": train_time,
        "gpu_mem_mb": gpu_mem,
        **metrics,
    }

def validate_metrics(result):
    """Validate that all metrics are in expected range [0, 1]."""
    metric_keys = ["img_auroc", "img_aupr", "img_f1max", "tpr_tnr95", "tpr_tnr90",
                   "pix_auroc", "pix_aupr", "spro", "auspro"]
    errors = []
    for key in metric_keys:
        val = result.get(key)
        if val is None:
            continue
        if not np.isfinite(val) or (val < 0 or val > 1):
            errors.append(f"{key}={val}")
    if errors:
        log(f"WARNING: Invalid metric values detected: {', '.join(errors)}")
        log(f"  Dataset: {result.get('dataset')}, Model: {result.get('model')}, Seed: {result.get('seed')}")
    return len(errors) == 0


def save_result(result):
    """Append result to CSV."""
    # Validate metrics before saving
    validate_metrics(result)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists()

    def fmt(v, decimals=4):
        if v is None:
            return ""
        return f"{v:.{decimals}f}"

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
            result["dataset"],
            result["model"],
            result["image_size"],
            result["train_limit"],
            result["n_train"],
            result["n_test"],
            result["seed"],
            result["epochs"],
            fmt(result["img_auroc"]),
            fmt(result["img_aupr"]),
            fmt(result.get("img_f1max")),
            fmt(result.get("tpr_tnr95")),
            fmt(result.get("tpr_tnr90")),
            fmt(result.get("pix_auroc")),
            fmt(result.get("pix_aupr")),
            fmt(result.get("spro")),
            fmt(result.get("auspro")),
            f"{result['train_time']:.1f}",
            f"{result['gpu_mem_mb']:.0f}",
        ])

# =============================================================================
# MAIN
# =============================================================================

def main():
    log("BASELINE BATCH TRAINING")
    log(f"Datasets: {len(DATASETS)}")
    log(f"Models: {MODELS}")
    log(f"Seeds: {SEEDS}")
    log(f"Train sizes: {TRAIN_SIZES}")
    log(f"Max image side: {MAX_IMAGE_SIDE}")
    log(f"Batch size: {BATCH_SIZE}")

    # Count total experiments
    total = len(DATASETS) * len(MODELS) * len(SEEDS) * len(TRAIN_SIZES)
    log(f"Total experiments: {total}")

    # Check CUDA
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"Device: {torch.cuda.get_device_name(0)}")

    cuda_cleanup("start main")
    
    # Get completed experiments
    completed = get_completed_experiments()
    log(f"Already completed: {len(completed)}")

    # Run experiments
    current = 0
    errors = []

    for train_size in TRAIN_SIZES:
        for seed in SEEDS:
            for model in MODELS:
                for dataset in DATASETS:
                    current += 1
                    train_limit_str = str(train_size) if train_size else "all"
                    key = (dataset, model, seed, train_limit_str)

                    if key in completed:
                        continue

                    # PaDiM at N=2 is ill-posed (singular covariance) and triggers
                    # an Anomalib datamodule bug with empty abnormal masks. Skip it.
                    if model == "padim" and train_size == 2:
                        log(f"[{current}/{total}] SKIP {dataset} | padim | seed={seed} | train=2 (N=2 not supported)")
                        continue

                    # Dinomaly requires N>=5; skip at N=2.
                    if model == "dinomaly" and train_size == 2:
                        log(f"[{current}/{total}] SKIP {dataset} | dinomaly | seed={seed} | train=2 (N>=5 required)")
                        continue

                    # EfficientAD's default Anomalib configuration is not stable at N=2.
                    if model == "efficient_ad" and train_size == 2:
                        log(f"[{current}/{total}] SKIP {dataset} | efficient_ad | seed={seed} | train=2 (N=2 not supported)")
                        continue

                    log(f"[{current}/{total}] {dataset} | {model} | seed={seed} | train={train_limit_str}")
                    cuda_cleanup("before experiment")

                    try:
                        result = train_single(
                            dataset=dataset,
                            model_name=model,
                            max_side=MAX_IMAGE_SIDE,
                            max_train=train_size,
                            seed=seed,
                        )
                        save_result(result)
                        pix_str = f"pix={result['pix_auroc']:.4f}" if result.get('pix_auroc') else "pix=N/A"
                        pro_str = f"auspro={result['auspro']:.4f}" if result.get('auspro') else ""
                        log(f"[{current}/{total}] DONE  img={result['img_auroc']:.4f} | {pix_str} | {pro_str} | {result['train_time']:.1f}s")

                    except Exception as e:
                        error_msg = f"{dataset} | {model} | seed={seed}: {str(e)}"
                        errors.append(error_msg)
                        log(f"[{current}/{total}] ERROR {error_msg}")
                        log_error_details(
                            dataset=dataset,
                            model=model,
                            seed=seed,
                            train_size=train_size,
                            image_size=MAX_IMAGE_SIDE,
                            batch_size=BATCH_SIZE,
                            exception=e
                        )

                        # Aggressive cleanup on error
                        cuda_cleanup("after experiment error")

    # Summary
    log("BATCH TRAINING COMPLETED")
    log(f"Total experiments: {total}")
    log(f"Errors: {len(errors)}")

    if errors:
        log("Failed experiments:")
        for e in errors:
            log(f"  - {e}")

    log(f"Results saved to: {CSV_PATH}")
    log(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
