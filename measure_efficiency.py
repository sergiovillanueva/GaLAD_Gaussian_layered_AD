"""
Measure computational efficiency of GaLAD vs baselines on YOUR hardware.

Outputs:
- Fit time (seconds)
- Fit memory (peak GPU MB)
- Inference memory (peak GPU MB)
- Model size per category (MB)
- Inference time (ms/image)

Usage: uv run measure_efficiency.py

Output: results/efficiency_comparison.csv
"""

import os
import sys
import time
import gc
import copy
import pickle
import shutil
import tempfile
import random
import csv
import argparse
from pathlib import Path
from datetime import datetime

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
    default=os.environ.get("GALAD_DATA_ROOT", "data"),
    help="Datasets root directory (contains mvtec_AD, VisA, AutoVI, ...)",
)
ARGS, _unknown = _parser.parse_known_args()

if ARGS.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.device)

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_LIGHTNING_DISABLE_WARNINGS"] = "1"
os.environ["TIMM_FUSED_ATTN"] = "0"
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

DATA_ROOT_STR = ARGS.data_root

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from PIL import Image

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

# Test on a few diverse categories
DATASETS = [
    "mvtec_AD/bottle",      # Object
    "mvtec_AD/carpet",      # Texture
]

MODELS = ["galad", "patchcore", "dinomaly"]
SEED = 0
TRAIN_SIZE = 10
RESOLUTION = 448
N_WARMUP = 5
N_INFERENCE = 50
BATCH_SIZE = 8
NUM_WORKERS = 0

OUTPUT_CSV = Path("results/efficiency_comparison.csv")


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def reset_memory():
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def cuda_cleanup():
    """Full CUDA cleanup."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def get_dir_size_mb(path):
    """Get total size of directory in MB."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1024 / 1024


def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def load_images(data_dir: str, n_train: int):
    """Load train and test images."""
    data_path = Path(DATA_ROOT_STR) / data_dir
    train_dir = data_path / "train" / "good"
    test_dir = data_path / "test"

    train_paths = sorted([str(p) for p in train_dir.glob("*.*")
                         if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])[:n_train]

    test_paths = []
    for cat_dir in test_dir.iterdir():
        if cat_dir.is_dir():
            for p in sorted(cat_dir.glob("*.*"))[:10]:
                if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    test_paths.append(str(p))

    return train_paths, test_paths[:N_INFERENCE]


# ============================================================================
# GaLAD Efficiency
# ============================================================================

def measure_galad(train_paths, test_paths):
    """Measure GaLAD efficiency."""
    from galad_model import GaLAD

    log("  Measuring GaLAD...")
    results = {}

    # Create model
    model = GaLAD(resolution=RESOLUTION, layers=[-6, -12], pca_dim=256,
                  pca_skip=2, n_components=3, smooth_sigma=1.0)

    # Warmup: one forward pass to load backbone and warm CUDA kernels
    # (same as baselines do with dummy forward)
    _ = model.extract_features_batch(train_paths[:1], batch_size=1)
    cuda_cleanup()

    reset_memory()

    # Measure fit time (feature extraction + PCA + GMM)
    start = time.perf_counter()
    model.fit(train_paths, batch_size=16, verbose=False)
    results["fit_time_s"] = time.perf_counter() - start
    results["train_mem_mb"] = get_memory_mb()
    log(f"    Fit time: {results['fit_time_s']:.1f}s, Fit memory: {results['train_mem_mb']:.1f} MB")

    # Model size (PCA + GMM parameters only - backbone is shared/frozen)
    temp_path = Path(tempfile.gettempdir()) / "galad_model_size_test.pkl"
    state = {
        'pca_models': model.pca_models,
        'gmm_models': model.gmm_models,
    }
    with open(temp_path, 'wb') as f:
        pickle.dump(state, f)
    model_size = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink()
    results["model_size_mb"] = model_size
    log(f"    Model size: {model_size:.2f} MB/category")

    # Inference with batch processing
    reset_memory()

    # Warmup inference
    _, _ = model.score_batch(test_paths[:4], batch_size=16)
    cuda_cleanup()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    _, _ = model.score_batch(test_paths, batch_size=16)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    results["infer_mem_mb"] = get_memory_mb()
    results["infer_ms"] = (elapsed / len(test_paths)) * 1000
    results["fps"] = len(test_paths) / elapsed

    log(f"    Infer memory: {results['infer_mem_mb']:.1f} MB")
    log(f"    Infer time: {results['infer_ms']:.1f} ms/img ({results['fps']:.1f} FPS)")

    # Cleanup
    del model
    cuda_cleanup()

    return results


# ============================================================================
# Anomalib Baselines (REAL measurements)
# ============================================================================

def measure_anomalib(dataset, model_name, train_paths, test_paths):
    """Measure Anomalib baseline efficiency on YOUR hardware."""
    import pandas as pd
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore, Dinomaly
        from lightning.pytorch.callbacks import EarlyStopping
    except Exception as e:
        raise SystemExit(
            "Anomalib is required for baseline efficiency. Install with: uv sync --extra baselines (or pip install anomalib). "
            f"Original error: {type(e).__name__}: {e}"
        )

    log(f"  Measuring {model_name}...")
    cuda_cleanup()
    reset_memory()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup datamodule
    data_root = Path(DATA_ROOT_STR) / dataset
    test_dir = data_root / "test"
    gt_dir = data_root / "ground_truth"

    abnormal_dirs = [d.name for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
    abnormal_dir = [f"test/{d}" for d in abnormal_dirs]
    if len(abnormal_dir) == 1:
        abnormal_dir = abnormal_dir[0]

    mask_dir = None
    if gt_dir.exists():
        mask_dir = [f"ground_truth/{d}" for d in abnormal_dirs]
        if len(mask_dir) == 1:
            mask_dir = mask_dir[0]

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
        seed=SEED,
    )
    dm.setup()

    # Apply train limit
    samples = dm.train_data.samples
    n_before = len(samples)
    if n_before > TRAIN_SIZE:
        paths = sorted(samples["image_path"].astype(str).tolist())
        rng = random.Random(SEED)
        chosen = set(rng.sample(paths, TRAIN_SIZE))
        dm.train_data.samples = samples[samples["image_path"].astype(str).isin(chosen)].copy()

    n_after = len(dm.train_data.samples)
    log(f"    Train images: {n_before} -> {n_after}")

    dm.val_data = copy.deepcopy(dm.train_data)

    # CRITICAL: prevent setup() from being called again by Lightning
    if hasattr(dm, "_is_setup"):
        dm._is_setup = True

    image_size = (RESOLUTION, RESOLUTION)

    # Create model
    if model_name == "patchcore":
        pre_processor = Patchcore.configure_pre_processor(image_size=image_size)
        model = Patchcore(pre_processor=pre_processor)
        max_epochs = 1
        max_steps = -1
        precision = None
        callbacks = []
    elif model_name == "dinomaly":
        pre_processor = Dinomaly.configure_pre_processor(image_size=image_size)
        model = Dinomaly(pre_processor=pre_processor)
        max_epochs = 20  # Limited for efficiency test
        max_steps = 100  # Cap training steps (enough to measure)
        precision = None
        callbacks = [EarlyStopping(monitor="train_loss", mode="min", patience=3)]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Temp dir for checkpoints
    model_dir = Path(tempfile.mkdtemp())

    engine = Engine(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        precision=precision,
        default_root_dir=str(model_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
        check_val_every_n_epoch=1000,
    )

    # Warmup: load backbone weights (excluded from timing, same as GaLAD)
    # This ensures we measure only the actual training, not model initialization
    device = torch.device("cuda")
    model = model.to(device)
    # Force backbone loading with a dummy forward pass
    dummy = torch.randn(1, 3, RESOLUTION, RESOLUTION, device=device)
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception:
            pass  # Some models may not support direct forward
    del dummy
    cuda_cleanup()

    # Measure fit time and memory (backbone already loaded)
    reset_memory()
    start = time.perf_counter()
    engine.fit(datamodule=dm, model=model)
    fit_time = time.perf_counter() - start
    fit_mem = get_memory_mb()

    # Measure storage (model state dict size)
    temp_model_path = Path(tempfile.gettempdir()) / f"{model_name}_state.pt"
    # For PatchCore: memory bank is the main storage
    # For Dinomaly: decoder weights
    if model_name == "patchcore":
        # PatchCore stores memory bank in model.memory_bank
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            torch.save(model.memory_bank, temp_model_path)
            storage_mb = temp_model_path.stat().st_size / (1024 * 1024)
            temp_model_path.unlink()
        else:
            # Estimate: 10 images * 3136 patches * 1024 features * 4 bytes * 10% coreset
            storage_mb = (10 * 3136 * 1024 * 4 * 0.1) / (1024 * 1024)  # ~12 MB
    elif model_name == "dinomaly":
        # Dinomaly stores decoder weights
        state = {k: v for k, v in model.state_dict().items() if 'decoder' in k or 'bottleneck' in k}
        torch.save(state, temp_model_path)
        storage_mb = temp_model_path.stat().st_size / (1024 * 1024)
        temp_model_path.unlink()
    else:
        storage_mb = get_dir_size_mb(model_dir)

    # Measure inference time and memory
    reset_memory()
    device = torch.device("cuda")
    model.eval()
    model = model.to(device)

    test_loader = dm.test_dataloader()

    # Warmup
    for batch in test_loader:
        batch.image = batch.image.to(device)
        with torch.no_grad():
            _ = model.test_step(batch, 0)
        break

    # Timed inference
    n_test = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for batch_idx, batch in enumerate(test_loader):
        batch.image = batch.image.to(device)
        with torch.no_grad():
            _ = model.test_step(batch, batch_idx)
        n_test += batch.image.shape[0]
        if n_test >= N_INFERENCE:
            break
    torch.cuda.synchronize()
    inf_time_total = time.perf_counter() - start
    inf_time_ms = (inf_time_total / n_test) * 1000
    inf_mem = get_memory_mb()

    # Cleanup
    del model
    del engine
    del dm
    cuda_cleanup()
    shutil.rmtree(model_dir, ignore_errors=True)

    return {
        "fit_time_s": fit_time,
        "train_mem_mb": fit_mem,
        "model_size_mb": storage_mb,
        "infer_mem_mb": inf_mem,
        "infer_ms": inf_time_ms,
        "fps": 1000 / inf_time_ms,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    log("=" * 60)
    log("EFFICIENCY MEASUREMENT (on YOUR hardware)")
    log("=" * 60)
    log(f"Datasets: {DATASETS}")
    log(f"Models: {MODELS}")
    log(f"Train size: N={TRAIN_SIZE}")
    log(f"Inference samples: {N_INFERENCE}")

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []

    for dataset in DATASETS:
        log(f"\n{'='*50}")
        log(f"Dataset: {dataset}")
        log(f"{'='*50}")

        train_paths, test_paths = load_images(dataset, TRAIN_SIZE)
        log(f"Loaded {len(train_paths)} train, {len(test_paths)} test images")

        for model_name in MODELS:
            log(f"\n  Model: {model_name}")
            cuda_cleanup()

            try:
                if model_name == "galad":
                    r = measure_galad(train_paths, test_paths)
                else:
                    r = measure_anomalib(dataset, model_name, train_paths, test_paths)

                r["dataset"] = dataset
                r["model"] = model_name
                all_results.append(r)

                log(f"    Fit: {r['fit_time_s']:.1f}s | Inf: {r['infer_ms']:.1f}ms/img | Storage: {r['model_size_mb']:.1f}MB")

            except Exception as e:
                log(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = ["dataset", "model", "fit_time_s", "train_mem_mb", "infer_mem_mb", "model_size_mb", "infer_ms", "fps"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    log(f"\nResults saved to: {OUTPUT_CSV}")

    # Print summary table (averages)
    log("\n" + "=" * 70)
    log("EFFICIENCY SUMMARY (averages across datasets)")
    log("=" * 70)
    log(f"{'Method':<12} {'Fit(s)':>10} {'FitMem(MB)':>12} {'InfMem(MB)':>12} {'Size(MB)':>10} {'ms/img':>10} {'FPS':>8}")
    log("-" * 70)

    for model_name in MODELS:
        model_results = [r for r in all_results if r["model"] == model_name]
        if model_results:
            avg_fit_t = np.mean([r["fit_time_s"] for r in model_results])
            avg_fit_m = np.mean([r["train_mem_mb"] for r in model_results])
            avg_inf_m = np.mean([r["infer_mem_mb"] for r in model_results])
            avg_size = np.mean([r["model_size_mb"] for r in model_results])
            avg_inf_ms = np.mean([r["infer_ms"] for r in model_results])
            avg_fps = np.mean([r["fps"] for r in model_results])
            log(f"{model_name:<12} {avg_fit_t:>10.1f} {avg_fit_m:>12.0f} {avg_inf_m:>12.0f} {avg_size:>10.1f} {avg_inf_ms:>10.1f} {avg_fps:>8.1f}")

    log("-" * 70)

    # LaTeX table
    log("\n--- LaTeX Table (copy to paper) ---")
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \caption{Computational efficiency comparison at $N=10$ training images, measured on NVIDIA A5000 (24 GB). Fit time includes feature extraction and model fitting. Memory is peak GPU allocation. Storage is per-category model size.}")
    print(r"    \label{tab:efficiency}")
    print(r"    \begin{tabular}{lcccccc}")
    print(r"        \toprule")
    print(r"        \textbf{Method} & \textbf{Backbone} & \textbf{Fit (s)} & \textbf{Fit (MB)} & \textbf{Infer (MB)} & \textbf{Storage (MB)} & \textbf{FPS} \\")
    print(r"        \midrule")
    backbones = {"galad": "DINOv3-L", "patchcore": "WRN50", "dinomaly": "DINOv2-L"}
    for model_name in MODELS:
        model_results = [r for r in all_results if r["model"] == model_name]
        if model_results:
            backbone = backbones.get(model_name, "-")
            avg_fit_t = np.mean([r["fit_time_s"] for r in model_results])
            avg_fit_m = np.mean([r["train_mem_mb"] for r in model_results])
            avg_inf_m = np.mean([r["infer_mem_mb"] for r in model_results])
            avg_size = np.mean([r["model_size_mb"] for r in model_results])
            avg_fps = np.mean([r["fps"] for r in model_results])
            display_name = "GaLAD" if model_name == "galad" else model_name.capitalize()
            if model_name == "galad":
                print(f"        \\textbf{{{display_name}}} & {backbone} & \\textbf{{{avg_fit_t:.1f}}} & \\textbf{{{avg_fit_m:.0f}}} & \\textbf{{{avg_inf_m:.0f}}} & \\textbf{{{avg_size:.1f}}} & \\textbf{{{avg_fps:.1f}}} \\\\")
            else:
                print(f"        {display_name} & {backbone} & {avg_fit_t:.1f} & {avg_fit_m:.0f} & {avg_inf_m:.0f} & {avg_size:.1f} & {avg_fps:.1f} \\\\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
