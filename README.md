# GaLAD: Gaussian Layered Anomaly Detector for Few-Shot Industrial Inspection

Official implementation of the paper *"GaLAD: Gaussian Layered Anomaly Detector for Few-Shot Industrial Inspection"*, submitted to Expert Systems with Applications.

## Overview

GaLAD is a gradient-free few-shot anomaly detector that models the distribution of normal patch embeddings using Gaussian Mixture Models (GMMs) fitted on intermediate-layer features from a self-supervised Vision Transformer (DINOv3 ViT-L/16).

Key results with only **5 training images** per category:
- Best or tied-best image-level AUPR on **6/6 benchmarks**
- Best pixel-level AUsPRO on **5/6 benchmarks**
- 1.9x smaller model storage than PatchCore
- 3-9x lower variance than gradient-based baselines

## Requirements

- Python 3.11
- CUDA-capable GPU (tested on NVIDIA RTX 4090)

### Installation

```bash
# Using uv (recommended)
pip install uv
uv sync

# Or using pip (CPU)
pip install torch torchvision transformers scikit-learn scipy scikit-image matplotlib tqdm

# For CUDA (adjust cu128 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Note:** Baseline experiments require [Anomalib](https://github.com/openvinotoolkit/anomalib):
```bash
uv sync --extra baselines

# Or with pip
pip install anomalib
```

## Data Preparation

Download the datasets and place them under `data/` following the standard MVTec-style structure:

```
data/
├── mvtec_AD/          # MVTec AD (https://www.mvtec.com/company/research/datasets/mvtec-ad)
│   ├── bottle/
│   │   ├── train/good/
│   │   └── test/{good,broken_large,...}/
│   ├── cable/
│   └── ...
├── mvtec_loco_AD/     # MVTec LOCO AD (https://www.mvtec.com/company/research/datasets/mvtec-loco)
├── VisA/              # VisA (https://github.com/amazon-science/spot-diff)
├── AutoVI/            # AutoVI
├── btad/              # BTAD (https://avires.dimi.uniud.it/papers/btad/)
└── GoodsAD/           # GoodsAD
```

Each category should follow the standard layout: `train/good/` for normal training images, `test/{good,defect_type}/` for test images, and `ground_truth/{defect_type}/` for pixel-level masks.

If your datasets are stored elsewhere, pass `--data-root <path>` (or set `GALAD_DATA_ROOT`).

## Usage

All commands below assume you run them from the repository root (this folder).

### Core Model

`galad_model.py` contains the self-contained GaLAD implementation:

```python
from galad_model import GaLAD, load_dataset

# Initialize model (default: GaLAD-A paper configuration)
model = GaLAD(
    resolution=448,
    layers=[-6, -12],
    pca_dim=256,
    pca_skip=2,
    n_components=3,
    smooth_sigma=1.0,
)

# Load dataset and fit
train_paths, test_info = load_dataset("data/mvtec_AD/bottle")
model.fit(train_paths[:5])  # few-shot: only 5 images

# Score test images
test_paths = [t['path'] for t in test_info]
scores, anomaly_maps = model.score_batch(test_paths)
```

### Reproduce Paper Results

#### 1. Train GaLAD on all benchmarks

```bash
uv run train_galad.py --device 0
```

Runs GaLAD across all 46 categories, 5 seeds, and training sizes {2, 5, 10, 20, 50}. Results are saved incrementally to `results/galad_results.csv`.

#### 2. Train baselines

```bash
uv run train_baselines.py --device 0
```

Trains PaDiM, PatchCore, Reverse Distillation, and Dinomaly baselines using Anomalib. Results are saved to `results/baseline_results.csv`.

#### 3. Run ablation study

```bash
uv run ablation_study.py
```

Evaluates layer selection, GMM components, PCA skip, and smoothing sigma on MVTec AD + VisA + GoodsAD (33 categories, 5 seeds).

#### 4. Measure computational efficiency

```bash
uv run measure_efficiency.py
```

Compares training time, inference speed, GPU memory, and model size across methods.

#### 5. Generate summary tables

```bash
uv run results_summary.py
uv run generate_tables.py
uv run generate_appendix_tables.py
```

## Pre-computed Results

The `results/` directory contains all experimental results used in the paper:

| File | Description |
|------|-------------|
| `galad_results.csv` | Raw GaLAD results (all seeds, all train sizes) |
| `baseline_results.csv` | Raw baseline results (PaDiM, PatchCore, RD, Dinomaly) |
| `results_per_benchmark.csv` | Aggregated results per benchmark (mean +/- std) |
| `efficiency_comparison.csv` | Computational efficiency measurements |
| `appendix/A1-A9_*.csv` | Per-category detailed results for the paper appendix |

Delimiter conventions:
- Raw results in `results/{galad_results,baseline_results}.csv` use `;`.
- Derived paper tables in `results/tables/` use `;`.
- Appendix per-category tables in `results/appendix/` use `,`.

## Method

GaLAD pipeline:
1. Extract patch features from intermediate DINOv3 layers (layers -6 and -12)
2. Reduce dimensionality with PCA, skipping the first 2 components
3. Fit a 3-component GMM per layer on the reduced features
4. Score test patches via negative log-likelihood
5. Aggregate layer scores and smooth anomaly maps

## Project Structure

```
.
├── galad_model.py              # Core GaLAD implementation
├── train_galad.py              # Full benchmark evaluation
├── train_baselines.py          # Baseline experiments (Anomalib)
├── ablation_study.py           # Ablation experiments (Tables T4-T5)
├── measure_efficiency.py       # Computational efficiency comparison
├── results_summary.py          # Aggregate results across benchmarks
├── generate_tables.py          # Generate paper tables (T2-T3)
├── generate_appendix_tables.py # Generate appendix tables (A1-A9)
├── pyproject.toml              # Dependencies
└── results/                    # Pre-computed experimental results
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{villanueva2026galad,
  title={GaLAD: Gaussian Layered Anomaly Detector for Few-Shot Industrial Inspection},
  author={Villanueva, Sergio and Soria-Olivas, Emilio and S{\'a}nchez-Monta{\~n}{\'e}s, Manuel},
  journal={Expert Systems with Applications},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
