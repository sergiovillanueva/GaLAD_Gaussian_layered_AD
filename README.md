# GaLAD: A New Framework for Few-Shot Industrial Anomaly Detection using Foundation Models and Gradient-Free Density Estimation

Official implementation and reproducibility package for the paper *"A New Framework for Few-Shot Industrial Anomaly Detection using Foundation Models and Gradient-Free Density Estimation"*. GaLAD is the name of the method.

## Overview

GaLAD is a gradient-free few-shot anomaly detector. It models the distribution of normal patch embeddings with Gaussian Mixture Models (GMMs) fitted on intermediate-layer features from a frozen self-supervised Vision Transformer (DINOv3 ViT-L/16). The fitting stage is closed-form (PCA) plus EM (GMM), runs on CPU in seconds once features have been extracted, and the per-category state is around 6.5 MB regardless of the training-set size.

Headline results in the few-shot regime ($N=5$ normal images per category):

- Best or tied-best image-level AUPR against five established baselines (PaDiM, PatchCore, Reverse Distillation, Dinomaly, EfficientAD) on all six benchmarks.
- Best pixel-level AUsPRO on five of six benchmarks, including a 25% relative improvement on logical anomalies (MVTec LOCO).
- Same-backbone analysis on the full 46 categories: statistically significant gain over a single Gaussian on identical DINOv3 features (Wilcoxon $p=1.8\times10^{-5}$), competitive on average with dense nearest-neighbor scoring at a fraction of the per-category storage.
- Cross-backbone scoring control on four ViT-L encoders (DINOv3, DINOv2, CLIP, SigLIP) with a uniform protocol: the GMM is the best scoring layer on every backbone, beats a single Gaussian on all four ($p\le1.8\times10^{-3}$) and dense nearest-neighbor scoring on three of four, so the contribution is the scoring stage rather than a single backbone.
- 1.9x smaller per-category state than PatchCore and often more stable across seeds than the evaluated gradient-based baselines (up to several-fold lower variance in the larger-N regime, though the gap narrows at the smallest sample sizes).

## Requirements

- Python 3.11
- CUDA-capable GPU (tested on NVIDIA RTX 4090 and A5000)

### Installation

```bash
# Using uv (recommended)
pip install uv
uv sync

# Or using pip (CPU)
pip install torch torchvision transformers scikit-learn scipy scikit-image matplotlib tqdm pandas

# For CUDA (adjust cu128 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Baseline experiments use [Anomalib](https://github.com/openvinotoolkit/anomalib):

```bash
uv sync --extra baselines
# or
pip install anomalib
```

EfficientAD additionally needs a local copy of Imagenette (used as the penalty dataset during student-teacher training) and the official pretrained teacher checkpoint. Anomalib downloads both the first time it runs EfficientAD; pre-download them if the training machine does not have internet access:

```
datasets/imagenette2/{train,val}/<class>/...
pre_trained/efficientad_pretrained_weights/{pretrained_teacher_small.pth,pretrained_teacher_medium.pth}
```

## Data Preparation

Download the datasets and place them under `data/` following the standard MVTec-style structure:

```
data/
├── mvtec_AD/          # MVTec AD - https://www.mvtec.com/company/research/datasets/mvtec-ad
│   ├── bottle/
│   │   ├── train/good/
│   │   └── test/{good,broken_large,...}/
│   └── ...
├── mvtec_loco_AD/     # MVTec LOCO AD - https://www.mvtec.com/company/research/datasets/mvtec-loco
├── VisA/              # VisA - https://github.com/amazon-science/spot-diff (uses split_csv/1cls.csv)
├── AutoVI/            # AutoVI - https://autovi.utc.fr/
├── btad/              # BTAD - https://avires.dimi.uniud.it/papers/btad/
└── GoodsAD/           # PKU-GoodsAD - https://github.com/jianzhang96/GoodsAD
```

Each category follows the standard layout: `train/good/` for normal training images, `test/{good,defect_type}/` for test images, and `ground_truth/{defect_type}/` for pixel-level masks. VisA is handled through its official `split_csv/1cls.csv`.

If your datasets are stored elsewhere, pass `--data-root <path>` or set the `GALAD_DATA_ROOT` environment variable.

## Usage

All commands assume you run them from the repository root (this folder).

### Core Model

`galad_model.py` contains the self-contained GaLAD implementation:

```python
from galad_model import GaLAD, load_dataset

# Initialize model (paper configuration)
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

The scripts are resumable: re-running them skips experiments already recorded in the corresponding CSV.

#### 1. Train GaLAD on all benchmarks

```bash
uv run train_galad.py --device 0
```

Runs GaLAD across all 46 categories, 5 seeds, and training sizes {2, 5, 10, 20, 50}. Results are appended to `results/galad_results.csv`.

#### 2. Train external baselines (Anomalib)

```bash
uv run train_baselines.py --device 0
```

Trains PaDiM, PatchCore, Reverse Distillation, Dinomaly, and EfficientAD via Anomalib. Results are appended to `results/baseline_results.csv`.

#### 3. Train same-backbone controls

```bash
uv run dinov3_baselines.py
```

Trains DINOv3-kNN (classical PatchCore-style: layers concatenated before PCA, max pooling, k=9) and DINOv3-K1 (single Gaussian) on the same DINOv3 features and PCA dimensionality budget. This is the strong external same-backbone baseline of Table 5. Results go to `results/dinov3_baselines_all.csv`. The cross-backbone scoring control (`crossbackbone_control.py`) is the complementary internal ablation: it keeps the exact GaLAD pipeline (per-layer PCA, percentile-95 pooling) and changes only the final scoring layer, with a uniform k=5 across encoders.

#### 4. Ablation study

```bash
uv run ablation_study.py
```

Evaluates layer selection, GMM components, PCA skip, and smoothing sigma on a representative subset.

#### 4b. Backbone-generality ablation

```bash
uv run backbone_ablation.py --full     # full run (4 backbones x 46 categories x 5 seeds)
uv run backbone_ablation.py --test     # smoke test (1 category)
uv run analyze_backbone_ablation.py    # summary table
uv run crossbackbone_control.py        # K1 / kNN / PCA-residual controls on the 4 encoders
uv run analyze_crossbackbone_control.py
```

Runs the full GaLAD pipeline on top of four frozen ViT-L encoders (DINOv3,
DINOv2, CLIP, SigLIP) at their native resolution, to show that the framework
is not tied to DINOv3. Results go to `results/backbone_ablation.csv`. Resumable
and robust (per-experiment error logging). The four backbones are downloaded
from Hugging Face on first use.

#### 5. Computational efficiency

```bash
uv run measure_efficiency.py
```

Compares fit time, inference speed, GPU memory, and per-category state size across methods.

#### 6. Statistical analysis and summary tables

```bash
uv run analyze_results.py     # Demsar protocol (Friedman + Nemenyi + Wilcoxon + Cliff's delta) + LaTeX-ready tables
uv run generate_cd_diagram.py     # Critical-difference (Nemenyi) diagram
uv run results_summary.py
uv run generate_tables.py
uv run generate_appendix_tables.py
```

`analyze_results.py` is the single entry point for the post-hoc statistical analysis: it loads all three CSVs (GaLAD, baselines, same-backbone controls), aggregates per-benchmark statistics with the benchmark-balanced protocol used in the paper, runs Friedman / Nemenyi / Wilcoxon (Demsar 2006), reports Cliff's delta effect sizes, and writes `results/results_overview.md` together with LaTeX-ready CSV tables in `results/tables_latex/`. The numbers it produces are the ones reported in the manuscript.

## Pre-computed Results

The `results/` directory contains everything needed to reproduce the tables and statistical analyses in the paper:

| File / folder | Description |
|---------------|-------------|
| `galad_results.csv` | Raw GaLAD results (all seeds, all train sizes) |
| `baseline_results.csv` | Raw baseline results (PaDiM, PatchCore, RD, Dinomaly, EfficientAD) |
| `dinov3_baselines_all.csv` | Same-backbone controls on the full 46 categories (DINOv3-kNN, DINOv3-K1) |
| `results_per_benchmark.csv` | Aggregated results per benchmark |
| `efficiency_comparison.csv` | Computational efficiency measurements |
| `appendix/A1-A9_*.csv` | Per-category tables for the paper appendix |
| `tables/T2_*.csv`, `tables/T3_*.csv`, `tables/significance_matrix.csv` | Main-text tables |
| `tables_latex/*.csv` | LaTeX-ready tables including EfficientAD and same-backbone controls |
| `results_overview.md` | Human-readable summary of all post-hoc analyses |
| `statistical_analysis_results.md` | Friedman / Nemenyi / Wilcoxon results writeup |

CSV delimiter conventions:

- Raw results (`galad_results.csv`, `baseline_results.csv`, `dinov3_baselines_all.csv`) use `;`.
- Derived paper tables in `results/tables/` and `results/tables_latex/` use `;`.
- Appendix per-category tables in `results/appendix/` use `,`.

## Method Summary

The GaLAD pipeline:

1. Extract patch embeddings from intermediate DINOv3 layers (layers $-6$ and $-12$).
2. Reduce dimensionality with PCA, skipping the first two principal components (which capture global illumination and background).
3. Fit a 3-component Gaussian mixture per layer on the reduced patch features. The number of components is automatically capped if too few training patches are available: $K = \min(3, \max(2, \lfloor n_{\text{patches}}/(d+10) \rfloor))$.
4. Score test patches via negative log-likelihood and reshape into a $G\times G$ anomaly map.
5. Smooth the per-layer maps with a Gaussian filter ($\sigma=1$), average them, and aggregate with 95th-percentile pooling for the image-level score.

The same configuration is applied to all 46 categories without per-category tuning.

## Statistical Analysis

The paper follows the protocol of Demsar (2006) for comparing multiple classifiers across multiple data sets:

- Friedman omnibus test on per-category mean AUPR (5 seeds, three training sizes averaged).
- Nemenyi post-hoc with critical-difference diagrams.
- Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction for targeted one-vs-all comparisons.
- Cliff's delta as a non-parametric effect-size measure.

Three analyses are reported: four established baselines, the same set extended with EfficientAD, and the same-backbone setting (GaLAD vs DINOv3-kNN vs DINOv3-K1).

Per-benchmark significance is reported separately with Welch's $t$-test as an exploratory local check, with the global non-parametric analysis as the basis for the conclusions.

## Project Structure

```
.
├── galad_model.py                # Core GaLAD implementation
├── train_galad.py                # Full benchmark evaluation
├── train_baselines.py            # Baseline experiments via Anomalib
├── dinov3_baselines.py           # Same-backbone controls (DINOv3-kNN, DINOv3-K1)
├── ablation_study.py             # Ablation experiments (Tables T4-T5)
├── backbone_ablation.py          # Backbone-generality ablation (DINOv3/DINOv2/CLIP/SigLIP)
├── analyze_backbone_ablation.py  # Backbone-ablation summary table
├── measure_efficiency.py         # Computational efficiency comparison
├── analyze_results.py        # Full post-hoc analysis (Demsar + tables)
├── generate_cd_diagram.py        # Critical-difference (Nemenyi) diagram
├── results_summary.py            # Aggregate results across benchmarks
├── generate_tables.py            # Generate paper tables (T2-T3)
├── generate_appendix_tables.py   # Generate appendix tables (A1-A9)
├── verify_results_consistency.py # Sanity-check raw vs aggregated results
├── pyproject.toml                # Dependencies
└── results/                      # Pre-computed experimental results
```

## Citation

This work is currently under peer review. Citation details will be added here once the paper is published.

## License

This project is licensed under the MIT License (see [LICENSE](LICENSE)). The DINOv3 ViT-L/16 checkpoint used as backbone is distributed by Meta AI under the DINOv3 release terms, which are more restrictive than a permissive open-source license; any deployment that uses this checkpoint must comply with them.
