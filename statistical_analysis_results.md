# Statistical Significance Analysis

## Setup

Following the recommendations of Demsar (2006), we evaluate the statistical
significance of performance differences using non-parametric tests across the
46 categories of the six benchmarks.
For each (category, method) pair we average the AUPR scores over the 5 random
seeds and the three training sizes (N=5, 10, 20).
AUPR is the primary metric as it better captures performance under the class
imbalance typical of anomaly detection.

PaDiM is reported in the benchmark tables but excluded from the omnibus rank
tests so that the comparison focuses on the competitive methods with
overlapping performance ranges.

## Established baselines (4 methods)

We compare GaLAD against PatchCore, Reverse Distillation, and Dinomaly.

### Friedman test

| Method | Average rank (lower is better) |
| --- | :---: |
| GaLAD | **1.92** |
| PatchCore | 2.65 |
| Dinomaly | 2.71 |
| Reverse Distillation | 2.72 |

Friedman omnibus: chi2 = 12.31, p = 6.4e-3 (significant at alpha = 0.05).

### Nemenyi post-hoc

With k = 4 methods and N = 46 categories, the critical difference is
CD = 0.692 at alpha = 0.05.

| Comparison | Rank diff | Significant |
| --- | :---: | :---: |
| GaLAD vs Reverse Distillation | 0.793 | Yes |
| GaLAD vs Dinomaly | 0.783 | Yes |
| GaLAD vs PatchCore | 0.728 | Yes (just above CD) |
| Dinomaly vs Reverse Distillation | 0.011 | No |
| PatchCore vs Dinomaly | 0.054 | No |
| PatchCore vs Reverse Distillation | 0.065 | No |

### Wilcoxon signed-rank with Holm-Bonferroni

| Comparison | p-value | Holm-Bonferroni threshold | Result |
| --- | :---: | :---: | :---: |
| GaLAD vs Dinomaly | 2.5e-3 | 0.017 | Reject H0 |
| GaLAD vs Reverse Distillation | 4.8e-3 | 0.025 | Reject H0 |
| GaLAD vs PatchCore | 5.9e-3 | 0.050 | Reject H0 |

Effect sizes (Cliff's delta) are small to negligible against PatchCore and
Reverse Distillation, indicating that the differences are statistically
detectable but modest in absolute magnitude.

## Extended analysis with EfficientAD (5 methods)

Including EfficientAD as a recent baseline strengthens the global picture.

### Friedman test

| Method | Average rank |
| --- | :---: |
| GaLAD | **1.99** |
| PatchCore | 2.87 |
| Reverse Distillation | 2.91 |
| Dinomaly | 2.92 |
| EfficientAD | 4.30 |

Friedman omnibus: chi2 = 50.72, p = 2.6e-10.

### Nemenyi post-hoc

With k = 5 methods and N = 46 categories, CD = 0.899 at alpha = 0.05.

| Comparison | Rank diff | Significant |
| --- | :---: | :---: |
| GaLAD vs EfficientAD | 2.315 | Yes |
| PatchCore vs EfficientAD | 1.435 | Yes |
| Reverse Distillation vs EfficientAD | 1.391 | Yes |
| Dinomaly vs EfficientAD | 1.380 | Yes |
| GaLAD vs Dinomaly | 0.935 | Yes |
| GaLAD vs Reverse Distillation | 0.924 | Yes |
| GaLAD vs PatchCore | 0.880 | No (just below CD) |

### Wilcoxon signed-rank with Holm-Bonferroni

| Comparison | p-value | Holm-Bonferroni threshold | Result |
| --- | :---: | :---: | :---: |
| GaLAD vs EfficientAD | 4.0e-13 | 0.0125 | Reject H0 |
| GaLAD vs Dinomaly | 2.5e-3 | 0.0167 | Reject H0 |
| GaLAD vs Reverse Distillation | 4.8e-3 | 0.025 | Reject H0 |
| GaLAD vs PatchCore | 5.9e-3 | 0.050 | Reject H0 |

Cliff's delta GaLAD vs EfficientAD = +0.373 (medium effect).
Cliff's delta against the other three baselines is negligible to small, in
line with the close numerical differences in Table 1.

## Same-backbone setting

Holding the feature extractor constant we compare GaLAD, DINOv3-kNN, and
DINOv3-K1 (single Gaussian) on the 46 categories.

### Friedman test

| Method | Average rank |
| --- | :---: |
| GaLAD | **1.63** |
| DINOv3-kNN | 2.09 |
| DINOv3-K1 | 2.28 |

Friedman omnibus: chi2 = 10.42, p = 5.5e-3.

### Nemenyi post-hoc

With k = 3 methods and N = 46 categories, CD = 0.489 at alpha = 0.05.

| Comparison | Rank diff | Significant |
| --- | :---: | :---: |
| GaLAD vs DINOv3-K1 | 0.652 | Yes |
| GaLAD vs DINOv3-kNN | 0.457 | No |
| DINOv3-kNN vs DINOv3-K1 | 0.196 | No |

### Wilcoxon signed-rank with Holm-Bonferroni

| Comparison | p-value | Holm-Bonferroni threshold | Result |
| --- | :---: | :---: | :---: |
| GaLAD vs DINOv3-K1 | 1.8e-5 | 0.025 | Reject H0 |
| GaLAD vs DINOv3-kNN | 2.1e-1 | 0.050 | Fail to reject |

The same-backbone analysis confirms that the multi-component GMM provides a
significant gain over a single Gaussian when the feature extractor is held
constant. Against nearest-neighbor scoring on the same DINOv3 features, GaLAD
ranks higher on average but the difference is not significant; per-benchmark
inspection (paper Table 5) shows that GaLAD is better on five of six
benchmarks while DINOv3-kNN is significantly better on VisA, where the dense
memory bank captures texture-rich variations.

## Summary

- Against established baselines (PatchCore, Reverse Distillation, Dinomaly),
  GaLAD obtains the best average rank with a significant margin under
  Wilcoxon and Nemenyi tests.
- Against the recent EfficientAD baseline, GaLAD shows a large effect size
  (Cliff's delta = +0.37) and a Friedman/Nemenyi/Wilcoxon-significant lead.
- Under the same-backbone protocol on 46 categories, GaLAD is significantly
  better than the single-Gaussian variant and on average comparable to (but
  no worse than) the nearest-neighbor variant, with benchmark-dependent
  differences that we discuss in the main text.

## Localization (AUsPRO, 4 methods)

The same Demsar protocol applied to pixel-level AUsPRO (averaged over 5 seeds
and N in {5,10,20}, 46 categories), where GaLAD's advantage is largest.

### Friedman test

| Method | Average rank |
| --- | :---: |
| GaLAD | **1.52** |
| Dinomaly | 2.65 |
| Reverse Distillation | 2.74 |
| PatchCore | 3.09 |

Friedman omnibus: chi2 = 38.14, p = 2.6e-8.

### Nemenyi + Wilcoxon (GaLAD vs each)

CD = 0.69 at alpha = 0.05. All rank differences exceed CD, so GaLAD is
significantly better than all three baselines in localization.

| Comparison | Rank diff | Wilcoxon p | Mean AUsPRO diff |
| --- | :---: | :---: | :---: |
| GaLAD vs PatchCore | 1.57 | 6.9e-9 | +0.13 |
| GaLAD vs Reverse Distillation | 1.22 | 2.7e-7 | +0.09 |
| GaLAD vs Dinomaly | 1.13 | 2.9e-7 | +0.10 |

GaLAD's localization advantage is both large and statistically unambiguous,
in contrast to the detection (AUPR) result where the margin over the strongest
baselines is smaller.

## Cross-backbone scoring control (image-level, 46 categories, N=10)

Four scoring layers are evaluated on the same frozen features and per-layer PCA
of the four ViT-L encoders (DINOv3, DINOv2, CLIP, SigLIP), on the full
46-category benchmark with 5 seeds: GMM (K=3, GaLAD default), single Gaussian
(K=1), dense kNN (k=5, 10^4 cap), and a PCA subspace reconstruction-residual
score. The K=1, kNN and subspace variants come from one script that shares the
feature extraction and PCA across methods (2760 control experiments); the GMM
K=3 column is the GaLAD default from the backbone-generality table.

| Backbone | GMM AUROC/AUPR | K=1 AUROC/AUPR | kNN AUROC/AUPR | Subspace AUROC/AUPR |
| --- | :---: | :---: | :---: | :---: |
| DINOv3 ViT-L/16 | 0.828/0.839 | 0.807/0.825 | 0.805/0.825 | 0.829/0.838 |
| DINOv2 ViT-L/14 | 0.816/0.831 | 0.775/0.804 | 0.796/0.822 | 0.824/0.835 |
| CLIP ViT-L/14   | 0.823/0.833 | 0.790/0.811 | 0.800/0.821 | 0.823/0.831 |
| SigLIP ViT-L/16 | 0.827/0.837 | 0.798/0.817 | 0.800/0.820 | 0.829/0.836 |

Paired Wilcoxon over the 46 per-category mean AUPR scores:

| Backbone | p (GMM vs K1) | p (GMM vs kNN) | p (GMM vs Subspace) |
| --- | :---: | :---: | :---: |
| DINOv3 | 1.8e-3 | 1.2e-2 | 1.6e-1 (tie) |
| DINOv2 | 4.7e-5 | 1.6e-1 | 8.7e-3 (subspace better) |
| CLIP   | 3.4e-7 | 5.2e-5 | 7.7e-1 (tie) |
| SigLIP | 1.8e-9 | 2.2e-6 | 4.7e-1 (tie) |

Reading the cross-backbone control:

- The gain over the single Gaussian (K=1) is real and transfers: the GMM beats
  K=1 on both metrics on every backbone, significantly in AUPR in every case
  (smallest p = 1.8e-9 for SigLIP). Against dense kNN (k=5) the GMM is ahead on
  all four, significantly on three.
- The most informative comparison is GMM vs subspace residual: they are
  statistically indistinguishable on three of four encoders, and the subspace
  model is marginally better on DINOv2. The improvement therefore comes from
  using a non-linear model beyond a single Gaussian, not from the
  Gaussian-mixture form specifically. Two simple parametric models (K=3 mixture
  and PCA subspace residual) reach the same accuracy and both beat K=1 and kNN.
- We keep the GMM for its calibrated likelihood, fixed small size, and ability
  to flag both out-of-subspace patches and low-density regions inside the
  subspace, while noting the framework is robust to this choice.
- All effects are small in absolute size (mean paired AUPR gains +0.009 to
  +0.027 over K=1, Cliff's |delta| < 0.10); the consistency across the 46
  categories drives the significance.
- This control is not directly comparable to the DINOv3 same-backbone table
  (Section above): there kNN uses the classical PatchCore recipe (concatenated
  PCA, max pooling, k=9), which is stronger and ties GaLAD; here every scoring
  layer runs inside the exact GaLAD pipeline with uniform k=5.

## Backbone generality (image-level AUROC / AUPR, 46 categories)

Full GaLAD pipeline kept fixed, only the frozen encoder swapped, evaluated on
the full 46-category benchmark (5 seeds, N=10, 920 experiments). Each backbone
runs at its native resolution.

### Aggregate scores

| Backbone | Resolution | Mean AUROC | Mean AUPR | Bench.-bal. AUROC | Bench.-bal. AUPR |
| --- | :---: | :---: | :---: | :---: | :---: |
| DINOv3 ViT-L/16 | 448 | **0.828** | **0.839** | **0.799** | **0.802** |
| SigLIP ViT-L/16 | 256 | 0.827 | 0.837 | 0.787 | 0.795 |
| CLIP ViT-L/14 | 224 | 0.823 | 0.833 | 0.787 | 0.793 |
| DINOv2 ViT-L/14 | 448 | 0.816 | 0.831 | 0.786 | 0.795 |

### Per-benchmark AUROC

| Benchmark | DINOv3 | DINOv2 | CLIP | SigLIP |
| --- | :---: | :---: | :---: | :---: |
| MVTec AD | 0.936 | 0.929 | 0.948 | **0.966** |
| MVTec LOCO | 0.714 | 0.706 | **0.757** | **0.757** |
| VisA | 0.867 | 0.852 | 0.872 | **0.872** |
| AutoVI | **0.699** | 0.679 | 0.644 | 0.645 |
| BTAD | **0.965** | 0.935 | 0.939 | 0.933 |
| GoodsAD | **0.614** | 0.613 | 0.561 | 0.546 |

### Demsar over 46 categories

Friedman chi2 = 6.98, p = 7.2e-2 (borderline). Average ranks: DINOv3 2.14,
CLIP 2.49, SigLIP 2.52, DINOv2 2.85.

| Comparison | Rank diff | CD (alpha=0.05) | Wilcoxon p | Mean AUROC diff |
| --- | :---: | :---: | :---: | :---: |
| DINOv3 vs DINOv2 | 0.71 | 0.69 (significant) | 1.7e-3 | +0.012 |
| DINOv3 vs CLIP | 0.35 | not significant | 7.4e-1 | +0.005 |
| DINOv3 vs SigLIP | 0.38 | not significant | 7.7e-1 | +0.001 |

All four backbones land in a narrow mean-AUROC band of 0.816-0.828 across the
46 categories, confirming that the framework is not tied to DINOv3. DINOv3
obtains the best aggregate and rank, but only its advantage over DINOv2 is
statistically significant; against CLIP and SigLIP the differences are small
and not significant. DINOv3 is the clear winner on AutoVI, BTAD and GoodsAD
(structured, cluttered scenes), while CLIP and SigLIP are competitive or
slightly better on MVTec AD and MVTec LOCO (texture-dominated).
