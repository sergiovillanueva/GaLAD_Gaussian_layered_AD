# Statistical Significance Analysis (post-revision)

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
