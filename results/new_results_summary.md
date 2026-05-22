# New Experimental Results - Paper Revision

This document summarizes the additional experiments run in response to 
reviewer comments (EfficientAD baseline and DINOv3 same-backbone controls 
extended to all 46 categories) together with re-computed global statistics.

## Data overview

Number of (dataset, seed) measurements per (model, N):

```
train_limit            2    5    10   20   50
model                                        
dinomaly                0  230  230  230  230
dinov3_k1               0  230  230  230    0
dinov3_knn              0  230  230  230    0
efficient_ad            0  230  230  230    0
galad                 230  230  230  230  230
padim                   1  230  230  230  230
patchcore             230  230  230  230  230
reverse_distillation  230  230  230  230  230
```

## 1. Global averages across 46 categories (image-level AUPR)

### Image AUPR

| Model | N=2 | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|
| dinomaly | - | 0.768±0.034 | 0.760±0.075 | 0.767±0.052 | 0.775±0.047 |
| dinov3_k1 | - | 0.772±0.023 | 0.777±0.019 | 0.778±0.012 | - |
| dinov3_knn | - | 0.756±0.038 | 0.786±0.026 | 0.798±0.026 | - |
| efficient_ad | - | 0.660±0.041 | 0.681±0.035 | 0.697±0.065 | - |
| galad | 0.754±0.058 | 0.794±0.041 | 0.804±0.032 | 0.811±0.023 | 0.811±0.015 |
| padim | 0.549±nan | 0.629±0.044 | 0.635±0.045 | 0.639±0.051 | 0.632±0.058 |
| patchcore | 0.744±0.028 | 0.754±0.029 | 0.763±0.016 | 0.773±0.024 | 0.792±0.035 |
| reverse_distillation | 0.735±0.027 | 0.758±0.051 | 0.773±0.023 | 0.785±0.036 | 0.778±0.029 |

### Image AUROC

| Model | N=2 | N=5 | N=10 | N=20 | N=50 |
|---|---|---|---|---|---|
| dinomaly | - | 0.747±0.042 | 0.738±0.078 | 0.743±0.068 | 0.750±0.066 |
| dinov3_k1 | - | 0.762±0.022 | 0.768±0.020 | 0.771±0.014 | - |
| dinov3_knn | - | 0.766±0.028 | 0.798±0.021 | 0.810±0.019 | - |
| efficient_ad | - | 0.628±0.060 | 0.653±0.058 | 0.666±0.062 | - |
| galad | 0.730±0.074 | 0.787±0.040 | 0.801±0.036 | 0.812±0.017 | 0.814±0.013 |
| padim | 0.540±nan | 0.580±0.061 | 0.586±0.063 | 0.582±0.069 | 0.561±0.080 |
| patchcore | 0.728±0.030 | 0.744±0.026 | 0.756±0.020 | 0.767±0.020 | 0.779±0.018 |
| reverse_distillation | 0.705±0.042 | 0.733±0.084 | 0.753±0.027 | 0.763±0.021 | 0.756±0.020 |

## 2. Per-benchmark results at N=5 (image-level AUPR)

| Benchmark | galad | patchcore | reverse_distillation | dinomaly | padim | efficient_ad | dinov3_knn | dinov3_k1 |
|---|---|---|---|---|---|---|---|---|
| AutoVI | 0.490±0.082 | 0.469±0.021 | 0.484±0.043 | 0.447±0.023 | 0.384±0.062 | 0.384±0.052 | 0.442±0.026 | 0.451±0.043 |
| BTAD | 0.993±0.002 | 0.927±0.050 | 0.961±0.103 | 0.980±0.004 | 0.693±0.045 | 0.687±0.042 | 0.901±0.079 | 0.988±0.007 |
| GoodsAD | 0.648±0.029 | 0.580±0.014 | 0.596±0.020 | 0.592±0.018 | 0.557±0.023 | 0.596±0.013 | 0.570±0.014 | 0.591±0.014 |
| MVTec AD | 0.967±0.013 | 0.940±0.025 | 0.941±0.025 | 0.965±0.059 | 0.858±0.024 | 0.883±0.032 | 0.950±0.020 | 0.963±0.010 |
| MVTec LOCO | 0.789±0.037 | 0.744±0.025 | 0.735±0.019 | 0.751±0.043 | 0.626±0.043 | 0.692±0.023 | 0.737±0.031 | 0.786±0.024 |
| VisA | 0.874±0.029 | 0.865±0.023 | 0.831±0.038 | 0.871±0.030 | 0.658±0.051 | 0.720±0.061 | 0.938±0.012 | 0.856±0.019 |

### Per-benchmark at N=10

| Benchmark | galad | patchcore | reverse_distillation | dinomaly | padim | efficient_ad | dinov3_knn | dinov3_k1 |
|---|---|---|---|---|---|---|---|---|
| AutoVI | 0.515±0.050 | 0.483±0.018 | 0.485±0.035 | 0.447±0.039 | 0.391±0.071 | 0.419±0.033 | 0.463±0.022 | 0.458±0.038 |
| BTAD | 0.992±0.002 | 0.938±0.015 | 0.989±0.002 | 0.940±0.157 | 0.688±0.026 | 0.725±0.034 | 0.943±0.053 | 0.990±0.003 |
| GoodsAD | 0.655±0.023 | 0.587±0.013 | 0.592±0.019 | 0.583±0.013 | 0.561±0.024 | 0.601±0.016 | 0.580±0.014 | 0.589±0.008 |
| MVTec AD | 0.973±0.022 | 0.945±0.015 | 0.950±0.014 | 0.955±0.062 | 0.863±0.031 | 0.889±0.032 | 0.972±0.011 | 0.967±0.012 |
| MVTec LOCO | 0.810±0.045 | 0.755±0.018 | 0.755±0.026 | 0.756±0.040 | 0.630±0.044 | 0.705±0.029 | 0.809±0.020 | 0.795±0.017 |
| VisA | 0.878±0.025 | 0.873±0.018 | 0.868±0.026 | 0.879±0.043 | 0.674±0.056 | 0.746±0.057 | 0.951±0.010 | 0.862±0.018 |

### Per-benchmark at N=20

| Benchmark | galad | patchcore | reverse_distillation | dinomaly | padim | efficient_ad | dinov3_knn | dinov3_k1 |
|---|---|---|---|---|---|---|---|---|
| AutoVI | 0.516±0.050 | 0.513±0.048 | 0.505±0.081 | 0.451±0.034 | 0.399±0.085 | 0.407±0.074 | 0.493±0.054 | 0.456±0.020 |
| BTAD | 0.991±0.002 | 0.928±0.010 | 0.988±0.003 | 0.984±0.004 | 0.689±0.049 | 0.777±0.128 | 0.951±0.018 | 0.990±0.003 |
| GoodsAD | 0.664±0.014 | 0.595±0.012 | 0.607±0.017 | 0.617±0.019 | 0.569±0.023 | 0.606±0.009 | 0.588±0.012 | 0.591±0.009 |
| MVTec AD | 0.979±0.013 | 0.951±0.011 | 0.957±0.015 | 0.956±0.060 | 0.861±0.030 | 0.907±0.024 | 0.978±0.010 | 0.968±0.008 |
| MVTec LOCO | 0.828±0.010 | 0.768±0.022 | 0.771±0.017 | 0.739±0.040 | 0.632±0.044 | 0.712±0.028 | 0.827±0.020 | 0.798±0.014 |
| VisA | 0.887±0.018 | 0.885±0.015 | 0.881±0.019 | 0.856±0.097 | 0.683±0.054 | 0.775±0.044 | 0.951±0.010 | 0.866±0.013 |

## 3. Welch's t-test: GaLAD vs new baselines (per benchmark)

Comparison of GaLAD vs the new baselines added in the revision, 
with Welch's $t$-test ($p<0.05$ marked with *):

### GaLAD vs efficient_ad
| Benchmark | N | GaLAD mean | Other mean | t | p | Significant |
|---|---|---|---|---|---|---|
| AutoVI | 5 | 0.490 | 0.384 | 1.42 | 1.63e-01 | n.s. |
| AutoVI | 10 | 0.515 | 0.419 | 1.23 | 2.24e-01 | n.s. |
| AutoVI | 20 | 0.516 | 0.407 | 1.37 | 1.77e-01 | n.s. |
| BTAD | 5 | 0.993 | 0.687 | 3.19 | 6.57e-03 | ✓ |
| BTAD | 10 | 0.992 | 0.725 | 3.18 | 6.68e-03 | ✓ |
| BTAD | 20 | 0.991 | 0.777 | 3.00 | 9.48e-03 | ✓ |
| GoodsAD | 5 | 0.648 | 0.596 | 2.29 | 2.61e-02 | ✓ |
| GoodsAD | 10 | 0.655 | 0.601 | 2.36 | 2.16e-02 | ✓ |
| GoodsAD | 20 | 0.664 | 0.606 | 2.48 | 1.60e-02 | ✓ |
| MVTec AD | 5 | 0.967 | 0.883 | 5.03 | 2.12e-06 | ✓ |
| MVTec AD | 10 | 0.973 | 0.889 | 4.99 | 2.91e-06 | ✓ |
| MVTec AD | 20 | 0.979 | 0.907 | 4.55 | 1.76e-05 | ✓ |
| MVTec LOCO | 5 | 0.789 | 0.692 | 3.25 | 2.11e-03 | ✓ |
| MVTec LOCO | 10 | 0.810 | 0.705 | 3.39 | 1.52e-03 | ✓ |
| MVTec LOCO | 20 | 0.828 | 0.712 | 4.00 | 3.17e-04 | ✓ |
| VisA | 5 | 0.874 | 0.720 | 5.94 | 3.48e-08 | ✓ |
| VisA | 10 | 0.878 | 0.746 | 5.19 | 9.44e-07 | ✓ |
| VisA | 20 | 0.887 | 0.775 | 4.87 | 3.55e-06 | ✓ |

### GaLAD vs dinov3_knn
| Benchmark | N | GaLAD mean | Other mean | t | p | Significant |
|---|---|---|---|---|---|---|
| AutoVI | 5 | 0.490 | 0.442 | 0.62 | 5.36e-01 | n.s. |
| AutoVI | 10 | 0.515 | 0.463 | 0.67 | 5.07e-01 | n.s. |
| AutoVI | 20 | 0.516 | 0.493 | 0.29 | 7.77e-01 | n.s. |
| BTAD | 5 | 0.993 | 0.901 | 2.40 | 3.11e-02 | ✓ |
| BTAD | 10 | 0.992 | 0.943 | 2.21 | 4.43e-02 | ✓ |
| BTAD | 20 | 0.991 | 0.951 | 2.54 | 2.34e-02 | ✓ |
| GoodsAD | 5 | 0.648 | 0.570 | 3.76 | 4.95e-04 | ✓ |
| GoodsAD | 10 | 0.655 | 0.580 | 3.82 | 4.36e-04 | ✓ |
| GoodsAD | 20 | 0.664 | 0.588 | 3.77 | 5.18e-04 | ✓ |
| MVTec AD | 5 | 0.967 | 0.950 | 1.81 | 7.30e-02 | n.s. |
| MVTec AD | 10 | 0.973 | 0.972 | 0.04 | 9.67e-01 | n.s. |
| MVTec AD | 20 | 0.979 | 0.978 | 0.26 | 7.97e-01 | n.s. |
| MVTec LOCO | 5 | 0.789 | 0.737 | 2.10 | 4.33e-02 | ✓ |
| MVTec LOCO | 10 | 0.810 | 0.809 | 0.04 | 9.69e-01 | n.s. |
| MVTec LOCO | 20 | 0.828 | 0.827 | 0.01 | 9.96e-01 | n.s. |
| VisA | 5 | 0.874 | 0.938 | -3.86 | 2.33e-04 | ✗ |
| VisA | 10 | 0.878 | 0.951 | -4.44 | 3.28e-05 | ✗ |
| VisA | 20 | 0.887 | 0.951 | -4.06 | 1.22e-04 | ✗ |

### GaLAD vs dinov3_k1
| Benchmark | N | GaLAD mean | Other mean | t | p | Significant |
|---|---|---|---|---|---|---|
| AutoVI | 5 | 0.490 | 0.451 | 0.51 | 6.11e-01 | n.s. |
| AutoVI | 10 | 0.515 | 0.458 | 0.77 | 4.47e-01 | n.s. |
| AutoVI | 20 | 0.516 | 0.456 | 0.76 | 4.49e-01 | n.s. |
| BTAD | 5 | 0.993 | 0.988 | 2.10 | 4.53e-02 | ✓ |
| BTAD | 10 | 0.992 | 0.990 | 1.23 | 2.29e-01 | n.s. |
| BTAD | 20 | 0.991 | 0.990 | 0.54 | 5.95e-01 | n.s. |
| GoodsAD | 5 | 0.648 | 0.591 | 2.59 | 1.26e-02 | ✓ |
| GoodsAD | 10 | 0.655 | 0.589 | 3.11 | 3.11e-03 | ✓ |
| GoodsAD | 20 | 0.664 | 0.591 | 3.35 | 1.54e-03 | ✓ |
| MVTec AD | 5 | 0.967 | 0.963 | 0.43 | 6.71e-01 | n.s. |
| MVTec AD | 10 | 0.973 | 0.967 | 0.79 | 4.31e-01 | n.s. |
| MVTec AD | 20 | 0.979 | 0.968 | 1.62 | 1.07e-01 | n.s. |
| MVTec LOCO | 5 | 0.789 | 0.786 | 0.10 | 9.19e-01 | n.s. |
| MVTec LOCO | 10 | 0.810 | 0.795 | 0.46 | 6.48e-01 | n.s. |
| MVTec LOCO | 20 | 0.828 | 0.798 | 0.86 | 3.95e-01 | n.s. |
| VisA | 5 | 0.874 | 0.856 | 0.78 | 4.37e-01 | n.s. |
| VisA | 10 | 0.878 | 0.862 | 0.68 | 5.00e-01 | n.s. |
| VisA | 20 | 0.887 | 0.866 | 0.93 | 3.55e-01 | n.s. |

## 4. Global statistical analysis (Demsar 2006)

Following the original paper's protocol, we average AUPR over 5 seeds 
and training sizes $N \in \{5,10,20\}$ to obtain one value per 
(category, method) pair. PaDiM is excluded from the omnibus rank test 
(consistently weaker, not in the same performance range).

### 4.1 Original 4-method analysis (replicated for comparison)

- N=46 categories, k=4 methods
- Friedman: chi^2=12.31, p=6.402e-03
- Average ranks (lower is better):
  - galad: 1.924
  - patchcore: 2.652
  - dinomaly: 2.707
  - reverse_distillation: 2.717
- Critical difference (Nemenyi, alpha=0.05): 0.692

Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):
  - vs dinomaly               p=2.533e-03  alpha_adj=0.0167  reject=True  mean_diff=+0.0296  cliffs_d=+0.067 (negligible)
  - vs reverse_distillation   p=4.778e-03  alpha_adj=0.0250  reject=True  mean_diff=+0.0290  cliffs_d=+0.114 (negligible)
  - vs patchcore              p=5.936e-03  alpha_adj=0.0500  reject=True  mean_diff=+0.0312  cliffs_d=+0.100 (negligible)

### 4.2 Extended analysis with EfficientAD (5 methods)

- N=46 categories, k=5 methods
- Friedman: chi^2=50.72, p=2.554e-10
- Average ranks (lower is better):
  - galad: 1.989
  - patchcore: 2.870
  - reverse_distillation: 2.913
  - dinomaly: 2.924
  - efficient_ad: 4.304
- Critical difference (Nemenyi, alpha=0.05): 0.899

Pairwise Nemenyi comparisons (significant marked with ***):
  - galad vs patchcore: rank diff=0.880
  - galad vs reverse_distillation: rank diff=0.924 ***
  - galad vs dinomaly: rank diff=0.935 ***
  - galad vs efficient_ad: rank diff=2.315 ***
  - patchcore vs reverse_distillation: rank diff=0.043
  - patchcore vs dinomaly: rank diff=0.054
  - patchcore vs efficient_ad: rank diff=1.435 ***
  - reverse_distillation vs dinomaly: rank diff=0.011
  - reverse_distillation vs efficient_ad: rank diff=1.391 ***
  - dinomaly vs efficient_ad: rank diff=1.380 ***

Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):
  - vs efficient_ad           p=3.979e-13  alpha_adj=0.0125  reject=True  mean_diff=+0.1079  cliffs_d=+0.373 (medium)
  - vs dinomaly               p=2.533e-03  alpha_adj=0.0167  reject=True  mean_diff=+0.0296  cliffs_d=+0.067 (negligible)
  - vs reverse_distillation   p=4.778e-03  alpha_adj=0.0250  reject=True  mean_diff=+0.0290  cliffs_d=+0.114 (negligible)
  - vs patchcore              p=5.936e-03  alpha_adj=0.0500  reject=True  mean_diff=+0.0312  cliffs_d=+0.100 (negligible)

### 4.3 Same-backbone analysis (DINOv3 features held constant)

- N=46 categories, k=3 methods
- Friedman: chi^2=10.42, p=5.468e-03
- Average ranks (lower is better):
  - galad: 1.630
  - dinov3_knn: 2.087
  - dinov3_k1: 2.283
- Critical difference (Nemenyi, alpha=0.05): 0.489

Pairwise Nemenyi comparisons:
  - galad vs dinov3_knn: rank diff=0.457
  - galad vs dinov3_k1: rank diff=0.652 ***
  - dinov3_knn vs dinov3_k1: rank diff=0.196

Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):
  - vs dinov3_k1              p=1.844e-05  alpha_adj=0.0250  reject=True  mean_diff=+0.0232  cliffs_d=+0.074 (negligible)
  - vs dinov3_knn             p=2.071e-01  alpha_adj=0.0500  reject=False  mean_diff=+0.0049  cliffs_d=+0.010 (negligible)

## 5. Same-backbone comparison per benchmark (image-level AUPR)

This is the critical comparison demanded by Reviewers #4 and #7. 
All three methods use identical DINOv3 ViT-L/16 features and PCA preprocessing. 
The only difference is the scoring mechanism: kNN (PatchCore-style), 
single Gaussian (K=1), or 3-component GMM (GaLAD).

### N=5
| Benchmark | galad | dinov3_knn | dinov3_k1 |
|---|---|---|---|
| AutoVI | 0.490±0.082 | 0.442±0.026 | 0.451±0.043 |
| BTAD | 0.993±0.002 | 0.901±0.079 | 0.988±0.007 |
| GoodsAD | 0.648±0.029 | 0.570±0.014 | 0.591±0.014 |
| MVTec AD | 0.967±0.013 | 0.950±0.020 | 0.963±0.010 |
| MVTec LOCO | 0.789±0.037 | 0.737±0.031 | 0.786±0.024 |
| VisA | 0.874±0.029 | 0.938±0.012 | 0.856±0.019 |

### N=10
| Benchmark | galad | dinov3_knn | dinov3_k1 |
|---|---|---|---|
| AutoVI | 0.515±0.050 | 0.463±0.022 | 0.458±0.038 |
| BTAD | 0.992±0.002 | 0.943±0.053 | 0.990±0.003 |
| GoodsAD | 0.655±0.023 | 0.580±0.014 | 0.589±0.008 |
| MVTec AD | 0.973±0.022 | 0.972±0.011 | 0.967±0.012 |
| MVTec LOCO | 0.810±0.045 | 0.809±0.020 | 0.795±0.017 |
| VisA | 0.878±0.025 | 0.951±0.010 | 0.862±0.018 |

### N=20
| Benchmark | galad | dinov3_knn | dinov3_k1 |
|---|---|---|---|
| AutoVI | 0.516±0.050 | 0.493±0.054 | 0.456±0.020 |
| BTAD | 0.991±0.002 | 0.951±0.018 | 0.990±0.003 |
| GoodsAD | 0.664±0.014 | 0.588±0.012 | 0.591±0.009 |
| MVTec AD | 0.979±0.013 | 0.978±0.010 | 0.968±0.008 |
| MVTec LOCO | 0.828±0.010 | 0.827±0.020 | 0.798±0.014 |
| VisA | 0.887±0.018 | 0.951±0.010 | 0.866±0.013 |

## 6. Localization (AUsPRO) for new baselines

AUsPRO per benchmark at N=5 (including EfficientAD):

| Benchmark | galad | patchcore | reverse_distillation | dinomaly | padim | efficient_ad |
|---|---|---|---|---|---|---|
| AutoVI | 0.654±0.080 | 0.650±0.066 | 0.672±0.060 | 0.518±0.066 | 0.536±0.092 | 0.104±0.049 |
| BTAD | 0.918±0.044 | 0.846±0.036 | 0.821±0.233 | 0.893±0.030 | 0.612±0.054 | 0.392±0.104 |
| GoodsAD | 0.698±0.038 | 0.565±0.026 | 0.626±0.019 | 0.660±0.031 | 0.502±0.048 | 0.143±0.074 |
| MPDD | - | 0.814±0.053 | 0.841±0.038 | 0.888±0.030 | 0.656±0.057 | - |
| MVTec AD | 0.910±0.039 | 0.771±0.039 | 0.817±0.047 | 0.871±0.055 | 0.549±0.099 | 0.605±0.124 |
| MVTec LOCO | 0.632±0.053 | 0.411±0.021 | 0.506±0.031 | 0.306±0.060 | 0.271±0.079 | 0.332±0.077 |
| VAD | - | 0.842±nan | 0.803±nan | 0.613±nan | 0.656±nan | - |
| VisA | 0.901±0.057 | 0.756±0.039 | 0.795±0.035 | 0.804±0.053 | 0.514±0.065 | 0.447±0.101 |

## 7. TPR@TNR=95% for EfficientAD

EfficientAD TPR@TNR=95% (N=5) per benchmark:

- AutoVI: 0.074
- BTAD: 0.379
- GoodsAD: 0.074
- MVTec AD: 0.455
- MVTec LOCO: 0.131
- VisA: 0.251

---

Generated tables in `output/tables_revision/`
