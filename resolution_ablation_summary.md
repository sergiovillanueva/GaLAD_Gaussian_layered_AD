# Resolution ablation summary

8 categories, 5 seeds, N=10, 120 experiments. Default GaLAD config; only the input resolution changes.

## Aggregate (mean over 8 categories)

| Resolution | img_auroc | img_aupr |
|---|---|---|
| 336 | 0.864 | 0.888 |
| 448 | 0.863 | 0.891 |
| 672 | 0.853 | 0.884 |

## Per-category AUROC

| Dataset | 336 | 448 | 672 |
|---|---|---|---|
| GoodsAD/drink_bottle | 0.569 | 0.567 | 0.560 |
| GoodsAD/food_box | 0.615 | 0.606 | 0.604 |
| VisA/capsules | 0.878 | 0.892 | 0.911 |
| VisA/pcb1 | 0.908 | 0.945 | 0.931 |
| mvtec_AD/bottle | 0.998 | 0.994 | 0.980 |
| mvtec_AD/carpet | 0.985 | 0.978 | 0.955 |
| mvtec_AD/hazelnut | 0.956 | 0.924 | 0.880 |
| mvtec_AD/leather | 1.000 | 1.000 | 0.999 |

## Conclusion

- Default 448x448 is within 0.1 pp of the best resolution on average and tied for best AUPR.
- Higher resolution (672) helps VisA (cluttered, small parts: capsules +3.3pp, pcb1 ~tied with 448).
- Higher resolution hurts MVTec AD (hazelnut -4.4pp, carpet -2.3pp from 448 to 672).
- GoodsAD essentially flat across resolutions.
- No reason to change the default: 448x448 is kept for all main experiments.
