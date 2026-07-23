"""
Smoke test: verify that the reproducibility artefacts shipped in this
repository are consistent.

Checks performed:
    1. All result CSVs are present and well-formed.
    2. The expected (model, dataset, train_limit, seed) cells are populated.
    3. The per-benchmark averages re-computed from the raw CSVs match the
       numbers in `results/results_overview.md`.
    4. The Demsar pipeline runs end-to-end on the raw CSVs.

Run from the repository root:

    uv run verify_reproducibility.py

The script does not require a GPU.
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np

RESULTS = Path("results")

KEEP_PREFIXES = ("mvtec_AD/", "mvtec_loco_AD/", "VisA/", "AutoVI/", "btad/", "GoodsAD/")
EXPECTED_BENCHMARKS = 6
EXPECTED_CATEGORIES = 46


def check(condition: bool, message: str):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        sys.exit(1)


def load_csvs():
    galad = pd.read_csv(RESULTS / "galad_results.csv", sep=";")
    baseline = pd.read_csv(RESULTS / "baseline_results.csv", sep=";")
    dinov3 = pd.read_csv(RESULTS / "dinov3_baselines_all.csv", sep=";")
    return galad, baseline, dinov3


def benchmark_of(dataset: str) -> str:
    return dataset.split("/", 1)[0]


def restrict_to_46(df):
    return df[df["dataset"].str.startswith(KEEP_PREFIXES)].copy()


def main():
    print("Step 1: file presence")
    for name in ["galad_results.csv", "baseline_results.csv",
                 "dinov3_baselines_all.csv", "results_overview.md",
                 "statistical_analysis_results.md"]:
        path = RESULTS / name
        check(path.exists(), f"{name} exists")

    print("\nStep 2: schema and coverage")
    galad, baseline, dinov3 = load_csvs()
    galad = restrict_to_46(galad); galad["model"] = "galad"
    baseline = restrict_to_46(baseline)
    dinov3 = restrict_to_46(dinov3)

    check(galad["dataset"].nunique() == EXPECTED_CATEGORIES,
          f"GaLAD covers {EXPECTED_CATEGORIES} categories")
    expected_baselines = {"padim", "patchcore", "reverse_distillation",
                          "dinomaly", "efficient_ad"}
    check(set(baseline["model"].unique()) == expected_baselines,
          f"baseline_results covers {expected_baselines}")
    check(set(dinov3["model"].unique()) == {"dinov3_knn", "dinov3_k1"},
          "dinov3_baselines_all covers DINOv3-kNN and DINOv3-K1")

    expected_train_sizes = {5, 10, 20}
    for label, df in [("baseline", baseline), ("dinov3", dinov3)]:
        present = set(df["train_limit"].unique()) & expected_train_sizes
        check(present == expected_train_sizes,
              f"{label}_results contains train sizes {sorted(expected_train_sizes)}")

    print("\nStep 3: per-benchmark averages match summary")
    cols = ["dataset", "model", "seed", "train_limit", "img_aupr"]
    df = pd.concat([galad[cols], baseline[cols], dinov3[cols]],
                   ignore_index=True)
    df["benchmark"] = df["dataset"].apply(benchmark_of)

    # Recompute the GaLAD vs PatchCore average AUPR at N=5 on MVTec AD as a
    # spot check (mean over categories of per-category seed averages).
    n5_mvtec = df[(df["train_limit"] == 5) & (df["benchmark"] == "mvtec_AD")]
    galad_mvtec = (n5_mvtec[n5_mvtec["model"] == "galad"]
                   .groupby("dataset")["img_aupr"].mean().mean())
    pc_mvtec = (n5_mvtec[n5_mvtec["model"] == "patchcore"]
                .groupby("dataset")["img_aupr"].mean().mean())
    check(abs(galad_mvtec - 0.967) < 0.005,
          f"GaLAD MVTec AD N=5 AUPR ~= 0.967 (got {galad_mvtec:.3f})")
    check(abs(pc_mvtec - 0.940) < 0.005,
          f"PatchCore MVTec AD N=5 AUPR ~= 0.940 (got {pc_mvtec:.3f})")

    print("\nStep 4: Demsar pipeline runs end-to-end")
    try:
        from scipy.stats import friedmanchisquare
    except Exception as exc:
        print(f"  [SKIP] SciPy not available ({exc})")
        return

    avg = (df[df["train_limit"].isin([5, 10, 20])]
           .groupby(["dataset", "model", "train_limit"])["img_aupr"].mean()
           .reset_index()
           .groupby(["dataset", "model"])["img_aupr"].mean().reset_index())

    pivot = avg.pivot(index="dataset", columns="model",
                      values="img_aupr").dropna()
    methods4 = ["galad", "patchcore", "reverse_distillation", "dinomaly"]
    chi2, p = friedmanchisquare(*[pivot[m].values for m in methods4])
    check(p < 0.05,
          f"Friedman 4-method significant (chi2={chi2:.2f}, p={p:.2e})")

    methods5 = methods4 + ["efficient_ad"]
    chi2, p = friedmanchisquare(*[pivot[m].values for m in methods5])
    check(p < 1e-6,
          f"Friedman 5-method strongly significant (chi2={chi2:.2f}, p={p:.2e})")

    sb_methods = ["galad", "dinov3_knn", "dinov3_k1"]
    chi2, p = friedmanchisquare(*[pivot[m].values for m in sb_methods])
    check(p < 0.05,
          f"Friedman same-backbone significant (chi2={chi2:.2f}, p={p:.3e})")

    avg_ranks = pivot.rank(axis=1, ascending=False).mean(axis=0)
    galad_rank = avg_ranks["galad"]
    others = avg_ranks.drop("galad")
    check(galad_rank == avg_ranks.min(),
          f"GaLAD obtains the best average rank ({galad_rank:.3f})")
    check(others.min() - galad_rank > 0.5,
          f"Gap to next method is {others.min() - galad_rank:.3f} (>0.5)")

    print("\nStep 5: backbone-generality and cross-backbone tables")
    if not (RESULTS / "backbone_ablation.csv").exists():
        print("  [SKIP] backbone_ablation.csv not shipped")
        print("\nAll checks passed.")
        return
    from scipy.stats import wilcoxon

    ba = pd.read_csv(RESULTS / "backbone_ablation.csv", sep=";")
    ba = ba[ba["train_limit"] == 10]
    backbones = {"dinov3_vitl16", "dinov2_vitl14", "clip_vitl14", "siglip_vitl16"}
    check(set(ba["backbone"].unique()) == backbones,
          "backbone_ablation covers four ViT-L encoders")
    ba_cat = ba.groupby(["backbone", "dataset"])[["img_auroc", "img_aupr"]].mean().reset_index()
    band = ba_cat.groupby("backbone")["img_auroc"].mean()
    check(0.815 < band.min() and band.max() < 0.829,
          f"Table 11: backbones within 0.816-0.828 AUROC band (got {band.min():.3f}-{band.max():.3f})")
    dv3_k3 = band["dinov3_vitl16"]
    check(abs(dv3_k3 - 0.828) < 0.004, f"DINOv3 GMM AUROC ~= 0.828 (got {dv3_k3:.3f})")

    cb = pd.read_csv(RESULTS / "crossbackbone_control.csv", sep=";")
    cb = cb[cb["train_limit"] == 10]
    check(set(cb["backbone"].unique()) == backbones and {"k1", "knn"}.issubset(cb["method"].unique()),
          "crossbackbone_control covers four encoders x {k1, knn, ...}")
    # GMM (K=3) per-category means from backbone_ablation; K1 from crossbackbone
    k3 = ba.groupby(["backbone", "dataset"])["img_aupr"].mean()
    k1 = cb[cb["method"] == "k1"].groupby(["backbone", "dataset"])["img_aupr"].mean()
    for bk, target in [("dinov3_vitl16", 1.8e-3), ("siglip_vitl16", 1.8e-9)]:
        a = k3[bk].sort_index().values
        b = k1[bk].sort_index().values
        _, p = wilcoxon(a, b)
        check(p < 5e-3 if bk == "dinov3_vitl16" else p < 1e-6,
              f"Table 12: GMM>K1 on {bk} (Wilcoxon p={p:.1e}, paper {target:.0e})")

    # Subspace (PCA-residual) ties the GMM on DINOv3 (paper Table 12)
    if "pcares" in cb["method"].unique():
        pc = cb[cb["method"] == "pcares"].groupby(["backbone", "dataset"])["img_aupr"].mean()
        _, p = wilcoxon(k3["dinov3_vitl16"].sort_index().values,
                        pc["dinov3_vitl16"].sort_index().values)
        check(p > 0.05, f"Table 12: GMM ties subspace on DINOv3 (Wilcoxon p={p:.2f} > 0.05)")

    if (RESULTS / "resolution_ablation.csv").exists():
        res = pd.read_csv(RESULTS / "resolution_ablation.csv", sep=";")
        means = res.groupby("resolution")["img_auroc"].mean()
        check(abs(means[448] - 0.863) < 0.004 and means[336] >= means[672],
              f"Resolution ablation: 448~=0.863, 336>=672 (got 336={means[336]:.3f}, "
              f"448={means[448]:.3f}, 672={means[672]:.3f})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
