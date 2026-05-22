"""
Rigorous analysis of new experiments for the paper revision.

Computes:
- Per-benchmark averages with proper std pooling
- Welch's t-test GaLAD vs each new baseline
- Friedman + Nemenyi + Wilcoxon (Demsar) including new methods
- Cliff's delta effect sizes
- LaTeX-ready tables

Outputs:
- results/new_results_summary.md (human-readable summary)
- results/tables_revision/*.csv (LaTeX-ready tables)
"""
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_ind, wilcoxon, studentized_range

OUTPUT_DIR = Path("results")
TABLES_DIR = OUTPUT_DIR / "tables_revision"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_MD = OUTPUT_DIR / "new_results_summary.md"

ALPHA = 0.05


def benchmark_of(dataset: str) -> str:
    bench = dataset.split("/", 1)[0]
    return {"mvtec_AD": "MVTec AD", "mvtec_loco_AD": "MVTec LOCO",
            "VisA": "VisA", "AutoVI": "AutoVI",
            "btad": "BTAD", "GoodsAD": "GoodsAD"}.get(bench, bench)


def load_all():
    """Load and merge all result CSVs into a unified long-format DataFrame."""
    # GaLAD
    galad = pd.read_csv(OUTPUT_DIR / "galad_results.csv", sep=";")
    galad["model"] = "galad"

    # Baselines (padim, patchcore, RD, dinomaly, efficient_ad)
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_results.csv", sep=";")

    # DINOv3 same-backbone
    dinov3 = pd.read_csv(OUTPUT_DIR / "dinov3_baselines_all.csv", sep=";")

    # Unify column names
    cols = ["dataset", "model", "seed", "train_limit", "img_auroc", "img_aupr"]
    galad_u = galad[cols].copy()
    baseline_u = baseline[cols].copy()
    dinov3_u = dinov3[cols].copy()

    df = pd.concat([galad_u, baseline_u, dinov3_u], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "model", "seed", "train_limit"], keep="last")
    df["benchmark"] = df["dataset"].apply(benchmark_of)

    # Restrict to the 46 categories that form the evaluation set in the paper.
    # MPDD, VAD and mvtec_ad_2 may appear in some CSVs but are not part of the
    # main evaluation.
    keep_prefixes = ("mvtec_AD/", "mvtec_loco_AD/", "VisA/", "AutoVI/", "btad/", "GoodsAD/")
    df = df[df["dataset"].str.startswith(keep_prefixes)].copy()

    return df


def summary_by_benchmark(df, metric="img_aupr"):
    """Mean and pooled std per (benchmark, model, train_limit)."""
    # First average over seeds per (dataset, model, train_limit)
    per_cat = df.groupby(["benchmark", "dataset", "model", "train_limit"])[metric].agg(["mean", "std"]).reset_index()
    per_cat.columns = ["benchmark", "dataset", "model", "train_limit", "mean", "std"]

    # Then aggregate across categories within benchmark
    rows = []
    for (bench, model, N), grp in per_cat.groupby(["benchmark", "model", "train_limit"]):
        mean = grp["mean"].mean()
        # Pooled std across categories: sqrt(mean of category variances)
        # If only 1 seed per cat (std=NaN), fall back to std across category means
        var = grp["std"].dropna().pow(2)
        if len(var) >= 2:
            pooled_std = float(np.sqrt(var.mean()))
        else:
            pooled_std = float(grp["mean"].std())
        rows.append({
            "benchmark": bench, "model": model, "train_limit": N,
            "mean": float(mean), "std": pooled_std, "n_categories": len(grp),
        })
    return pd.DataFrame(rows)


def global_average(df, metric="img_aupr"):
    """Global average per (model, train_limit), using the protocol from the paper.

    Protocol (to keep each benchmark weighted equally regardless of its number of
    categories):
      1. For each (dataset, model, N): mean and std over the 5 seeds.
      2. For each (benchmark, model, N): mean of the per-dataset means;
         pooled std at seed level = sqrt(mean of per-dataset variances).
      3. Global value: mean of the per-benchmark means;
         pooled std = sqrt(mean of per-benchmark pooled variances).
    """
    # Per-(dataset, model, N) seed statistics
    per_cat = df.groupby(["dataset", "model", "train_limit"])[metric].agg(["mean", "std"]).reset_index()
    per_cat.columns = ["dataset", "model", "train_limit", "cat_mean", "cat_std"]
    per_cat["benchmark"] = per_cat["dataset"].apply(benchmark_of)

    # Per-(benchmark, model, N) aggregates
    rows_b = []
    for (bench, model, N), grp in per_cat.groupby(["benchmark", "model", "train_limit"]):
        bench_mean = grp["cat_mean"].mean()
        var = grp["cat_std"].dropna().pow(2)
        bench_pooled_std = float(np.sqrt(var.mean())) if len(var) >= 1 else float("nan")
        rows_b.append({"benchmark": bench, "model": model, "train_limit": N,
                       "bench_mean": bench_mean, "bench_pooled_std": bench_pooled_std})
    bench_df = pd.DataFrame(rows_b)

    # Global aggregate over benchmarks
    rows = []
    for (model, N), grp in bench_df.groupby(["model", "train_limit"]):
        global_mean = grp["bench_mean"].mean()
        var = grp["bench_pooled_std"].dropna().pow(2)
        global_std = float(np.sqrt(var.mean())) if len(var) >= 1 else float("nan")
        rows.append({"model": model, "train_limit": N,
                     "mean": float(global_mean), "std": global_std})
    return pd.DataFrame(rows)


def welch_test_per_benchmark(df, ref_model="galad", metric="img_aupr"):
    """Welch's t-test: GaLAD vs each other model, per benchmark.

    Pools 5-seed measurements across all categories within each benchmark
    (same protocol as the paper's existing per-benchmark tests).
    """
    rows = []
    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench]
        for N in sorted(sub["train_limit"].unique()):
            sub_N = sub[sub["train_limit"] == N]
            ref = sub_N[sub_N["model"] == ref_model][metric].values
            if len(ref) == 0:
                continue
            for model in sorted(sub_N["model"].unique()):
                if model == ref_model:
                    continue
                other = sub_N[sub_N["model"] == model][metric].values
                if len(other) == 0:
                    continue
                try:
                    stat, p = ttest_ind(ref, other, equal_var=False)
                except Exception:
                    stat, p = float("nan"), float("nan")
                rows.append({
                    "benchmark": bench, "train_limit": N, "vs_model": model,
                    "n_ref": len(ref), "n_other": len(other),
                    "ref_mean": float(np.mean(ref)), "other_mean": float(np.mean(other)),
                    "t_stat": float(stat), "p_value": float(p),
                    "ref_better": float(np.mean(ref)) > float(np.mean(other)),
                    "significant": (p < ALPHA) if not math.isnan(p) else False,
                })
    return pd.DataFrame(rows)


def average_over_seeds_and_N(df, train_limits, metric="img_aupr"):
    """One value per (category, model) by averaging over seeds and selected N values."""
    sub = df[df["train_limit"].isin(train_limits)]
    avg = sub.groupby(["dataset", "model", "train_limit"])[metric].mean().reset_index()
    return avg.groupby(["dataset", "model"])[metric].mean().reset_index()


def score_matrix(df_avg, models, metric="img_aupr"):
    pivot = df_avg.pivot(index="dataset", columns="model", values=metric)
    pivot = pivot.dropna(subset=models)
    return pivot[models]


def friedman_nemenyi(matrix, name):
    """Friedman omnibus + Nemenyi post-hoc.

    Returns a dict with key statistics.
    """
    k = matrix.shape[1]
    N = matrix.shape[0]
    ranks = matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0)
    groups = [matrix[c].values for c in matrix.columns]
    chi2, p = friedmanchisquare(*groups)

    # Critical difference (Nemenyi)
    q_tukey = studentized_range.ppf(1 - ALPHA, k, df=1e10)
    q_alpha = q_tukey / math.sqrt(2)
    CD = q_alpha * math.sqrt(k * (k + 1) / (6 * N))

    pairwise = []
    for m1, m2 in combinations(avg_ranks.sort_values().index, 2):
        diff = abs(avg_ranks[m1] - avg_ranks[m2])
        pairwise.append({"a": m1, "b": m2, "rank_diff": float(diff), "CD": float(CD),
                         "significant": bool(diff > CD)})

    return {
        "name": name, "k": k, "N": N,
        "chi2": float(chi2), "p_value": float(p),
        "avg_ranks": {m: float(avg_ranks[m]) for m in avg_ranks.index},
        "CD": float(CD), "pairwise": pairwise,
    }


def wilcoxon_holm(matrix, reference):
    ref = matrix[reference].values
    rows = []
    for model in matrix.columns:
        if model == reference:
            continue
        other = matrix[model].values
        diff = ref - other
        try:
            stat, p = wilcoxon(diff, alternative="two-sided", method="auto")
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({"model": model, "W": float(stat), "p_raw": float(p),
                     "mean_diff": float(np.mean(diff)),
                     "ref_better": float(np.mean(diff)) > 0})
    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    m = len(out)
    out["rank"] = range(1, m + 1)
    out["alpha_adj"] = ALPHA / (m + 1 - out["rank"])
    out["reject"] = out["p_raw"] < out["alpha_adj"]
    # Step-down: stop at first non-reject
    keep_rejecting = True
    for i in range(len(out)):
        if not keep_rejecting:
            out.loc[out.index[i], "reject"] = False
        elif not out.iloc[i]["reject"]:
            keep_rejecting = False
    return out


def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    d = (more - less) / (nx * ny)
    ad = abs(d)
    if ad < 0.147:
        m = "negligible"
    elif ad < 0.33:
        m = "small"
    elif ad < 0.474:
        m = "medium"
    else:
        m = "large"
    return float(d), m


def fmt_pair(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def write_md(lines):
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading data...")
    df = load_all()
    print(f"  rows={len(df)}, models={sorted(df['model'].unique())}")
    print(f"  datasets={df['dataset'].nunique()}, benchmarks={df['benchmark'].nunique()}")

    md = []
    md.append("# New Experimental Results - Paper Revision\n")
    md.append("This document summarizes the additional experiments run in response to ")
    md.append("reviewer comments (EfficientAD baseline and DINOv3 same-backbone controls ")
    md.append("extended to all 46 categories) together with re-computed global statistics.\n")

    # Data overview
    md.append("## Data overview\n")
    summary = df.groupby(["model", "train_limit"]).size().unstack(fill_value=0)
    md.append("Number of (dataset, seed) measurements per (model, N):\n")
    md.append("```")
    md.append(summary.to_string())
    md.append("```\n")

    # ============================================
    # 1) GLOBAL AVERAGES across 46 categories
    # ============================================
    md.append("## 1. Global averages across 46 categories (image-level AUPR)\n")
    for metric, label in [("img_aupr", "Image AUPR"), ("img_auroc", "Image AUROC")]:
        g = global_average(df, metric=metric)
        pivot_mean = g.pivot(index="model", columns="train_limit", values="mean")
        pivot_std = g.pivot(index="model", columns="train_limit", values="std")
        # Format combined
        rows = []
        for model in pivot_mean.index:
            row = {"model": model}
            for N in pivot_mean.columns:
                m, s = pivot_mean.loc[model, N], pivot_std.loc[model, N]
                row[f"N={N}"] = f"{m:.3f}±{s:.3f}" if not np.isnan(m) else "-"
            rows.append(row)
        out = pd.DataFrame(rows)
        out_csv = TABLES_DIR / f"global_average_{metric}.csv"
        out.to_csv(out_csv, index=False, sep=";")
        md.append(f"### {label}\n")
        md.append("| Model | " + " | ".join(f"N={N}" for N in pivot_mean.columns) + " |")
        md.append("|" + "---|" * (len(pivot_mean.columns) + 1))
        for _, row in out.iterrows():
            md.append("| " + " | ".join(str(row[c]) for c in out.columns) + " |")
        md.append("")

    # ============================================
    # 2) PER-BENCHMARK at N=5
    # ============================================
    md.append("## 2. Per-benchmark results at N=5 (image-level AUPR)\n")
    bench_summary = summary_by_benchmark(df, metric="img_aupr")
    n5 = bench_summary[bench_summary["train_limit"] == 5]
    if len(n5):
        pivot = n5.pivot(index="benchmark", columns="model", values="mean")
        pivot_std = n5.pivot(index="benchmark", columns="model", values="std")
        models_order = ["galad", "patchcore", "reverse_distillation", "dinomaly",
                        "padim", "efficient_ad", "dinov3_knn", "dinov3_k1"]
        models_order = [m for m in models_order if m in pivot.columns]
        md.append("| Benchmark | " + " | ".join(models_order) + " |")
        md.append("|" + "---|" * (len(models_order) + 1))
        for bench in pivot.index:
            cells = [bench]
            for m in models_order:
                mu, sd = pivot.loc[bench, m], pivot_std.loc[bench, m]
                if np.isnan(mu):
                    cells.append("-")
                else:
                    cells.append(f"{mu:.3f}±{sd:.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
        # CSV
        out = n5.copy()
        out.to_csv(TABLES_DIR / "per_benchmark_N5_aupr.csv", index=False, sep=";")

    # Same for N=10 and N=20
    for N in [10, 20]:
        sub = bench_summary[bench_summary["train_limit"] == N]
        if len(sub) == 0:
            continue
        md.append(f"### Per-benchmark at N={N}\n")
        pivot = sub.pivot(index="benchmark", columns="model", values="mean")
        pivot_std = sub.pivot(index="benchmark", columns="model", values="std")
        models_order = ["galad", "patchcore", "reverse_distillation", "dinomaly",
                        "padim", "efficient_ad", "dinov3_knn", "dinov3_k1"]
        models_order = [m for m in models_order if m in pivot.columns]
        md.append("| Benchmark | " + " | ".join(models_order) + " |")
        md.append("|" + "---|" * (len(models_order) + 1))
        for bench in pivot.index:
            cells = [bench]
            for m in models_order:
                mu, sd = pivot.loc[bench, m], pivot_std.loc[bench, m]
                cells.append("-" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
        sub.to_csv(TABLES_DIR / f"per_benchmark_N{N}_aupr.csv", index=False, sep=";")

    # ============================================
    # 3) Welch's t-test GaLAD vs new baselines per benchmark
    # ============================================
    md.append("## 3. Welch's t-test: GaLAD vs new baselines (per benchmark)\n")
    welch = welch_test_per_benchmark(df, ref_model="galad", metric="img_aupr")
    new_baselines = ["efficient_ad", "dinov3_knn", "dinov3_k1"]
    welch_new = welch[welch["vs_model"].isin(new_baselines)].copy()
    welch_new.to_csv(TABLES_DIR / "welch_tests_new_baselines.csv", index=False, sep=";")
    md.append("Comparison of GaLAD vs the new baselines added in the revision, ")
    md.append("with Welch's $t$-test ($p<0.05$ marked with *):\n")
    for vs_model in new_baselines:
        sub = welch_new[welch_new["vs_model"] == vs_model]
        if len(sub) == 0:
            continue
        md.append(f"### GaLAD vs {vs_model}")
        md.append("| Benchmark | N | GaLAD mean | Other mean | t | p | Significant |")
        md.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            sig = "✓" if r["significant"] and r["ref_better"] else ("✗" if r["significant"] else "n.s.")
            md.append(f"| {r['benchmark']} | {int(r['train_limit'])} | {r['ref_mean']:.3f} | {r['other_mean']:.3f} | {r['t_stat']:.2f} | {r['p_value']:.2e} | {sig} |")
        md.append("")

    # ============================================
    # 4) Demsar protocol (Friedman + Nemenyi + Wilcoxon) - updated with new baselines
    # ============================================
    md.append("## 4. Global statistical analysis (Demsar 2006)\n")
    md.append("Following the original paper's protocol, we average AUPR over 5 seeds ")
    md.append("and training sizes $N \\in \\{5,10,20\\}$ to obtain one value per ")
    md.append("(category, method) pair. PaDiM is excluded from the omnibus rank test ")
    md.append("(consistently weaker, not in the same performance range).\n")

    md.append("### 4.1 Original 4-method analysis (replicated for comparison)\n")
    df_avg = average_over_seeds_and_N(df, [5, 10, 20])
    matrix4 = score_matrix(df_avg, ["galad", "patchcore", "reverse_distillation", "dinomaly"])
    f4 = friedman_nemenyi(matrix4, "Original 4 methods")
    md.append(f"- N={f4['N']} categories, k={f4['k']} methods")
    md.append(f"- Friedman: chi^2={f4['chi2']:.2f}, p={f4['p_value']:.3e}")
    md.append("- Average ranks (lower is better):")
    for m, r in sorted(f4["avg_ranks"].items(), key=lambda kv: kv[1]):
        md.append(f"  - {m}: {r:.3f}")
    md.append(f"- Critical difference (Nemenyi, alpha=0.05): {f4['CD']:.3f}\n")

    # Wilcoxon
    wilc4 = wilcoxon_holm(matrix4, "galad")
    md.append("Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):")
    for _, r in wilc4.iterrows():
        d, dmag = cliffs_delta(matrix4["galad"].values, matrix4[r["model"]].values)
        md.append(f"  - vs {r['model']:<22} p={r['p_raw']:.3e}  alpha_adj={r['alpha_adj']:.4f}  "
                  f"reject={r['reject']}  mean_diff={r['mean_diff']:+.4f}  cliffs_d={d:+.3f} ({dmag})")
    md.append("")
    wilc4.to_csv(TABLES_DIR / "wilcoxon_4methods.csv", index=False, sep=";")

    md.append("### 4.2 Extended analysis with EfficientAD (5 methods)\n")
    matrix5 = score_matrix(df_avg, ["galad", "patchcore", "reverse_distillation", "dinomaly", "efficient_ad"])
    f5 = friedman_nemenyi(matrix5, "Extended 5 methods")
    md.append(f"- N={f5['N']} categories, k={f5['k']} methods")
    md.append(f"- Friedman: chi^2={f5['chi2']:.2f}, p={f5['p_value']:.3e}")
    md.append("- Average ranks (lower is better):")
    for m, r in sorted(f5["avg_ranks"].items(), key=lambda kv: kv[1]):
        md.append(f"  - {m}: {r:.3f}")
    md.append(f"- Critical difference (Nemenyi, alpha=0.05): {f5['CD']:.3f}\n")
    md.append("Pairwise Nemenyi comparisons (significant marked with ***):")
    for pair in f5["pairwise"]:
        marker = " ***" if pair["significant"] else ""
        md.append(f"  - {pair['a']} vs {pair['b']}: rank diff={pair['rank_diff']:.3f}{marker}")
    md.append("")
    wilc5 = wilcoxon_holm(matrix5, "galad")
    md.append("Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):")
    for _, r in wilc5.iterrows():
        d, dmag = cliffs_delta(matrix5["galad"].values, matrix5[r["model"]].values)
        md.append(f"  - vs {r['model']:<22} p={r['p_raw']:.3e}  alpha_adj={r['alpha_adj']:.4f}  "
                  f"reject={r['reject']}  mean_diff={r['mean_diff']:+.4f}  cliffs_d={d:+.3f} ({dmag})")
    md.append("")
    wilc5.to_csv(TABLES_DIR / "wilcoxon_5methods.csv", index=False, sep=";")

    md.append("### 4.3 Same-backbone analysis (DINOv3 features held constant)\n")
    matrix_sb = score_matrix(df_avg, ["galad", "dinov3_knn", "dinov3_k1"])
    f_sb = friedman_nemenyi(matrix_sb, "Same-backbone 3 methods")
    md.append(f"- N={f_sb['N']} categories, k={f_sb['k']} methods")
    md.append(f"- Friedman: chi^2={f_sb['chi2']:.2f}, p={f_sb['p_value']:.3e}")
    md.append("- Average ranks (lower is better):")
    for m, r in sorted(f_sb["avg_ranks"].items(), key=lambda kv: kv[1]):
        md.append(f"  - {m}: {r:.3f}")
    md.append(f"- Critical difference (Nemenyi, alpha=0.05): {f_sb['CD']:.3f}\n")
    md.append("Pairwise Nemenyi comparisons:")
    for pair in f_sb["pairwise"]:
        marker = " ***" if pair["significant"] else ""
        md.append(f"  - {pair['a']} vs {pair['b']}: rank diff={pair['rank_diff']:.3f}{marker}")
    md.append("")
    wilc_sb = wilcoxon_holm(matrix_sb, "galad")
    md.append("Wilcoxon signed-rank + Holm-Bonferroni (GaLAD vs each):")
    for _, r in wilc_sb.iterrows():
        d, dmag = cliffs_delta(matrix_sb["galad"].values, matrix_sb[r["model"]].values)
        md.append(f"  - vs {r['model']:<22} p={r['p_raw']:.3e}  alpha_adj={r['alpha_adj']:.4f}  "
                  f"reject={r['reject']}  mean_diff={r['mean_diff']:+.4f}  cliffs_d={d:+.3f} ({dmag})")
    md.append("")
    wilc_sb.to_csv(TABLES_DIR / "wilcoxon_same_backbone.csv", index=False, sep=";")

    # ============================================
    # 5) Per-benchmark same-backbone analysis (NEW: now covers all 6 benchmarks)
    # ============================================
    md.append("## 5. Same-backbone comparison per benchmark (image-level AUPR)\n")
    md.append("This is the critical comparison demanded by Reviewers #4 and #7. ")
    md.append("All three methods use identical DINOv3 ViT-L/16 features and PCA preprocessing. ")
    md.append("The only difference is the scoring mechanism: kNN (PatchCore-style), ")
    md.append("single Gaussian (K=1), or 3-component GMM (GaLAD).\n")

    sb_models = ["galad", "dinov3_knn", "dinov3_k1"]
    for N in [5, 10, 20]:
        sub = bench_summary[(bench_summary["train_limit"] == N) & (bench_summary["model"].isin(sb_models))]
        if len(sub) == 0:
            continue
        md.append(f"### N={N}")
        pivot = sub.pivot(index="benchmark", columns="model", values="mean")
        pivot_std = sub.pivot(index="benchmark", columns="model", values="std")
        avail = [m for m in sb_models if m in pivot.columns]
        md.append("| Benchmark | " + " | ".join(avail) + " |")
        md.append("|" + "---|" * (len(avail) + 1))
        for bench in pivot.index:
            cells = [bench]
            for m in avail:
                mu, sd = pivot.loc[bench, m], pivot_std.loc[bench, m]
                cells.append("-" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

    # ============================================
    # 6) Localization (AUsPRO) - only from baseline CSV
    # ============================================
    md.append("## 6. Localization (AUsPRO) for new baselines\n")
    if "auspro" in df.columns:
        # Available only for galad and baseline models
        pass
    # Load directly from baseline+galad which have pixel metrics
    galad_pix = pd.read_csv(OUTPUT_DIR / "galad_results.csv", sep=";")
    base_pix = pd.read_csv(OUTPUT_DIR / "baseline_results.csv", sep=";")
    pix_cols = ["dataset", "model", "seed", "train_limit", "auspro"]
    pix_df = pd.concat([galad_pix[pix_cols], base_pix[pix_cols]], ignore_index=True)
    pix_df = pix_df[~pix_df["dataset"].str.startswith("mvtec_ad_2/")].copy()
    pix_df = pix_df.drop_duplicates(subset=["dataset", "model", "seed", "train_limit"], keep="last")
    pix_df["benchmark"] = pix_df["dataset"].apply(benchmark_of)
    pix_df = pix_df.dropna(subset=["auspro"])

    pix_summary = summary_by_benchmark(pix_df, metric="auspro")
    md.append("AUsPRO per benchmark at N=5 (including EfficientAD):\n")
    sub = pix_summary[pix_summary["train_limit"] == 5]
    pivot = sub.pivot(index="benchmark", columns="model", values="mean")
    pivot_std = sub.pivot(index="benchmark", columns="model", values="std")
    models_order = ["galad", "patchcore", "reverse_distillation", "dinomaly", "padim", "efficient_ad"]
    avail = [m for m in models_order if m in pivot.columns]
    md.append("| Benchmark | " + " | ".join(avail) + " |")
    md.append("|" + "---|" * (len(avail) + 1))
    for bench in pivot.index:
        cells = [bench]
        for m in avail:
            mu, sd = pivot.loc[bench, m], pivot_std.loc[bench, m]
            cells.append("-" if np.isnan(mu) else f"{mu:.3f}±{sd:.3f}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    sub.to_csv(TABLES_DIR / "per_benchmark_N5_auspro.csv", index=False, sep=";")

    # ============================================
    # 7) Practical-operating-point (TPR@TNR=95%)
    # ============================================
    md.append("## 7. TPR@TNR=95% for EfficientAD\n")
    if "tpr_tnr95" in base_pix.columns:
        ea = base_pix[(base_pix["model"] == "efficient_ad") & (base_pix["train_limit"] == 5)]
        if len(ea):
            ea = ea.copy()
            ea["benchmark"] = ea["dataset"].apply(benchmark_of)
            mean_by_bench = ea.groupby("benchmark")["tpr_tnr95"].mean()
            md.append("EfficientAD TPR@TNR=95% (N=5) per benchmark:\n")
            for bench, v in mean_by_bench.items():
                md.append(f"- {bench}: {v:.3f}")
            md.append("")

    md.append("---\n")
    md.append("Generated tables in `results/tables_revision/`\n")
    write_md(md)
    print(f"\nSummary written to {SUMMARY_MD}")
    print(f"Tables in {TABLES_DIR}/")


if __name__ == "__main__":
    main()
