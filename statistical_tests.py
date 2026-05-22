"""
Statistical comparison of ML models following Demsar (2006).
Friedman omnibus + Nemenyi post-hoc + Wilcoxon with Holm-Bonferroni + Cliff's delta.

Usage: uv run statistical_tests.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, studentized_range
from itertools import combinations
import math

# --- Configuration ---
OUTPUT_DIR = Path("results")
METRICS = ["img_auroc", "img_aupr"]
ALPHA = 0.05

MODELS = ["galad", "patchcore", "reverse_distillation", "dinomaly"]
REFERENCE_MODEL = "galad"

ANALYSES = [
    {"name": "N=5 (4 models)", "train_limits": [5]},
    {"name": "N=10 (4 models)", "train_limits": [10]},
    {"name": "N=20 (4 models)", "train_limits": [20]},
    {"name": "N=5+10 averaged (4 models)", "train_limits": [5, 10]},
    {"name": "N=10+20 averaged (4 models)", "train_limits": [10, 20]},
    {"name": "N=5+10+20 averaged (4 models)", "train_limits": [5, 10, 20]},
]


def load_and_prepare():
    """Load both CSVs, merge, deduplicate."""
    galad = pd.read_csv(OUTPUT_DIR / "galad_results.csv", sep=";")
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_results.csv", sep=";")
    df = pd.concat([galad, baseline], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "model", "train_limit", "seed"], keep="last")
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets")
    return df


def average_over_seeds(df, metric):
    """Average metric over seeds for each (dataset, model, train_limit)."""
    grouped = df.groupby(["dataset", "model", "train_limit"])[metric].mean().reset_index()
    grouped.columns = ["dataset", "model", "train_limit", "score"]
    return grouped


def average_over_train_limits(df_seeds_avg, train_limits):
    """Average across specified train_limits per (dataset, model)."""
    subset = df_seeds_avg[df_seeds_avg["train_limit"].isin(train_limits)]
    averaged = subset.groupby(["dataset", "model"])["score"].mean().reset_index()
    return averaged


def build_score_matrix(df_avg, models):
    """Build (datasets x models) matrix. Only keep datasets with ALL models present."""
    df_filtered = df_avg[df_avg["model"].isin(models)]
    matrix = df_filtered.pivot(index="dataset", columns="model", values="score")
    matrix = matrix.dropna(subset=models)
    matrix = matrix[models]
    print(f"Score matrix: {matrix.shape[0]} datasets x {matrix.shape[1]} models")
    return matrix


def friedman_test(score_matrix):
    """Friedman omnibus test."""
    k = score_matrix.shape[1]
    N = score_matrix.shape[0]

    ranks = score_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0)

    groups = [score_matrix[col].values for col in score_matrix.columns]
    chi2_stat, p_value = friedmanchisquare(*groups)

    print(f"\nFriedman test: chi2={chi2_stat:.4f}, p={p_value:.2e}, k={k}, N={N}")
    print("Average ranks (1=best):")
    for m in avg_ranks.sort_values().index:
        print(f"  {m:<25} {avg_ranks[m]:.3f}")

    if p_value < ALPHA:
        print(f"  -> SIGNIFICANT at alpha={ALPHA}")
    else:
        print(f"  -> NOT significant at alpha={ALPHA}")

    return chi2_stat, p_value, avg_ranks


def nemenyi_cd(avg_ranks, N, alpha=0.05):
    """Nemenyi post-hoc test. Returns CD and pairwise results."""
    k = len(avg_ranks)
    q_tukey = studentized_range.ppf(1 - alpha, k, df=1e10)
    q_alpha = q_tukey / math.sqrt(2)
    CD = q_alpha * math.sqrt(k * (k + 1) / (6 * N))

    print(f"\nNemenyi: q_alpha={q_alpha:.4f}, CD={CD:.4f}")

    models = avg_ranks.sort_values().index.tolist()
    results = []
    for m1, m2 in combinations(models, 2):
        diff = abs(avg_ranks[m1] - avg_ranks[m2])
        sig = diff > CD
        results.append({"model_1": m1, "model_2": m2,
                        "rank_diff": round(diff, 4), "CD": round(CD, 4),
                        "significant": sig})
        marker = " ***" if sig else ""
        print(f"  {m1} vs {m2}: |{avg_ranks[m1]:.3f} - {avg_ranks[m2]:.3f}| "
              f"= {diff:.3f} {'>' if sig else '<='} {CD:.3f}{marker}")

    return CD, pd.DataFrame(results)


def wilcoxon_holm(score_matrix, reference="galad"):
    """Wilcoxon signed-rank with Holm-Bonferroni correction (1-vs-all)."""
    ref_scores = score_matrix[reference].values
    comparisons = [m for m in score_matrix.columns if m != reference]

    raw_results = []
    for model in comparisons:
        other_scores = score_matrix[model].values
        diff = ref_scores - other_scores
        try:
            stat, p = wilcoxon(diff, alternative="two-sided", method="auto")
        except ValueError:
            stat, p = 0, 1.0
        raw_results.append({
            "model": model, "W": stat, "p_raw": p,
            "mean_diff": np.mean(diff),
            "ref_better": np.mean(diff) > 0,
        })

    df_r = pd.DataFrame(raw_results).sort_values("p_raw").reset_index(drop=True)

    # Holm-Bonferroni correction (step-down)
    m = len(df_r)
    df_r["rank"] = range(1, m + 1)
    df_r["alpha_adj"] = ALPHA / (m + 1 - df_r["rank"])
    df_r["reject"] = df_r["p_raw"] < df_r["alpha_adj"]

    reject = True
    for i in range(len(df_r)):
        if not reject:
            df_r.loc[df_r.index[i], "reject"] = False
        elif not df_r.iloc[i]["reject"]:
            reject = False

    print(f"\nWilcoxon signed-rank + Holm-Bonferroni ({reference} vs each):")
    for _, row in df_r.iterrows():
        sig = "REJECT H0" if row["reject"] else "fail to reject"
        direction = f"{reference} better" if row["ref_better"] else f"{row['model']} better"
        print(f"  vs {row['model']:<25} W={row['W']:>7.0f}  p={row['p_raw']:.2e}  "
              f"adj_alpha={row['alpha_adj']:.4f}  {sig}  ({direction}, diff={row['mean_diff']:+.4f})")

    return df_r


def cliffs_delta(x, y):
    """Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    delta = (more - less) / (n_x * n_y)

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return delta, interp


def compute_all_cliffs_delta(score_matrix, reference="galad"):
    """Cliff's delta for reference vs all other models."""
    ref_scores = score_matrix[reference].values
    comparisons = [m for m in score_matrix.columns if m != reference]

    results = []
    print(f"\nCliff's delta ({reference} vs each):")
    for model in comparisons:
        other_scores = score_matrix[model].values
        delta, interp = cliffs_delta(ref_scores, other_scores)
        direction = f"{reference} > {model}" if delta > 0 else f"{model} > {reference}"
        results.append({"model": model, "delta": round(delta, 4), "magnitude": interp})
        print(f"  vs {model:<25} delta={delta:+.4f}  ({interp})  {direction}")

    return pd.DataFrame(results)


def run_analysis(df_raw, config, metric):
    """Run full Demsar pipeline for one configuration."""
    name = config["name"]
    train_limits = config["train_limits"]
    models = config.get("models") or MODELS

    print("\n" + "=" * 70)
    print(f"ANALYSIS: {name}")
    print(f"Metric: {metric}, Train limits: {train_limits}, Models: {models}")
    print("=" * 70)

    df_seeds = average_over_seeds(df_raw, metric)
    df_avg = average_over_train_limits(df_seeds, train_limits)
    matrix = build_score_matrix(df_avg, models)

    if matrix.shape[0] < 10:
        print("WARNING: fewer than 10 datasets, results may be unreliable.")

    # Friedman
    chi2_stat, p_value, avg_ranks = friedman_test(matrix)

    if p_value >= ALPHA:
        print("Friedman not significant. Skipping post-hoc tests.")
        return

    # Nemenyi
    CD, nemenyi_results = nemenyi_cd(avg_ranks, N=matrix.shape[0], alpha=ALPHA)

    # Wilcoxon + Holm-Bonferroni
    ref = REFERENCE_MODEL if REFERENCE_MODEL in models else models[0]
    wilcoxon_results = wilcoxon_holm(matrix, reference=ref)

    # Cliff's delta
    delta_results = compute_all_cliffs_delta(matrix, reference=ref)

    # Summary
    print("\n--- SUMMARY ---")
    summary = wilcoxon_results[["model", "p_raw", "reject", "mean_diff"]].merge(
        delta_results, on="model"
    )
    print(summary.to_string(index=False))


def main():
    print("Statistical Comparison of Models (Demsar 2006)")
    print("=" * 70)

    df_raw = load_and_prepare()

    # Data overview
    print("\nData overview (datasets per model/train_limit):")
    for model in sorted(df_raw["model"].unique()):
        counts = []
        for tl in sorted(df_raw["train_limit"].unique()):
            n = df_raw[(df_raw["model"] == model) &
                       (df_raw["train_limit"] == tl)]["dataset"].nunique()
            counts.append(f"N={tl}:{n}")
        print(f"  {model:<25} {', '.join(counts)}")

    for metric in METRICS:
        print(f"\n{'#' * 70}")
        print(f"# METRIC: {metric}")
        print(f"{'#' * 70}")
        for config in ANALYSES:
            run_analysis(df_raw, config, metric=metric)


if __name__ == "__main__":
    main()
