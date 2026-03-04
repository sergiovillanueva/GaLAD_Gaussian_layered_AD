"""
Generate paper tables from experimental results.

Creates per-table CSVs with proper statistics for the paper:
- T2_average_detection.csv: Average metrics across benchmarks per N
- T2_per_benchmark.csv: Per-benchmark results at N=5 (AUPR primary)
- T3_localization.csv: AUsPRO per benchmark at N=5
- significance_matrix.csv: Welch's t-test p-values

Usage: uv run generate_tables.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Import shared config from results_summary
from results_summary import MODELS, BENCHMARKS, load_results

# Metrics for each table
DETECTION_METRICS = ["img_aupr", "img_auroc", "tpr_tnr95"]
LOCALIZATION_METRICS = ["auspro"]

# Train limits for paper (exclude N=50)
TRAIN_LIMITS = [2, 5, 10, 20]

# Reference model for significance tests
REFERENCE_MODEL = "galad"

# Output directory
OUTPUT_DIR = Path("results/tables")


def welch_ttest(group1, group2):
    """Welch's t-test, returns p-value. NaN if insufficient data."""
    g1 = np.array(group1).astype(float)
    g2 = np.array(group2).astype(float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    _, p = stats.ttest_ind(g1, g2, equal_var=False)
    return p


def significance_marker(p, ref_better=True):
    """Return significance marker. Only mark if reference (GaLAD) is better."""
    if pd.isna(p) or not ref_better:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_cell(mean, std, sig=""):
    """Format as 'mean ± std' with significance marker."""
    if pd.isna(mean):
        return "-"
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}{sig}"
    return f"{mean:.3f} ± {std:.3f}{sig}"


def compute_per_class_stats(df_raw, metrics):
    """Compute per-class mean/std over seeds."""
    results = []
    for (dataset, train_limit, model), group in df_raw.groupby(["dataset", "train_limit", "model"]):
        row = {"dataset": dataset, "train_limit": train_limit, "model": model, "n_seeds": len(group)}
        for metric in metrics:
            if metric in group.columns:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = vals.mean()
                row[f"{metric}_std"] = vals.std(ddof=1)  # Sample std
        results.append(row)
    return pd.DataFrame(results)


def compute_benchmark_stats(df_per_class, metrics):
    """Aggregate per-class to per-benchmark with pooled std."""
    results = []
    for benchmark, datasets in BENCHMARKS.items():
        for train_limit in df_per_class["train_limit"].unique():
            for model in df_per_class["model"].unique():
                mask = (
                    df_per_class["dataset"].isin(datasets) &
                    (df_per_class["train_limit"] == train_limit) &
                    (df_per_class["model"] == model)
                )
                subset = df_per_class[mask]
                if len(subset) == 0:
                    continue

                row = {
                    "benchmark": benchmark,
                    "train_limit": train_limit,
                    "model": model,
                    "n_classes": len(subset),
                }

                for metric in metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    if mean_col in subset.columns:
                        # Mean of class means
                        row[f"{metric}_mean"] = subset[mean_col].mean()
                        # Pooled std: sqrt(mean of variances)
                        variances = subset[std_col] ** 2
                        row[f"{metric}_std"] = np.sqrt(variances.mean())
                results.append(row)
    return pd.DataFrame(results)


def compute_average_stats(df_benchmark, metrics):
    """Compute average across all benchmarks."""
    results = []
    for train_limit in df_benchmark["train_limit"].unique():
        for model in df_benchmark["model"].unique():
            mask = (df_benchmark["train_limit"] == train_limit) & (df_benchmark["model"] == model)
            subset = df_benchmark[mask]
            if len(subset) == 0:
                continue

            row = {"train_limit": train_limit, "model": model, "n_benchmarks": len(subset)}
            for metric in metrics:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                if mean_col in subset.columns:
                    row[f"{metric}_mean"] = subset[mean_col].mean()
                    variances = subset[std_col] ** 2
                    row[f"{metric}_std"] = np.sqrt(variances.mean())
            results.append(row)
    return pd.DataFrame(results)


def compute_significance(df_raw, metrics):
    """Compute Welch's t-test for GaLAD vs all other models."""
    results = []
    for benchmark, datasets in BENCHMARKS.items():
        bench_data = df_raw[df_raw["dataset"].isin(datasets)]
        for train_limit in bench_data["train_limit"].unique():
            tl_data = bench_data[bench_data["train_limit"] == train_limit]
            galad_data = tl_data[tl_data["model"] == REFERENCE_MODEL]
            if len(galad_data) == 0:
                continue

            for model in tl_data["model"].unique():
                if model == REFERENCE_MODEL:
                    continue
                model_data = tl_data[tl_data["model"] == model]
                row = {"benchmark": benchmark, "train_limit": train_limit, "model": model}

                for metric in metrics:
                    if metric in tl_data.columns:
                        galad_vals = pd.to_numeric(galad_data[metric], errors="coerce").dropna()
                        model_vals = pd.to_numeric(model_data[metric], errors="coerce").dropna()
                        p = welch_ttest(galad_vals, model_vals)
                        galad_mean = galad_vals.mean() if len(galad_vals) > 0 else np.nan
                        model_mean = model_vals.mean() if len(model_vals) > 0 else np.nan
                        galad_better = galad_mean > model_mean if not pd.isna(galad_mean) and not pd.isna(model_mean) else False
                        row[f"{metric}_pvalue"] = p
                        row[f"{metric}_galad_better"] = galad_better
                        row[f"{metric}_sig"] = significance_marker(p, galad_better)
                results.append(row)
    return pd.DataFrame(results)


def generate_T2_average(df_benchmark, df_sig, metrics):
    """Generate T2 average detection table (Model x N)."""
    df_avg = compute_average_stats(df_benchmark, metrics)

    # Filter to paper train limits
    df_avg = df_avg[df_avg["train_limit"].isin(TRAIN_LIMITS)]

    # Build pivot table for each metric
    tables = {}
    for metric in metrics:
        pivot_data = []
        for _, row in df_avg.iterrows():
            mean = row.get(f"{metric}_mean", np.nan)
            std = row.get(f"{metric}_std", np.nan)
            cell = format_cell(mean, std)
            pivot_data.append({
                "model": row["model"],
                "train_limit": row["train_limit"],
                metric: cell,
                f"{metric}_raw": mean,
            })
        pivot_df = pd.DataFrame(pivot_data)
        tables[metric] = pivot_df.pivot(index="model", columns="train_limit", values=metric)

    # Combine into single output
    output_rows = []
    model_order = ["galad", "patchcore", "reverse_distillation", "dinomaly", "padim"]
    for model in model_order:
        for metric in metrics:
            if model in tables[metric].index:
                row_data = {"model": model, "metric": metric}
                for tl in TRAIN_LIMITS:
                    if tl in tables[metric].columns:
                        row_data[f"N={tl}"] = tables[metric].loc[model, tl]
                    else:
                        row_data[f"N={tl}"] = "-"
                output_rows.append(row_data)

    return pd.DataFrame(output_rows)


def generate_T2_benchmark(df_benchmark, df_sig, metric="img_aupr", train_limit=5):
    """Generate T2 per-benchmark table at specific N."""
    subset = df_benchmark[df_benchmark["train_limit"] == train_limit].copy()

    # Merge significance
    if not df_sig.empty:
        sig_cols = ["benchmark", "model", f"{metric}_sig", f"{metric}_galad_better"]
        sig_subset = df_sig[df_sig["train_limit"] == train_limit][sig_cols].copy()
        subset = subset.merge(sig_subset, on=["benchmark", "model"], how="left")

    # Format cells
    rows = []
    for _, row in subset.iterrows():
        mean = row.get(f"{metric}_mean", np.nan)
        std = row.get(f"{metric}_std", np.nan)
        sig = row.get(f"{metric}_sig", "") if row["model"] != REFERENCE_MODEL else ""
        # For GaLAD, mark with * if significantly better than ALL baselines for this benchmark
        if row["model"] == REFERENCE_MODEL and not df_sig.empty:
            bench_sig = df_sig[(df_sig["benchmark"] == row["benchmark"]) & (df_sig["train_limit"] == train_limit)]
            if len(bench_sig) > 0 and all(bench_sig[f"{metric}_galad_better"]) and all(bench_sig[f"{metric}_sig"] != ""):
                sig = "†"  # Mark GaLAD as consistently better
        rows.append({
            "benchmark": row["benchmark"],
            "model": row["model"],
            metric: format_cell(mean, std, sig),
            f"{metric}_raw": mean,
        })

    df_out = pd.DataFrame(rows)
    pivot = df_out.pivot(index="benchmark", columns="model", values=metric)

    # Reorder columns
    col_order = ["galad", "patchcore", "reverse_distillation", "dinomaly", "padim"]
    col_order = [c for c in col_order if c in pivot.columns]
    return pivot[col_order]


def generate_T3_localization(df_benchmark, df_sig, metric="auspro", train_limit=5):
    """Generate T3 localization table."""
    return generate_T2_benchmark(df_benchmark, df_sig, metric=metric, train_limit=train_limit)


def main():
    print("=" * 60)
    print("GENERATE PAPER TABLES")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    df_raw = load_results()
    print(f"Loaded {len(df_raw)} rows")

    # Filter to paper train limits
    df_raw = df_raw[df_raw["train_limit"].isin(TRAIN_LIMITS)]
    print(f"Filtered to train_limits {TRAIN_LIMITS}: {len(df_raw)} rows")

    # All metrics
    all_metrics = DETECTION_METRICS + LOCALIZATION_METRICS

    # Compute per-class stats
    df_per_class = compute_per_class_stats(df_raw, all_metrics)
    print(f"Per-class combinations: {len(df_per_class)}")

    # Compute per-benchmark stats
    df_benchmark = compute_benchmark_stats(df_per_class, all_metrics)
    print(f"Per-benchmark combinations: {len(df_benchmark)}")

    # Compute significance tests
    df_sig = compute_significance(df_raw, all_metrics)
    print(f"Significance tests: {len(df_sig)}")

    # Generate T2 Average
    print("\n--- T2 Average Detection ---")
    t2_avg = generate_T2_average(df_benchmark, df_sig, DETECTION_METRICS)
    t2_avg.to_csv(OUTPUT_DIR / "T2_average_detection.csv", index=False, sep=";")
    print(t2_avg.to_string())

    # Generate T2 Per-Benchmark (AUPR)
    print("\n--- T2 Per-Benchmark (img_aupr, N=5) ---")
    t2_bench = generate_T2_benchmark(df_benchmark, df_sig, metric="img_aupr", train_limit=5)
    t2_bench.to_csv(OUTPUT_DIR / "T2_per_benchmark_aupr.csv", sep=";")
    print(t2_bench.to_string())

    # Generate T2 Per-Benchmark (AUROC)
    print("\n--- T2 Per-Benchmark (img_auroc, N=5) ---")
    t2_auroc = generate_T2_benchmark(df_benchmark, df_sig, metric="img_auroc", train_limit=5)
    t2_auroc.to_csv(OUTPUT_DIR / "T2_per_benchmark_auroc.csv", sep=";")
    print(t2_auroc.to_string())

    # Generate T3 Localization
    print("\n--- T3 Localization (auspro, N=5) ---")
    t3_loc = generate_T3_localization(df_benchmark, df_sig, metric="auspro", train_limit=5)
    t3_loc.to_csv(OUTPUT_DIR / "T3_localization.csv", sep=";")
    print(t3_loc.to_string())

    # Save significance matrix
    print("\n--- Significance Matrix ---")
    df_sig.to_csv(OUTPUT_DIR / "significance_matrix.csv", index=False, sep=";")
    print(f"Saved {len(df_sig)} significance tests")

    # Summary
    print("\n" + "=" * 60)
    print(f"Generated tables in {OUTPUT_DIR}:")
    for f in OUTPUT_DIR.glob("*.csv"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
