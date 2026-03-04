"""
Results summary script for paper.

Reads baseline_results.csv and galad_results.csv, computes mean ± std over seeds,
performs Welch's t-test for significance, and outputs:
  - results/results_per_class.csv: One row per (dataset, train_size, model)
  - results/results_per_benchmark.csv: Aggregated by benchmark (MVTec AD, LOCO, VisA, AutoVI)

Usage: uv run results_summary.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Models to include (comment out to exclude)
MODELS = [
    "padim",
    "dinomaly",
    "patchcore",
    "reverse_distillation",
    "galad",
]

# Metrics to report
METRICS = ["img_auroc", "img_aupr", "tpr_tnr90", "tpr_tnr95", "pix_auroc", "spro", "auspro"]

# Benchmark groupings (comment out to exclude)
BENCHMARKS = {
    "MVTec AD": [f"mvtec_AD/{c}" for c in [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ]],
    "MVTec LOCO": [f"mvtec_loco_AD/{c}" for c in [
        "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"
    ]],
    "VisA": [f"VisA/{c}" for c in [
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    ]],
    "AutoVI": [f"AutoVI/{c}" for c in [
        "underbody_pipes", "underbody_screw", "engine_wiring", "pipe_clip", "tank_screw"
    ]],
    "BTAD": ["btad/01", "btad/02", "btad/03"],
    "GoodsAD": [f"GoodsAD/{c}" for c in [
        "cigarette_box", "drink_bottle", "drink_can", "food_bottle", "food_box", "food_package"
    ]],
}


def validate_metrics(df, source_name):
    """Check for invalid metric values (outside [0,1] range or NaN)."""
    issues = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        # Check NaN
        nan_count = df[metric].isna().sum()
        if nan_count > 0:
            issues.append(f"{metric}: {nan_count} NaN values")
        # Check out of range
        valid = df[metric].dropna()
        out_of_range = valid[(valid < 0) | (valid > 1)]
        if len(out_of_range) > 0:
            issues.append(f"{metric}: {len(out_of_range)} values out of [0,1] range")
    if issues:
        print(f"WARNING: Invalid values in {source_name}:")
        for issue in issues:
            print(f"  - {issue}")
        print("  This may indicate data corruption. Check source CSV files.")
    return len(issues) == 0


def load_results():
    """Load all baseline_results*.csv and galad_results*.csv files."""
    dfs = []
    output_dir = Path("results")

    # Load all baseline_results*.csv files
    for csv_file in sorted(output_dir.glob("baseline_results*.csv")):
        df = pd.read_csv(csv_file, sep=";")
        validate_metrics(df, csv_file.name)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {csv_file.name}")

    # Load all galad_results*.csv files
    for csv_file in sorted(output_dir.glob("galad_results*.csv")):
        df = pd.read_csv(csv_file, sep=";")
        validate_metrics(df, csv_file.name)
        dfs.append(df)
        print(f"Loaded {len(df)} rows from {csv_file.name}")

    if not dfs:
        raise FileNotFoundError("No results CSVs found in results/")

    # Combine and remove duplicates (keep last occurrence)
    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["dataset", "model", "train_limit", "seed"],
        keep="last"
    )
    after = len(combined)
    if before != after:
        print(f"Removed {before - after} duplicate rows (kept last)")

    # Filter by MODELS list
    combined = combined[combined["model"].isin(MODELS)]
    print(f"Filtered to models: {MODELS}")

    return combined


def welch_ttest(group1, group2):
    """Welch's t-test, returns p-value. NaN if insufficient data."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    _, p = stats.ttest_ind(g1, g2, equal_var=False)
    return p


def format_mean_std(mean, std):
    """Format as 'mean ± std' with 3 decimals."""
    if pd.isna(mean):
        return ""
    if pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def compute_per_class(df):
    """Compute per-class results with mean ± std over seeds."""
    results = []

    # Get unique combinations
    groups = df.groupby(["dataset", "train_limit", "model"])

    for (dataset, train_limit, model), group in groups:
        row = {
            "dataset": dataset,
            "train_limit": train_limit,
            "model": model,
            "n_seeds": len(group),
        }

        for metric in METRICS:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
            else:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def compute_per_benchmark(df_per_class):
    """Aggregate per-class results by benchmark."""
    results = []

    for benchmark, datasets in BENCHMARKS.items():
        for train_limit in df_per_class["train_limit"].unique():
            for model in df_per_class["model"].unique():
                # Filter to this benchmark/train_limit/model
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

                # Average of class means, average of class stds (from seeds)
                for metric in METRICS:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    if mean_col in subset.columns:
                        row[f"{metric}_mean"] = subset[mean_col].mean()
                        row[f"{metric}_std"] = subset[std_col].mean()
                    else:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_std"] = np.nan

                results.append(row)

    return pd.DataFrame(results)


def add_significance_tests(df_per_class, df_raw):
    """Add Welch's t-test p-values comparing each model to galad."""
    # Get galad as reference
    galad_ref = "galad"

    results = []

    for (dataset, train_limit), group in df_raw.groupby(["dataset", "train_limit"]):
        models = group["model"].unique()

        if galad_ref not in models:
            continue

        galad_data = group[group["model"] == galad_ref]

        for model in models:
            if model == galad_ref:
                continue

            model_data = group[group["model"] == model]

            row = {
                "dataset": dataset,
                "train_limit": train_limit,
                "model": model,
                "vs": galad_ref,
            }

            for metric in METRICS:
                if metric in group.columns:
                    galad_vals = pd.to_numeric(galad_data[metric], errors="coerce")
                    model_vals = pd.to_numeric(model_data[metric], errors="coerce")
                    p = welch_ttest(galad_vals, model_vals)
                    row[f"{metric}_pvalue"] = p

            results.append(row)

    return pd.DataFrame(results)


def format_output(df_per_class, df_pvalues):
    """Format per-class dataframe for paper output."""
    # Merge p-values
    if not df_pvalues.empty:
        df = df_per_class.merge(
            df_pvalues,
            on=["dataset", "train_limit", "model"],
            how="left"
        )
    else:
        df = df_per_class.copy()

    # Create formatted columns
    output_cols = ["dataset", "train_limit", "model", "n_seeds"]

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        pval_col = f"{metric}_pvalue"

        if mean_col in df.columns:
            # Format mean ± std
            df[metric] = df.apply(
                lambda r: format_mean_std(r.get(mean_col), r.get(std_col)),
                axis=1
            )
            output_cols.append(metric)

            # Add significance marker
            if pval_col in df.columns:
                df[f"{metric}_sig"] = df[pval_col].apply(
                    lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                )

    return df[output_cols]


def delete_previous_outputs():
    """Delete previous output CSVs to ensure fresh results."""
    out_dir = Path("results")
    for fname in ["results_per_class.csv", "results_per_benchmark.csv"]:
        fpath = out_dir / fname
        if fpath.exists():
            fpath.unlink()
            print(f"Deleted previous: {fname}")


def main():
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Delete previous outputs
    delete_previous_outputs()

    # Load data
    df_raw = load_results()
    print(f"Total rows: {len(df_raw)}")
    print(f"Models: {df_raw['model'].unique().tolist()}")
    print(f"Train sizes: {df_raw['train_limit'].unique().tolist()}")

    # Validate seeds per experiment
    print("\n--- Seed Validation ---")
    expected_seeds = {0, 1, 2, 3, 4}
    incomplete = []
    for (dataset, model, train_limit), group in df_raw.groupby(["dataset", "model", "train_limit"]):
        seeds = set(group["seed"].unique())
        if seeds != expected_seeds:
            missing = expected_seeds - seeds
            incomplete.append(f"{dataset}/{model}/train={train_limit}: seeds={sorted(seeds)}, missing={sorted(missing)}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} experiments with incomplete seeds:")
        for item in incomplete[:10]:
            print(f"  - {item}")
        if len(incomplete) > 10:
            print(f"  ... and {len(incomplete) - 10} more")
    else:
        print("All experiments have 5 seeds (0-4)")

    # Compute per-class statistics
    df_per_class = compute_per_class(df_raw)
    print(f"\nPer-class combinations: {len(df_per_class)}")

    # Compute significance tests
    df_pvalues = add_significance_tests(df_per_class, df_raw)

    # Format per-class output
    df_per_class_out = format_output(df_per_class, df_pvalues)

    # Compute per-benchmark
    df_per_benchmark = compute_per_benchmark(df_per_class)
    print(f"Per-benchmark combinations: {len(df_per_benchmark)}")

    # Format benchmark output
    output_cols = ["benchmark", "train_limit", "model", "n_classes"]
    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in df_per_benchmark.columns:
            df_per_benchmark[metric] = df_per_benchmark.apply(
                lambda r: format_mean_std(r.get(mean_col), r.get(std_col)),
                axis=1
            )
            output_cols.append(metric)

    df_per_benchmark_out = df_per_benchmark[output_cols]

    # Save outputs (use mode that doesn't lock)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    per_class_path = out_dir / "results_per_class.csv"
    per_benchmark_path = out_dir / "results_per_benchmark.csv"

    df_per_class_out.to_csv(per_class_path, index=False, sep=";")
    df_per_benchmark_out.to_csv(per_benchmark_path, index=False, sep=";")

    print(f"\nSaved: {per_class_path}")
    print(f"Saved: {per_benchmark_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (img_auroc)")
    print("=" * 60)

    # Pivot for easy reading
    for train_limit in sorted(df_per_benchmark["train_limit"].unique()):
        print(f"\n--- Train size: {train_limit} ---")
        subset = df_per_benchmark[df_per_benchmark["train_limit"] == train_limit]
        pivot = subset.pivot(index="benchmark", columns="model", values="img_auroc")
        print(pivot.to_string())


if __name__ == "__main__":
    main()
