"""
Generate Appendix Tables for GaLAD Paper

Creates CSV files in results/appendix/ for supplementary material.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config
OUTPUT_DIR = Path("results/appendix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
galad = pd.read_csv("results/galad_results.csv", sep=";")
baselines = pd.read_csv("results/baseline_results.csv", sep=";")

# Combine
galad['model'] = 'GaLAD'
all_data = pd.concat([galad, baselines], ignore_index=True)

# Extract benchmark from dataset
def get_benchmark(dataset):
    if dataset.startswith("mvtec_AD/"):
        return "MVTec AD"
    elif dataset.startswith("mvtec_loco_AD/"):
        return "MVTec LOCO"
    elif dataset.startswith("VisA/"):
        return "VisA"
    elif dataset.startswith("AutoVI/"):
        return "AutoVI"
    elif dataset.startswith("btad/") or dataset.startswith("BTAD/"):
        return "BTAD"
    elif dataset.startswith("GoodsAD/"):
        return "GoodsAD"
    elif dataset.startswith("mvtec_ad_2/"):
        return "MVTec AD 2"  # Extra categories, exclude from main
    return "Other"

all_data['benchmark'] = all_data['dataset'].apply(get_benchmark)
all_data['category'] = all_data['dataset'].apply(lambda x: x.split("/")[-1])

print(f"Total rows: {len(all_data)}")
print(f"Models: {all_data['model'].unique()}")
print(f"Benchmarks: {all_data['benchmark'].unique()}")
print(f"Train limits: {sorted(all_data['train_limit'].unique())}")

# =============================================================================
# Table A1-A6: Per-category results at N=5
# =============================================================================
def generate_per_category_table(benchmark_name, n=5):
    """Generate per-category results for a benchmark."""
    df = all_data[(all_data['benchmark'] == benchmark_name) & (all_data['train_limit'] == n)]

    if len(df) == 0:
        print(f"  No data for {benchmark_name} at N={n}")
        return None

    # Aggregate by category and model
    agg = df.groupby(['category', 'model']).agg({
        'img_auroc': ['mean', 'std'],
        'img_aupr': ['mean', 'std'],
        'auspro': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    agg.columns = ['category', 'model',
                   'auroc_mean', 'auroc_std',
                   'aupr_mean', 'aupr_std',
                   'auspro_mean', 'auspro_std']

    # Pivot to have models as columns
    result = []
    for cat in sorted(agg['category'].unique()):
        row = {'category': cat}
        cat_data = agg[agg['category'] == cat]
        for _, r in cat_data.iterrows():
            model = r['model']
            row[f'{model}_auroc'] = f"{r['auroc_mean']:.3f}"
            row[f'{model}_aupr'] = f"{r['aupr_mean']:.3f}"
            row[f'{model}_auspro'] = f"{r['auspro_mean']:.3f}"
        result.append(row)

    return pd.DataFrame(result)

print("\n" + "="*60)
print("Generating per-category tables (N=5)")
print("="*60)

benchmarks = ["MVTec AD", "MVTec LOCO", "VisA", "AutoVI", "BTAD", "GoodsAD"]
for i, bench in enumerate(benchmarks, 1):
    print(f"\nA{i}: {bench}")
    table = generate_per_category_table(bench, n=5)
    if table is not None:
        filename = f"A{i}_{bench.replace(' ', '_').lower()}_n5.csv"
        table.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"  Saved: {filename}")
        print(f"  Categories: {len(table)}")

# =============================================================================
# Table A7: TPR@TNR=95% (Industrial metric)
# =============================================================================
print("\n" + "="*60)
print("A7: TPR@TNR=95% per benchmark at N=5")
print("="*60)

df_n5 = all_data[all_data['train_limit'] == 5]

# Check if tpr_tnr95 column exists
if 'tpr_tnr95' in df_n5.columns:
    tpr_table = df_n5.groupby(['benchmark', 'model']).agg({
        'tpr_tnr95': ['mean', 'std']
    }).reset_index()
    tpr_table.columns = ['benchmark', 'model', 'tpr_mean', 'tpr_std']

    # Pivot
    tpr_pivot = tpr_table.pivot(index='benchmark', columns='model', values='tpr_mean')
    tpr_pivot.to_csv(OUTPUT_DIR / "A7_tpr_tnr95_n5.csv")
    print("  Saved: A7_tpr_tnr95_n5.csv")
    print(tpr_pivot.round(3))
else:
    print("  tpr_tnr95 column not found")

# =============================================================================
# Table A8: Results at different N (N=2, 10, 20, 50)
# =============================================================================
print("\n" + "="*60)
print("A8: Results at different training sizes")
print("="*60)

for n in [2, 10, 20, 50]:
    df_n = all_data[all_data['train_limit'] == n]

    if len(df_n) == 0:
        print(f"  No data for N={n}")
        continue

    # Aggregate by benchmark and model
    agg = df_n.groupby(['benchmark', 'model']).agg({
        'img_auroc': ['mean', 'std'],
        'img_aupr': ['mean', 'std'],
    }).reset_index()
    agg.columns = ['benchmark', 'model', 'auroc_mean', 'auroc_std', 'aupr_mean', 'aupr_std']

    # Create formatted strings
    agg['auroc'] = agg.apply(lambda r: f"{r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}", axis=1)
    agg['aupr'] = agg.apply(lambda r: f"{r['aupr_mean']:.3f} ± {r['aupr_std']:.3f}", axis=1)

    # Pivot for AUPR (primary metric)
    pivot = agg.pivot(index='benchmark', columns='model', values='aupr')
    pivot.to_csv(OUTPUT_DIR / f"A8_aupr_n{n}.csv")
    print(f"  Saved: A8_aupr_n{n}.csv")

# =============================================================================
# Table A9: Pixel-level metrics (AUsPRO) per benchmark
# =============================================================================
print("\n" + "="*60)
print("A9: Pixel-level AUsPRO per benchmark at N=5")
print("="*60)

auspro_table = df_n5.groupby(['benchmark', 'model']).agg({
    'auspro': ['mean', 'std']
}).reset_index()
auspro_table.columns = ['benchmark', 'model', 'auspro_mean', 'auspro_std']

auspro_pivot = auspro_table.pivot(index='benchmark', columns='model', values='auspro_mean')
auspro_pivot.to_csv(OUTPUT_DIR / "A9_auspro_n5.csv")
print("  Saved: A9_auspro_n5.csv")
print(auspro_pivot.round(3))

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Output directory: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.glob("*.csv")):
    print(f"  - {f.name}")
