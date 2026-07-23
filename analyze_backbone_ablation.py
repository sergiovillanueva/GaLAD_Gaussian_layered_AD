"""
Analyze the backbone-generality ablation.

Reads output/backbone_ablation.csv and produces a per-backbone summary
(mean over the 8 ablation categories and 5 seeds at N=10), plus a per-category
breakdown. Writes a LaTeX-ready CSV and prints a markdown table.

Usage: uv run analyze_backbone_ablation.py
"""
from pathlib import Path
import pandas as pd

CSV = Path("results/backbone_ablation.csv")
OUT = Path("results/tables_latex/backbone_ablation_summary.csv")

ORDER = ["dinov3_vitl16", "dinov2_vitl14", "clip_vitl14", "siglip_vitl16"]
DISPLAY = {
    "dinov3_vitl16": "DINOv3 ViT-L/16",
    "dinov2_vitl14": "DINOv2 ViT-L/14",
    "clip_vitl14": "CLIP ViT-L/14",
    "siglip_vitl16": "SigLIP ViT-L/16",
}


def main():
    df = pd.read_csv(CSV, sep=";")
    df = df[df["train_limit"] == 10].copy()

    # Per-backbone mean over the 8 categories (after averaging seeds per category)
    per_cat = df.groupby(["dataset", "backbone"])[["img_auroc", "img_aupr"]].mean().reset_index()
    summary = per_cat.groupby("backbone")[["img_auroc", "img_aupr"]].agg(["mean", "std"])
    summary.columns = ["auroc_mean", "auroc_std", "aupr_mean", "aupr_std"]
    summary = summary.reindex(ORDER)

    res = {"dinov3_vitl16": 448, "dinov2_vitl14": 448, "clip_vitl14": 224, "siglip_vitl16": 256}

    print(f"Backbone-generality ablation ({per_cat['dataset'].nunique()} categories, "
          f"5 seeds, N=10)\n")
    print(f"{'Backbone':<18} {'Res':>4} {'AUROC':>14} {'AUPR':>14}")
    rows = []
    for b in ORDER:
        r = summary.loc[b]
        print(f"{DISPLAY[b]:<18} {res[b]:>4} "
              f"{r['auroc_mean']:.3f}+/-{r['auroc_std']:.3f}   "
              f"{r['aupr_mean']:.3f}+/-{r['aupr_std']:.3f}")
        rows.append({
            "backbone": DISPLAY[b], "resolution": res[b],
            "auroc_mean": round(r["auroc_mean"], 4), "auroc_std": round(r["auroc_std"], 4),
            "aupr_mean": round(r["aupr_mean"], 4), "aupr_std": round(r["aupr_std"], 4),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, sep=";")
    print(f"\nSaved {OUT}")

    print("\nPer-category AUROC (mean over seeds):")
    piv = per_cat.pivot(index="dataset", columns="backbone", values="img_auroc")[ORDER]
    piv.columns = [DISPLAY[c] for c in piv.columns]
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
