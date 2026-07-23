"""
Analyze the cross-backbone scoring control (Table 12).

Two canonical sources, all 46 categories, N=10, 5 seeds:
    - results/backbone_ablation.csv     -> GMM K=3 for all four backbones
    - results/crossbackbone_control.csv -> K1 and kNN (k=5) for all four backbones

For each backbone, aggregates over 46 categories (mean over the 5 seeds first)
and reports image-level AUROC / AUPR for GMM (K=3) vs K=1 vs kNN.

Per backbone: paired Wilcoxon on the 46 per-category mean AUPR scores
(GMM vs K1, GMM vs kNN) and mean paired AUPR delta. Holm-Bonferroni is applied
across the full family of tests (4 backbones x comparisons). Cliff's delta as
effect size, with its qualitative label.

Also prints, per backbone, which scoring layer is best on AUROC and AUPR, so
the LaTeX table can bold the true winner per row.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cliff_label(d):
    ad = abs(d)
    return ("negligible" if ad < 0.147 else "small" if ad < 0.33
            else "medium" if ad < 0.474 else "large")


def load_k3():
    df = pd.read_csv("results/backbone_ablation.csv", sep=";")
    df = df[df["train_limit"] == 10].copy()
    df["method"] = "k3"
    return df[["backbone", "dataset", "seed", "method", "img_auroc", "img_aupr"]]


def load_controls():
    df = pd.read_csv("results/crossbackbone_control.csv", sep=";")
    df = df[df["train_limit"] == 10].copy()
    return df[["backbone", "dataset", "seed", "method", "img_auroc", "img_aupr"]]


def holm(pvals):
    """Return Holm-adjusted reject decisions at alpha=0.05 for a list of
    (key, p) pairs. Returns dict key -> (adjusted_threshold, reject)."""
    m = len(pvals)
    order = sorted(pvals, key=lambda kv: kv[1])
    out = {}
    reject_so_far = True
    for i, (key, p) in enumerate(order):
        thr = 0.05 / (m - i)
        rej = reject_so_far and (p < thr)
        reject_so_far = rej
        out[key] = (thr, rej)
    return out


def main():
    df = pd.concat([load_k3(), load_controls()], ignore_index=True)
    print("Combined rows:", len(df))
    print(df.groupby(["backbone", "method"]).size())

    per_cat = (df.groupby(["backbone", "dataset", "method"])
                 [["img_auroc", "img_aupr"]].mean().reset_index())

    BACKBONES = ["dinov3_vitl16", "dinov2_vitl14", "clip_vitl14", "siglip_vitl16"]
    DISP = {"dinov3_vitl16": "DINOv3 ViT-L/16", "dinov2_vitl14": "DINOv2 ViT-L/14",
            "clip_vitl14": "CLIP ViT-L/14", "siglip_vitl16": "SigLIP ViT-L/16"}

    # collect all p-values for the family-wise Holm correction
    pvals = []
    stats = {}
    for b in BACKBONES:
        sub = per_cat[per_cat.backbone == b]
        if sub.empty:
            continue
        pa = sub.pivot(index="dataset", columns="method", values="img_aupr")
        pr = sub.pivot(index="dataset", columns="method", values="img_auroc")
        methods = ["k3", "k1", "knn"] + (["pcares"] if "pcares" in pa.columns else [])
        if not {"k3", "k1", "knn"}.issubset(pa.columns):
            print(f"  [skip {b}] missing {set(['k3','k1','knn'])-set(pa.columns)}")
            continue
        _, p1 = wilcoxon(pa["k3"], pa["k1"])
        _, p2 = wilcoxon(pa["k3"], pa["knn"])
        stats[b] = {
            "auroc": {m: pr[m].mean() for m in methods},
            "aupr": {m: pa[m].mean() for m in methods},
            "p_k1": p1, "p_knn": p2,
            "d_k1": cliffs_delta(pa["k3"], pa["k1"]),
            "d_knn": cliffs_delta(pa["k3"], pa["knn"]),
            "delta_aupr_k1": (pa["k3"] - pa["k1"]).mean(),
            "delta_aupr_knn": (pa["k3"] - pa["knn"]).mean(),
        }
        pvals += [((b, "k1"), p1), ((b, "knn"), p2)]
        if "pcares" in pa.columns:
            _, p3 = wilcoxon(pa["k3"], pa["pcares"])
            stats[b]["p_pcares"] = p3
            stats[b]["delta_aupr_pcares"] = (pa["k3"] - pa["pcares"]).mean()
            pvals += [((b, "pcares"), p3)]

    holm_res = holm(pvals)

    print("\n=== Per-backbone summary (Table 12) ===")
    rows = []
    for b in BACKBONES:
        if b not in stats:
            continue
        s = stats[b]
        best_auroc = max(s["auroc"], key=s["auroc"].get)
        best_aupr = max(s["aupr"], key=s["aupr"].get)
        print(f"\n{DISP[b]}:")
        print(f"  AUROC  K3={s['auroc']['k3']:.3f}  K1={s['auroc']['k1']:.3f}  "
              f"kNN={s['auroc']['knn']:.3f}   best={best_auroc}")
        print(f"  AUPR   K3={s['aupr']['k3']:.3f}  K1={s['aupr']['k1']:.3f}  "
              f"kNN={s['aupr']['knn']:.3f}   best={best_aupr}")
        h1 = holm_res[(b, "k1")]; h2 = holm_res[(b, "knn")]
        print(f"  K3 vs K1 : p={s['p_k1']:.2e}  Holm thr={h1[0]:.4f} "
              f"reject={h1[1]}  dAUPR={s['delta_aupr_k1']:+.3f}  "
              f"Cliff={s['d_k1']:+.3f} ({cliff_label(s['d_k1'])})")
        print(f"  K3 vs kNN: p={s['p_knn']:.2e}  Holm thr={h2[0]:.4f} "
              f"reject={h2[1]}  dAUPR={s['delta_aupr_knn']:+.3f}  "
              f"Cliff={s['d_knn']:+.3f} ({cliff_label(s['d_knn'])})")
        row = {
            "backbone": DISP[b],
            "auroc_k3": s["auroc"]["k3"], "auroc_k1": s["auroc"]["k1"], "auroc_knn": s["auroc"]["knn"],
            "aupr_k3": s["aupr"]["k3"], "aupr_k1": s["aupr"]["k1"], "aupr_knn": s["aupr"]["knn"],
            "best_auroc": best_auroc, "best_aupr": best_aupr,
            "p_k3_vs_k1": s["p_k1"], "holm_reject_k1": holm_res[(b, "k1")][1],
            "p_k3_vs_knn": s["p_knn"], "holm_reject_knn": holm_res[(b, "knn")][1],
            "delta_aupr_k1": s["delta_aupr_k1"], "delta_aupr_knn": s["delta_aupr_knn"],
            "cliff_k1": s["d_k1"], "cliff_knn": s["d_knn"],
        }
        if "p_pcares" in s:
            print(f"  K3 vs Sub: p={s['p_pcares']:.2e}  dAUPR={s['delta_aupr_pcares']:+.3f}  "
                  f"(subspace AUROC={s['auroc']['pcares']:.3f} AUPR={s['aupr']['pcares']:.3f})")
            row.update({"auroc_pcares": s["auroc"]["pcares"], "aupr_pcares": s["aupr"]["pcares"],
                        "p_k3_vs_pcares": s["p_pcares"], "delta_aupr_pcares": s["delta_aupr_pcares"]})
        rows.append(row)

    n_sig = sum(1 for _, (_, r) in holm_res.items() if r)
    print(f"\nHolm over the family of {len(pvals)} tests: "
          f"{n_sig} remain significant at alpha=0.05.")

    out = Path("results/tables_latex")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "crossbackbone_control_summary.csv", index=False, sep=";")
    print(f"Saved {out/'crossbackbone_control_summary.csv'}")


if __name__ == "__main__":
    main()
