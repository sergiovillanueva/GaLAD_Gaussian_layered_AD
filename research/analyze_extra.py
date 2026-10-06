"""Extra analyses from saved per-image scores (no GPU) -> output/extra_analyses.md.

1. Seed dispersion of the main comparison (GaLAD vs AnomalyDINO, DINOv3-L, no mask).
2. Benchmark-balanced means and leave-one-benchmark-out sensitivity.
3. Anomaly prevalence per benchmark (AP of a random detector).
4. MVTec LOCO: logical and structural anomalies separately (each against all normal images).
5. Defect size: per-category gain against median defect area (ground-truth masks).
6. Single global threshold without defects (IMPROVEMENT_PLAN.md section 27, step 4): tau = 0.95 or
   0.99 quantile of the normal test scores of the 27 development categories (categories weighted
   equally, per N and seed), applied unchanged to the 19 external categories. AnomalyDINO with its
   own global tau, raw and normalized by the mean leave-one-out score of its references.

Sources: output/server_run/l_square.csv and scores/l_square (GaLAD and AnomalyDINO, server run);
output/tune/scores_adino (AnomalyDINO with reference leave-one-out scores, same configuration).
Usage: uv run analyze_extra.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import galad_train_all as G

R = Path("output/server_run")
SC = R / "scores/l_square"
SA = Path("output/tune/scores_adino")
OUT = Path("output/extra_analyses.md")
NS, SEEDS = [1, 2, 4, 5], [0, 1, 2, 3, 4]
DEV = ("mvtec_AD", "VisA")
BENCH = ["mvtec_AD", "VisA", "mvtec_loco_AD", "AutoVI", "btad", "GoodsAD"]
rng = np.random.default_rng(0)
lines = []


def say(s=""):
    lines.append(s)


def ci(x):
    x = np.asarray(x)
    bs = [rng.choice(x, len(x)).mean() for _ in range(10000)]
    return f"{x.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"


L = pd.read_csv(R / "l_square.csv")
L["bench"] = L.dataset.str.split("/").str[0]
cats = sorted(L.dataset.unique())
ext = [c for c in cats if not c.startswith(DEV)]
bench_of = {c: c.split("/")[0] for c in cats}
piv = L.pivot_table(index=["dataset", "n_train", "seed"], columns="method", values="img_ap") * 100
piv["d"] = piv.galad - piv.adino
diff_cat = piv.d.groupby("dataset").mean()  # per category: mean over N and seeds

say("# Análisis extra a partir de las puntuaciones guardadas (`analyze_extra.py`)")
say()
say("AP de imagen en puntos, DINOv3-L, 448×448, sin máscara. Diferencia = GaLAD − AnomalyDINO.")

# 1. Seed dispersion
say("\n## 1. Dispersión entre sorteos")
per_seed = piv.d.groupby(["seed", "dataset"]).mean().groupby("seed")
say("Diferencia media sobre N y categorías, calculada con cada sorteo por separado:")
say()
say("| Sorteo | 46 cat. | 19 externas |")
say("|---|---|---|")
for s, x in per_seed:
    x = x.droplevel(0)
    say(f"| {s} | {x.mean():+.2f} | {x[x.index.isin(ext)].mean():+.2f} |")
sd = piv.groupby(["dataset", "n_train"])[["galad", "adino"]].std().groupby("dataset").mean()
say(f"\nDesviación típica entre sorteos de la AP de una categoría (media sobre categorías y N): "
    f"GaLAD {sd.galad.mean():.2f}, AnomalyDINO {sd.adino.mean():.2f} puntos.")
seed_wins = (piv.d.groupby(["dataset", "seed"]).mean() > 0).groupby("dataset").mean()
say(f"Categorías donde GaLAD gana en los 5 sorteos: {(seed_wins == 1).sum()}; en ninguno: {(seed_wins == 0).sum()}.")

# 2. Benchmark balance and leave-one-benchmark-out
say("\n## 2. Peso por benchmark y dejar un benchmark fuera")
by_b = diff_cat.groupby(lambda c: bench_of[c]).mean()
say("Diferencia media por benchmark: " + ", ".join(f"{b} {by_b[b]:+.2f}" for b in BENCH))
say(f"\nMedia con los 6 benchmarks a igual peso: {by_b.mean():+.2f}; "
    f"4 externos a igual peso: {by_b[BENCH[2:]].mean():+.2f}.")
say("\n| Benchmark fuera | 46 cat. sin él | 19 externas sin él |")
say("|---|---|---|")
for b in BENCH:
    keep = [c for c in cats if bench_of[c] != b]
    ke = [c for c in ext if bench_of[c] != b]
    say(f"| {b} | {ci(diff_cat[keep])} | {ci(diff_cat[ke]) if b not in DEV else '='} |")

# 3. Prevalence
say("\n## 3. Prevalencia de anomalías en test (AP de un detector al azar)")
say("\n| Benchmark | Prevalencia | AP GaLAD | AP AnomalyDINO | AUROC GaLAD | AUROC AnomalyDINO |")
say("|---|---|---|---|---|---|")
prev = {}
for c in cats:
    z = np.load(SC / c.replace("/", "__") / "N1_s0_galad.npz")
    prev[c] = z["labels"].mean() * 100
for b in BENCH:
    x = L[L.bench == b].groupby(["method", "dataset"])[["img_ap", "img_auroc"]].mean().groupby("method").mean() * 100
    pb = np.mean([prev[c] for c in cats if bench_of[c] == b])
    say(f"| {b} | {pb:.1f} | {x.img_ap.galad:.1f} | {x.img_ap.adino:.1f} | {x.img_auroc.galad:.1f} | {x.img_auroc.adino:.1f} |")

# 4. LOCO logical vs structural
say("\n## 4. MVTec LOCO: anomalías lógicas y estructurales por separado")
say("Cada tipo frente a todas las imágenes normales; media de 5 categorías y 5 sorteos.")
say("\n| N | AP lógicas GaLAD / AnomalyDINO | AP estructurales GaLAD / AnomalyDINO | AUROC lógicas | AUROC estructurales |")
say("|---|---|---|---|---|")
loco = [c for c in cats if c.startswith("mvtec_loco_AD")]
kinds = {}
for c in loco:
    _, info = G.load_dataset(c)
    kinds[c] = np.array([Path(e["path"]).parent.name for e in info])
for n in NS:
    res = {}
    for m in ("galad", "adino"):
        for kind in ("logical_anomalies", "structural_anomalies"):
            ap, au = [], []
            for c in loco:
                for s in SEEDS:
                    z = np.load(SC / c.replace("/", "__") / f"N{n}_s{s}_{m}.npz")
                    sel = (kinds[c] == "good") | (kinds[c] == kind)
                    y = (kinds[c][sel] == kind).astype(int)
                    ap.append(average_precision_score(y, z["test"][sel]))
                    au.append(roc_auc_score(y, z["test"][sel]))
            res[m, kind] = (100 * np.mean(ap), 100 * np.mean(au))
    lo, st = "logical_anomalies", "structural_anomalies"
    say(f"| {n} | {res['galad', lo][0]:.1f} / {res['adino', lo][0]:.1f} | {res['galad', st][0]:.1f} / {res['adino', st][0]:.1f} | "
        f"{res['galad', lo][1]:.1f} / {res['adino', lo][1]:.1f} | {res['galad', st][1]:.1f} / {res['adino', st][1]:.1f} |")

# 5. Defect size
say("\n## 5. Tamaño del defecto")
area = {}
for c in cats:
    _, info = G.load_dataset(c)
    fr = []
    for e in info:
        if e["label"] == 1:
            m = G.load_mask(e)
            if m is not None and (m > 0).any():
                fr.append((m > 0).mean())
    area[c] = np.median(fr) * 100 if fr else np.nan
A = pd.Series(area).dropna()
rho, p = spearmanr(A, diff_cat[A.index])
say(f"Área mediana del defecto por categoría (% de la imagen) frente a la diferencia de AP: "
    f"Spearman {rho:+.2f} (p = {p:.3f}, {len(A)} categorías).")
for b in BENCH:
    cb = [c for c in A.index if bench_of[c] == b]
    if len(cb) >= 5:
        r, pp = spearmanr(A[cb], diff_cat[cb])
        say(f"- dentro de {b} ({len(cb)} cat.): Spearman {r:+.2f} (p = {pp:.2f})")
q = A.quantile([1 / 3, 2 / 3]).values
for lab, sel in (("defectos pequeños", A <= q[0]), ("medianos", (A > q[0]) & (A <= q[1])), ("grandes", A > q[1])):
    say(f"- tercio de {lab} (área ≤ {A[sel].max():.1f} %): diferencia {ci(diff_cat[A[sel].index])}")

# 6. Single threshold
say("\n## 6. Umbral único sin defectos (regla de §27, paso 4)")
say("τ = cuantil de las puntuaciones de imágenes normales de test de las 27 categorías de desarrollo "
    "(cada categoría con el mismo peso), por N y sorteo; se aplica sin cambios a las 19 externas.")


def wquantile(vals, weights, q):
    o = np.argsort(vals)
    cw = np.cumsum(weights[o])
    return vals[o][np.searchsorted(cw, q * cw[-1])]


def load_scores(method, c, n, s):
    if method == "galad":
        z = np.load(SC / c.replace("/", "__") / f"N{n}_s{s}_galad.npz")
        return z["test"].astype(float), z["labels"]
    z = np.load(SA / c.replace("/", "__") / f"N{n}_s{s}.npz")
    x = z["test"].astype(float)
    if method == "adino_loo":
        x = x / z["ref_loo"].mean()
    return x, z["labels"]


say("\n| Método | Objetivo FPR | N | FPR externas: mediana / p90 / máx | Recall medio externas |")
say("|---|---|---|---|---|")
summary = {}
for method, name in (("galad", "GaLAD (sin normalizar)"), ("adino", "AnomalyDINO (sin normalizar)"),
                     ("adino_loo", "AnomalyDINO / media dejar-una-fuera")):
    for q in (0.95, 0.99):
        for n in NS:
            fprs, recs = [], []
            for s in SEEDS:
                v, w = [], []
                for c in cats:
                    if c.startswith(DEV):
                        x, y = load_scores(method, c, n, s)
                        v.append(x[y == 0])
                        w.append(np.full((y == 0).sum(), 1 / (y == 0).sum()))
                tau = wquantile(np.concatenate(v), np.concatenate(w), q)
                for c in ext:
                    x, y = load_scores(method, c, n, s)
                    fprs.append(((x > tau) & (y == 0)).sum() / (y == 0).sum())
                    recs.append(((x > tau) & (y == 1)).sum() / (y == 1).sum())
            f = np.array(fprs).reshape(len(SEEDS), len(ext)).mean(0) * 100
            r = np.array(recs).reshape(len(SEEDS), len(ext)).mean(0) * 100
            summary[method, q, n] = (np.median(f), np.percentile(f, 90), f.max(), r.mean())
            say(f"| {name} | {100 * (1 - q):.0f} % | {n} | {np.median(f):.1f} / {np.percentile(f, 90):.1f} / "
                f"{f.max():.1f} | {r.mean():.1f} |")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
