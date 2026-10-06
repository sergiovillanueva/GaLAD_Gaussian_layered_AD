"""Overnight screening of GaLAD modifications (see IMPROVEMENT_PLAN.md, section 12).

Phases (each resumable):
  cache    Extract and store DINOv3 test features (original + horizontal flip) per category.
  screen   Fit variants on GPU (PCA + full-covariance GMM in torch, same logic as
           GaLADConfig) and write one row per (dataset, variant, N, seed).

Each fit configuration emits rows for several post-processing options (pooling,
test-time flip) from the same fitted model, so those cost almost nothing extra.

Usage:
  uv run tune_experiments.py --phase cache  --set dev
  uv run tune_experiments.py --phase screen --set dev --fits base none d8 l2 agg3
  uv run tune_experiments.py --phase screen --set dev --plan
"""

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--phase", required=True, choices=["cache", "screen", "loo"])
p.add_argument("--set", default="dev", choices=["dev", "ext"])
p.add_argument("--datasets", nargs="*", default=None)
p.add_argument("--fits", nargs="*", default=None, help="Fit configs to run (default: all).")
p.add_argument("--ns", nargs="*", type=int, default=[1, 2, 4])
p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--out", default="screen", help="CSV name under output/tune/.")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--plan", action="store_true")
p.add_argument("--no-tta", action="store_true", help="Skip test-time flip rows (halves test scoring).")
p.add_argument("--gmm-device", default="cuda", choices=["cuda", "cpu"],
               help="Where PCA/GMM run (feature extraction always uses the GPU).")
args = p.parse_args()

# Reuse loaders, backbone, augmentations and metrics from improve_experiments.py.
_argv, sys.argv = sys.argv, [sys.argv[0], "--stage", "augment", "--threads", str(args.threads)]
import improve_experiments as E  # noqa: E402
sys.argv = _argv
G = E.G

import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

DEV = E.LIT_CATS  # 15 MVTec AD + 12 VisA
EXT = ["AutoVI/engine_wiring", "AutoVI/pipe_clip", "AutoVI/tank_screw", "AutoVI/underbody_pipes",
       "AutoVI/underbody_screw", "mvtec_loco_AD/breakfast_box", "mvtec_loco_AD/juice_bottle",
       "mvtec_loco_AD/pushpins", "mvtec_loco_AD/screw_bag", "mvtec_loco_AD/splicing_connectors",
       "btad/01", "btad/02", "btad/03", "GoodsAD/cigarette_box", "GoodsAD/drink_bottle",
       "GoodsAD/drink_can", "GoodsAD/food_bottle", "GoodsAD/food_box", "GoodsAD/food_package"]
LAYERS = (-6, -12)
CACHE = Path("output/cache_feats")
OUT = Path("output/tune")
OUT.mkdir(parents=True, exist_ok=True)
DEV_T = torch.device("cuda" if (torch.cuda.is_available() and args.gmm_device == "cuda") else "cpu")

# =============================================================================
# Variants. BASE = GaLAD-D4 as frozen in section 8 of the plan.
# =============================================================================

BASE = dict(aug="d4", l2=False, agg=False, dim=256, skip=2, reg=1e-2, reg_rel=None,
            cov="full", whiten=False, layers=LAYERS, k=3, ninit=1)
# Fit configurations. Post-processing candidates (pooling, test-time flip) are read from the
# 'base' fit, so they cost no extra fitting.
FITS = {
    "base": {},                          # S1 anchor: GaLAD-D4 (frozen in section 8)
    "none": dict(aug="none"),            # control: plain GaLAD, same implementation
    "rot8": dict(aug="rot8"),            # S5: AnomalyDINO's eight 45-degree rotations
    "dim128": dict(dim=128),             # S6
    "dim64": dict(dim=64),               # S7
    "tied": dict(cov="tied"),            # S8: one covariance shared by the K components
    "diag": dict(cov="diag"),            # S9: diagonal covariances
    "l2rel": dict(l2=True, reg_rel=1e-3),  # S10: unit-norm patches + relative ridge
    "l6": dict(layers=(-6,)),            # S11: layer -6 only
    "agg3": dict(agg=True),              # S12: 3x3 neighbourhood average of patch features
    "ni3": dict(ninit=3),                # section 19: EM restarts, keep the highest likelihood
}
# S2 base|p99, S3 base|top1pct, S4 base|p95|tta come from the 'base' fit.
CANDIDATES = ["base|p95|notta", "base|p99|notta", "base|top1pct|notta", "base|p95|tta",
              "rot8|p95|notta", "dim128|p95|notta", "dim64|p95|notta", "tied|p95|notta",
              "diag|p95|notta", "l2rel|p95|notta", "l6|p95|notta", "agg3|p95|notta"]
POOLS = ["p95", "p99", "top1pct"]
TTAS = [False] if args.no_tta else [False, True]
FIELDS = ["timestamp", "dataset", "fit", "pool", "tta", "variant", "n_train", "seed",
          "img_auroc", "img_ap", "auspro", "k_used", "seconds"]


def cfg_of(fit):
    """'l6+l2rel' merges the listed modifications on top of BASE."""
    c = dict(BASE)
    for part in fit.split("+"):
        c.update(FITS[part])
    return c


def variant_name(fit, pool, tta):
    return f"{fit}|{pool}|{'tta' if tta else 'notta'}"


def datasets():
    return args.datasets or (DEV if args.set == "dev" else EXT)


# =============================================================================
# Feature cache
# =============================================================================

def cdir(ds):
    return CACHE / ds.replace("/", "__")


def cache_category(bb, ds):
    d = cdir(ds)
    if (d / "done").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    _, test_info = G.load_dataset(ds)
    paths = [t["path"] for t in test_info]
    for tag, flip in (("orig", False), ("flip", True)):
        feats = {l: [] for l in LAYERS}
        for i in range(0, len(paths), 32):
            imgs = [Image.open(q).convert("RGB") for q in paths[i:i + 32]]
            if flip:
                imgs = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]
            f = bb.extract(imgs, list(LAYERS))
            for l in LAYERS:
                feats[l].append(f[l].half())
        for l in LAYERS:
            np.save(d / f"test_{tag}_L{l}.npy", torch.cat(feats[l]).numpy())
    (d / "done").write_text(f"{len(paths)} images\n")
    E.log(f"  cached {ds}: {len(paths)} test images")


# =============================================================================
# GPU PCA + GMM (mirrors GaLADConfig.fit: subsample >100k, PCA dim+skip, drop skip,
# K = min(k, max(2, n // (dim + 10))), full covariance, reg_covar, tol 1e-3, 200 iters)
# =============================================================================

def kmeans_labels(X, k, gen, iters=100):
    n = X.shape[0]
    centers = X[torch.randint(n, (1,), generator=gen, device=X.device)]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        i = torch.multinomial(d2 / d2.sum(), 1, generator=gen)
        centers = torch.cat([centers, X[i]])
        d2 = torch.minimum(d2, ((X - X[i]) ** 2).sum(1))
    for _ in range(iters):
        lab = torch.cdist(X, centers).argmin(1)
        new = torch.stack([X[lab == j].mean(0) if (lab == j).any() else centers[j] for j in range(k)])
        if ((new - centers) ** 2).sum() < 1e-8:
            break
        centers = new
    return lab


class TorchGMM:
    def __init__(self, k, reg, seed, cov="full", tol=1e-3, max_iter=200, ninit=1):
        self.k, self.reg, self.seed, self.cov, self.tol, self.max_iter = k, reg, seed, cov, tol, max_iter
        self.ninit = ninit

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(0) + 10 * torch.finfo(X.dtype).eps
        self.means = (resp.T @ X) / nk[:, None]
        covs = []
        for j in range(self.k):
            diff = X - self.means[j]
            covs.append((resp[:, j:j + 1] * diff).T @ diff / nk[j])
        cov = torch.stack(covs)
        if self.cov == "tied":  # as sklearn: weighted average of the component covariances
            cov = ((nk[:, None, None] * cov).sum(0) / nk.sum()).expand(self.k, d, d).clone()
        elif self.cov == "diag":
            cov = torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
        cov = cov + self.reg * torch.eye(d, dtype=X.dtype, device=X.device)
        # fp64 Cholesky of k x 256 x 256 is far faster on CPU than on consumer GPUs.
        self.chol = torch.linalg.cholesky(cov.cpu()).to(X.device)
        self.log_w = torch.log(nk / n)

    def _log_prob(self, X):
        d = X.shape[1]
        out = []
        for j in range(self.k):
            y = torch.linalg.solve_triangular(self.chol[j], (X - self.means[j]).T, upper=False)
            logdet = 2 * torch.log(torch.diagonal(self.chol[j])).sum()
            out.append(-0.5 * (d * np.log(2 * np.pi) + logdet + (y ** 2).sum(0)) + self.log_w[j])
        return torch.stack(out, 1)

    def _fit_once(self, X, seed):
        gen = torch.Generator(device=X.device).manual_seed(seed)
        lab = kmeans_labels(X, self.k, gen)
        self._m_step(X, F.one_hot(lab, self.k).to(X.dtype))
        lb = -np.inf
        for _ in range(self.max_iter):
            lp = self._log_prob(X)
            norm = torch.logsumexp(lp, 1)
            self._m_step(X, torch.exp(lp - norm[:, None]))
            new = norm.mean().item()
            if abs(new - lb) < self.tol:
                break
            lb = new
        return new

    def fit(self, X):
        """ninit=1 is the original behaviour (same seed, same result). ninit>1: as sklearn's n_init,
        restart from different k-means++ seeds and keep the parameters with the highest
        training log-likelihood (no labels involved)."""
        best = None
        for i in range(self.ninit):
            ll = self._fit_once(X, self.seed + 7919 * i)
            if best is None or ll > best[0]:
                best = (ll, self.means, self.chol, self.log_w)
        _, self.means, self.chol, self.log_w = best
        return self

    def score_samples(self, X, chunk=60000):
        return torch.cat([torch.logsumexp(self._log_prob(X[i:i + chunk]), 1) for i in range(0, X.shape[0], chunk)])


def preprocess(f, cfg, grid):
    """f: [n_img, n_patch, D] on GPU (any float dtype) -> float32 [n_img * n_patch, D]."""
    f = f.float()
    if cfg["agg"]:
        n, P, D = f.shape
        g = f.reshape(n, grid, grid, D).permute(0, 3, 1, 2)
        g = F.avg_pool2d(g, 3, stride=1, padding=1, count_include_pad=False)
        f = g.permute(0, 2, 3, 1).reshape(n, P, D)
    if cfg["l2"]:
        f = F.normalize(f, dim=-1)
    return f.reshape(-1, f.shape[-1])


class LayerModel:
    def __init__(self, X, cfg, seed):
        if X.shape[0] > 100000:  # same subsampling as GaLADConfig.fit
            np.random.seed(seed)
            X = X[torch.from_numpy(np.random.choice(X.shape[0], 100000, replace=False)).to(X.device)]
        X = X.double()
        self.mean = X.mean(0)
        Xc = X - self.mean
        evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
        order = torch.argsort(evals, descending=True)[: cfg["dim"] + cfg["skip"]]
        self.comp = evecs[:, order][:, cfg["skip"]:]
        self.scale = torch.sqrt(evals[order][cfg["skip"]:].clamp_min(1e-12)) if cfg["whiten"] else None
        Z = self.project(X)
        del X, Xc  # ~1.6 GB at 100k x 1024 fp64; not needed during EM
        k = min(cfg["k"], max(2, Z.shape[0] // (cfg["dim"] + 10)))
        self.k = k
        reg = cfg["reg"] if cfg["reg_rel"] is None else cfg["reg_rel"] * Z.var(0).mean().item()
        self.gmm = TorchGMM(k, reg, seed, cov=cfg["cov"], ninit=cfg["ninit"]).fit(Z)

    def project(self, X):
        Z = (X.double() - self.mean) @ self.comp
        return Z / self.scale if self.scale is not None else Z

    def nll(self, X):
        return -self.gmm.score_samples(self.project(X))


# =============================================================================
# Screening
# =============================================================================

def out_csv():
    return OUT / f"{args.out}.csv"


def done_keys():
    if not out_csv().exists():
        return set()
    with open(out_csv(), newline="") as f:
        return {(r["dataset"], r["variant"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}


def save_rows(rows):
    new = not out_csv().exists()
    with open(out_csv(), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def ref_images(img_paths, aug):
    imgs = [Image.open(q).convert("RGB").resize((E.RES, E.RES), Image.BICUBIC) for q in img_paths]
    if aug == "none":
        return imgs
    if aug == "d4":
        return E.augment(imgs, E.dihedral)
    if aug == "rot8":
        return E.augment(imgs, E.rot45)
    if aug == "d8":
        return E.augment(imgs, lambda im: [r for x in E.rot45(im) for r in E.hflip(x)])
    raise ValueError(aug)


def pool_scores(maps, how):
    return E.pool(maps, how)


def run_category(bb, ds, fits, todo):
    train_all, test_info = G.load_dataset(ds)
    labels = np.array([t["label"] for t in test_info])
    d = cdir(ds)
    # Memory-mapped: RAM and GPU hold only the chunk being scored.
    test = {(tag, l): np.load(d / f"test_{tag}_L{l}.npy", mmap_mode="r")
            for tag in ("orig", "flip") for l in LAYERS}
    grid = bb.grid
    for n in args.ns:
        for seed in args.seeds:
            need = [f for f in fits if any((variant_name(f, pl, t), n, seed) in todo
                                           for pl in POOLS for t in TTAS)]
            if not need:
                continue
            refs = E.sample_train(train_all, n, seed)
            by_aug = {}
            for f in need:
                by_aug.setdefault(cfg_of(f)["aug"], []).append(f)
            for aug, group in by_aug.items():
                tr = bb.extract(ref_images(refs, aug), list(LAYERS))
                for fit in group:
                    t0 = time.time()
                    cfg = cfg_of(fit)
                    maps = {False: [], True: []}
                    ks = []
                    for l in cfg["layers"]:
                        m = LayerModel(preprocess(tr[l].to(DEV_T), cfg, grid), cfg, seed)
                        ks.append(m.k)
                        for tta in TTAS:
                            tag = "flip" if tta else "orig"
                            x = test[(tag, l)]
                            nll = torch.cat([m.nll(preprocess(torch.from_numpy(np.ascontiguousarray(x[i:i + 32])).to(DEV_T), cfg, grid))
                                             for i in range(0, x.shape[0], 32)])
                            a = nll.reshape(x.shape[0], grid, grid).float().cpu().numpy()
                            if tta:
                                a = np.ascontiguousarray(a[:, :, ::-1])  # back to original orientation
                            maps[tta].append(a)
                    rows = []
                    orig = E.smooth({i: v for i, v in enumerate(maps[False])}, 1)
                    flipb = E.smooth({i: v for i, v in enumerate(maps[True])}, 1)
                    for tta in TTAS:
                        if tta:
                            mm = {i: (orig[i] + flipb[i]) / 2 for i in orig}
                        else:
                            mm = orig
                        pro = None if tta else E.auspro_448(E.fused_map(mm), test_info)
                        for pl in POOLS:
                            v = variant_name(fit, pl, tta)
                            if (v, n, seed) not in todo:
                                continue
                            au, ap = E.metrics(labels, E.image_scores(mm, pl))
                            rows.append(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds,
                                             fit=fit, pool=pl, tta=int(tta), variant=v, n_train=n,
                                             seed=seed, img_auroc=f"{au:.4f}", img_ap=f"{ap:.4f}",
                                             auspro=E.fmt(pro), k_used="/".join(map(str, ks)),
                                             seconds=f"{time.time() - t0:.1f}"))
                    save_rows(rows)
                    del m
                    torch.cuda.empty_cache()
                del tr
            E.log(f"  {ds} N={n} seed {seed}: {len(need)} fits")
    del test
    torch.cuda.empty_cache()


# =============================================================================
# Label-free augmentation choice (IMPROVEMENT_PLAN.md, section 14)
# =============================================================================

LOO_FIELDS = ["timestamp", "dataset", "n_train", "seed", "fpr_none", "fpr_rot8", "choice"]


def loo_csv():
    return OUT / "loo.csv"


def loo_done():
    if not loo_csv().exists():
        return set()
    with open(loo_csv(), newline="") as f:
        return {(r["dataset"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}


def smoothed_maps(model, feats, cfg, grid):
    nll = model.nll(preprocess(feats.to(DEV_T), cfg, grid))
    a = nll.reshape(feats.shape[0], grid, grid).float().cpu().numpy()
    return np.stack([E.gaussian_filter(x, sigma=1) for x in a])


def heldout_fpr(bb, fit_refs, held, fit, seed):
    """Fraction of the held-out normal image's patches above the model's own 95th percentile
    on its (non-augmented) fitting references, averaged over layers."""
    cfg = cfg_of(fit)
    grid = bb.grid
    orig = bb.extract(ref_images(fit_refs, "none"), list(LAYERS))
    tr = orig if cfg["aug"] == "none" else bb.extract(ref_images(fit_refs, cfg["aug"]), list(LAYERS))
    ho = bb.extract(ref_images([held], "none"), list(LAYERS))
    rates = []
    for l in cfg["layers"]:
        m = LayerModel(preprocess(tr[l].to(DEV_T), cfg, grid), cfg, seed)
        tau = np.percentile(smoothed_maps(m, orig[l], cfg, grid), 95)
        rates.append(float((smoothed_maps(m, ho[l], cfg, grid) > tau).mean()))
    return float(np.mean(rates))


def run_loo(bb, ds, done):
    train_all, _ = G.load_dataset(ds)
    for n in args.ns:
        if n < 2:
            continue
        for seed in args.seeds:
            if (ds, n, seed) in done:
                continue
            refs = E.sample_train(train_all, n, seed)
            fpr = {"none": [], "rot8": []}
            for i in range(n if n <= 5 else 2):
                fit_refs, held = refs[:i] + refs[i + 1:], refs[i]
                for fit in fpr:
                    fpr[fit].append(heldout_fpr(bb, fit_refs, held, fit, seed))
            fn, fr = float(np.mean(fpr["none"])), float(np.mean(fpr["rot8"]))
            row = dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, n_train=n, seed=seed,
                       fpr_none=f"{fn:.5f}", fpr_rot8=f"{fr:.5f}", choice="rot8" if fr < fn else "none")
            new = not loo_csv().exists()
            with open(loo_csv(), "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOO_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            E.log(f"  {ds} N={n} seed {seed}: fpr none {fn:.4f} rot8 {fr:.4f} -> {row['choice']}")
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()


def main():
    cats = datasets()
    if args.phase == "cache":
        bb = E.Backbone("dinov3l")
        for i, ds in enumerate(cats):
            E.log(f"[cache {i + 1}/{len(cats)}] {ds}")
            try:
                cache_category(bb, ds)
            except Exception as e:
                E.log(f"ERROR cache {ds}: {type(e).__name__}: {e}")
        E.log("Cache phase finished.")
        return

    if args.phase == "loo":
        done = loo_done()
        left = [ds for ds in cats if any((ds, n, s) not in done for n in args.ns if n >= 2 for s in args.seeds)]
        E.log(f"LOO phase: {len(cats) - len(left)}/{len(cats)} categories done, N {args.ns}")
        if args.plan or not left:
            return
        bb = E.Backbone("dinov3l")
        for i, ds in enumerate(left):
            E.log(f"[loo {i + 1}/{len(left)}] {ds}")
            try:
                run_loo(bb, ds, done)
            except Exception as e:
                E.log(f"ERROR loo {ds}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
        E.log("LOO phase finished.")
        return

    fits = list(dict.fromkeys(args.fits or list(FITS)))  # dedupe, keep order
    done = done_keys()
    todo = {(variant_name(f, pl, t), n, s) for f in fits for pl in POOLS for t in TTAS
            for n in args.ns for s in args.seeds}
    left = {ds: {k for k in todo if (ds, *k) not in done} for ds in cats}
    total = len(cats) * len(todo)
    E.log(f"Screen {args.out}: fits {fits}, N {args.ns}, seeds {args.seeds}: "
          f"{total - sum(map(len, left.values()))}/{total} rows done")
    if args.plan:
        for ds in cats:
            if left[ds]:
                E.log(f"  pending {ds}: {len(left[ds])} rows" + ("" if (cdir(ds) / "done").exists() else " (NO CACHE)"))
        return
    bb = E.Backbone("dinov3l")
    for i, ds in enumerate(cats):
        if not left[ds]:
            continue
        if not (cdir(ds) / "done").exists():
            E.log(f"SKIP {ds}: no feature cache")
            continue
        E.log(f"[{i + 1}/{len(cats)}] {ds}")
        try:
            run_category(bb, ds, fits, left[ds])
        except Exception as e:
            E.log(f"ERROR {ds}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    E.log("Screen phase finished.")


if __name__ == "__main__":
    main()
