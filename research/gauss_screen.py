"""Gaussian-modelling ideas from the external LLM round (IMPROVEMENT_PLAN.md, section 26).

Rot8 references, cached DINOv3 test features, PCA + GMM in fp64 on GPU (as tune_experiments.py).
Fit variants:
  k3          current GaLAD+rot8 (K=3, reg_covar 1e-2); must reproduce screen.csv rot8|p95|notta
  k3s{kappa}  shrinkage toward the pooled covariance, which is diag(PCA eigenvalues) in PCA
              coordinates: S_k = (n_k C_k + kappa diag(L)) / (n_k + kappa) + reg I (a shrinkage
              estimator; as an inverse-Wishart MAP it implies an improper prior for kappa <= 2d+1).
              EM stops on the unpenalised likelihood change, as for the anchor.
  k{K}s{kappa} more components with shrinkage
  k3v         K=3, reference patches of the 45-degree rotations lying in reflected borders removed
  k{K}lw      Ledoit-Wolf shrinkage of each component covariance toward a scaled identity (mu I, mu =
              trace / d), intensity from the Ledoit-Wolf formula with responsibility weights
  <fit>_<aug> reference augmentation other than rot8: none | rot4 (0/90/180/270) | small (0, +-15,
              +-30 degrees) | rot16 (22.5-degree steps); same bilinear rotation with reflected borders
  <fit>_d{D}  PCA to D dimensions (default 256);  <fit>_skip{S}: drop the first S components (default 2)
  <fit>_diag  diagonal covariances (each component keeps only the diagonal of its shrunk covariance)
  <fit>_ppca  probabilistic-PCA completion: adds to every Mahalanobis distance the residual outside the
              top D+S principal directions divided by the mean of the remaining eigenvalues, and uses
              the full dimension minus S as chi-square degrees of freedom (the part SubspaceAD scores)
--post adds, for the chi2 score, the inherited post-processing variants on the same fitted model:
  chi2_sig0/chi2_sig2 (smoothing sigma 0 or 2 patches), chi2_max/chi2_p99 (pooling), chi2_L6/chi2_L12
  (one layer only); the default chi2 row is sigma 1, p95, mean of the two layers.
--post2 adds patch scores that use more of the mixture (round 5 ideas), same smoothing/pooling:
  chi2_wtail  -log sum_k pi_k P(chi2 >= d_k^2) (mixture-weighted tail)
  maha_det    min_k (d_k^2 + log|S_k| - mean_k log|S_k|) (component volume), raw like maha
  ent         posterior entropy of the component assignment
--save-maps stores the raw (unsmoothed) chi2 patch maps per layer, test and unrotated references, in
  output/tune/maps/<dataset>/N{n}_s{seed}_{fit}.npz (float32), for offline pooling analyses (pool_explore.py).
Post-processing per fit (same fitted model):
  score   nll (current) | maha (min_k Mahalanobis^2) | chi2 (-log chi2_d tail of min_k Mahalanobis^2)
          | sp5, sp10 (nll with mixture weights conditioned on patch position, estimated from the
          unrotated references: pi_k(u) = (1-lam) pi_k + lam pi_k^spatial(u), lam = 0.5 / 1.0)
  fusion  raw (current: mean over layers of p95) | z (each layer standardised by the mean/std of
          its smoothed map on the unrotated references before pooling)

Usage: uv run gauss_screen.py --datasets mvtec_AD/screw --ns 1 2 4 --seeds 0 1 2 --fits k3 k3s2000
"""

import argparse
import sys

p = argparse.ArgumentParser()
p.add_argument("--datasets", nargs="+", required=True)
p.add_argument("--fits", nargs="+", default=["k3", "k3s500", "k3s2000", "k3s8000", "k6s2000", "k12s2000", "k3v"])
p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--out", default="gauss_screen")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--post", action="store_true", help="Also write the inherited post-processing variants (chi2).")
p.add_argument("--post2", action="store_true", help="Also write the mixture-use score variants.")
p.add_argument("--save-maps", action="store_true", help="Store the raw chi2 patch maps per layer.")
p.add_argument("--fill-scores", action="store_true",
               help="Only run units whose per-image score file is missing (rows go to --out).")
a = p.parse_args()

_argv, sys.argv = sys.argv, [sys.argv[0], "--phase", "screen", "--threads", str(a.threads), "--no-tta"]
import tune_experiments as TE  # noqa: E402
sys.argv = _argv
E, G = TE.E, TE.G

import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import log_ndtr  # noqa: E402

OUT = TE.OUT / f"{a.out}.csv"
SCORES = TE.OUT / "scores"  # per-image scores of every variant, for threshold-transfer analyses
FIELDS = ["timestamp", "dataset", "fit", "score", "fusion", "variant", "n_train", "seed",
          "img_ap", "img_auroc", "k_used", "seconds"]
SCORES_LIST, FUSIONS = ["nll", "maha", "chi2", "sp5", "sp10"], ["raw", "z"]
DIM, SKIP, REG = 256, int(os.environ.get("GAUSS_SKIP", "2")), 1e-2  # GAUSS_SKIP: exploratory override
DEV = torch.device(os.environ.get("GAUSS_GMM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
BATCHED = os.environ.get("GAUSS_BATCHED", "1") == "1"  # 0 = per-component loops (stage-1 code path)


AUG_ANGLES = {"rot8": list(range(0, 360, 45)), "rot4": [0, 90, 180, 270], "small": [0, 15, -15, 30, -30],
              "none": [0], "rot16": [i * 22.5 for i in range(16)]}


def parse_fit(fit):
    """k{K}[s{kappa}|lw][v][_{aug}][_d{D}][_skip{S}] -> (K, kappa, valid_only, aug, D, S); kappa = -1: Ledoit-Wolf."""
    m = re.fullmatch(r"k(\d+)(?:s(\d+|inf)|(lw))?(v?)(?:_(none|rot4|small|rot8|rot16))?(?:_d(\d+))?(?:_skip(\d+))?(_ppca)?(_diag)?", fit)
    kappa = -1.0 if m.group(3) else float(m.group(2) or 0)  # "sinf": kappa = infinity
    return (int(m.group(1)), kappa, bool(m.group(4)), m.group(5) or "rot8", int(m.group(6) or DIM),
            int(m.group(7) if m.group(7) is not None else SKIP), bool(m.group(8)), bool(m.group(9)))


def rotate_views(img_paths, angles):
    """Resize to RES, then rotate (bilinear, reflected borders), as TE.ref_images(.., 'rot8')."""
    import cv2
    out = []
    for q in img_paths:
        a = np.asarray(Image.open(q).convert("RGB").resize((E.RES, E.RES), Image.BICUBIC))
        h, w = a.shape[:2]
        for ang in angles:
            mtx = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            out.append(Image.fromarray(cv2.warpAffine(a, mtx, (w, h), flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_DEFAULT)))
    return out


MAX_ITER = int(os.environ.get("GAUSS_MAX_ITER", "200"))  # EM iteration cap (exploratory speed test)


class ShrinkGMM(TE.TorchGMM):
    def __init__(self, k, reg, seed, prior, kappa, diag=False):
        super().__init__(k, reg, seed, max_iter=MAX_ITER)
        self.prior, self.kappa, self.diag = prior, kappa, diag

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(0) + 10 * torch.finfo(X.dtype).eps
        self.means = (resp.T @ X) / nk[:, None]
        if BATCHED:  # same algebra, all components in one batched product
            diff = X[None] - self.means[:, None]
            s = (resp.T[:, :, None] * diff).transpose(1, 2) @ diff
            del diff
            k = nk[:, None, None]
            if self.kappa < 0:  # Ledoit-Wolf toward mu I, weighted by the responsibilities
                S = s / k
                eye = torch.eye(d, dtype=X.dtype, device=X.device)
                mu = torch.diagonal(S, dim1=1, dim2=2).sum(1) / d
                F_ = mu[:, None, None] * eye
                delta2 = ((S - F_) ** 2).sum((1, 2))
                diff = X[None] - self.means[:, None]
                sq = (diff ** 2).sum(2)
                xsx = ((diff @ S) * diff).sum(2)
                r2 = resp.T ** 2
                beta2 = ((r2 * (sq ** 2 - 2 * xsx)).sum(1) + r2.sum(1) * (S ** 2).sum((1, 2))) / nk ** 2
                del diff
                rho = (torch.minimum(beta2, delta2) / delta2).clamp(0, 1)
                cov = (1 - rho)[:, None, None] * S + rho[:, None, None] * F_
            elif np.isinf(self.kappa):  # every component uses the pooled covariance
                cov = self.prior.expand(self.k, d, d).clone()
            else:
                cov = (s + self.kappa * self.prior) / (k + self.kappa) if self.kappa else s / k
        else:
            covs = []
            for j in range(self.k):
                diff = X - self.means[j]
                s = (resp[:, j:j + 1] * diff).T @ diff
                covs.append((s + self.kappa * self.prior) / (nk[j] + self.kappa) if self.kappa else s / nk[j])
            cov = torch.stack(covs)
        if self.diag:
            cov = torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
        cov = cov + self.reg * torch.eye(d, dtype=X.dtype, device=X.device)
        self.chol = torch.linalg.cholesky(cov.cpu()).to(X.device)
        self.log_w = torch.log(nk / n)

    def _log_prob(self, X):
        if not BATCHED:
            return super()._log_prob(X)
        d2, logdet = self.maha(X)
        return -0.5 * (X.shape[1] * np.log(2 * np.pi) + logdet[None] + d2) + self.log_w[None]

    def maha(self, Z):
        """Squared Mahalanobis distance to every component [n, K] and log-determinants [K]."""
        logdet = 2 * torch.log(torch.diagonal(self.chol, dim1=1, dim2=2)).sum(1)
        if BATCHED:
            y = torch.linalg.solve_triangular(self.chol, (Z[None] - self.means[:, None]).transpose(1, 2), upper=False)
            return (y ** 2).sum(1).T, logdet
        d2 = torch.stack([(torch.linalg.solve_triangular(self.chol[j], (Z - self.means[j]).T, upper=False) ** 2).sum(0)
                          for j in range(self.k)], 1)
        return d2, logdet


class Layer:
    """PCA (dim + skip, drop skip) + ShrinkGMM, same as TE.LayerModel with whiten=False."""

    def __init__(self, X, k, kappa, seed, dim=DIM, skip=SKIP, diag=False):
        X = X.double()
        self.dim = dim
        self.mean = X.mean(0)
        Xc = X - self.mean
        evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
        order = torch.argsort(evals, descending=True)[: dim + skip]
        self.comp = evecs[:, order][:, skip:]
        self.evals = evals[order][skip:].clamp_min(1e-12)  # PCA variances (whitened 1-NN control)
        prior = torch.diag(self.evals)
        self.top = evecs[:, order]  # the D+S leading directions; residual = what lies outside them
        rest = torch.sort(evals, descending=True).values[dim + skip:]
        self.sig2 = float(rest.clamp_min(0).mean())
        self.dof = X.shape[1] - skip
        Z = self.project(X)
        del X, Xc
        self.k = min(k, max(2, Z.shape[0] // (dim + 10)))
        self.gmm = ShrinkGMM(self.k, REG, seed, prior, kappa, diag).fit(Z)

    def project(self, X):
        return (X.double() - self.mean) @ self.comp

    def residual(self, X, chunk=32 * 784):
        out = []
        for i in range(0, X.shape[0], chunk):
            xc = X[i:i + chunk].double() - self.mean
            out.append(((xc ** 2).sum(1) - ((xc @ self.top) ** 2).sum(1)).clamp_min(0).cpu())
        return torch.cat(out) / self.sig2

    def stats(self, X, chunk=32 * 784):
        out = [self.gmm.maha(self.project(X[i:i + chunk])) for i in range(0, X.shape[0], chunk)]
        return torch.cat([o[0] for o in out]).cpu(), out[0][1].cpu()


def valid_patches(n_img, grid, patch, thr=0.75):
    """Rot8 reference patches whose pixels come mostly from the image (not reflected borders)."""
    res = grid * patch
    masks = []
    for angle in range(0, 360, 45):
        m = cv2.getRotationMatrix2D((res / 2, res / 2), angle, 1.0)
        cov = cv2.warpAffine(np.ones((res, res), np.float32), m, (res, res), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        masks.append(cov.reshape(grid, patch, grid, patch).mean((1, 3)).ravel() >= thr)
    return np.tile(np.concatenate(masks), n_img)


def spatial_prior(resp, grid, log_w, lam, sigma=2.0, alpha=1.0):
    """resp: [n_ref * grid^2, K] posteriors of the unrotated references -> log pi_k(u) [grid^2, K]."""
    k = resp.shape[1]
    h = resp.reshape(-1, grid, grid, k).sum(0)
    h = np.stack([gaussian_filter(h[..., j], sigma, mode="nearest") for j in range(k)], -1).reshape(-1, k)
    pi = np.exp(log_w)
    sp = (h + alpha * pi) / (h.sum(1, keepdims=True) + alpha)
    return np.log((1 - lam) * pi + lam * sp)


def chi2_logsf(x, d):
    """log P(chi2_d >= x) via Wilson-Hilferty; finite for any x (scipy's logsf is -inf beyond ~1e3)."""
    z = ((x / d) ** (1 / 3) - (1 - 2 / (9 * d))) / np.sqrt(2 / (9 * d))
    return log_ndtr(-z)


def patch_scores(d2, logdet, log_w, kind, log_pi_u=None, n_img=None, dim=DIM):
    """d2: [n_img * P, K] numpy float64. Returns [n_img, P] patch scores (higher = more anomalous)."""
    if kind in ("maha", "chi2"):
        m = d2.min(1)
        s = -chi2_logsf(m, dim) if kind == "chi2" else m
        return s.reshape(n_img, -1)
    lw = np.broadcast_to(log_w, d2.shape) if log_pi_u is None else np.tile(log_pi_u, (n_img, 1))
    z = lw - 0.5 * logdet[None] - 0.5 * d2
    zmax = z.max(1, keepdims=True)
    return -(zmax[:, 0] + np.log(np.exp(z - zmax).sum(1))).reshape(n_img, -1)


def smooth_grid(s, grid, sigma=1):
    g = s.reshape(-1, grid, grid)
    return g.copy() if sigma == 0 else np.stack([gaussian_filter(x, sigma=sigma) for x in g])


def done_fits():
    """A fit counts as done only when all its score x fusion rows were written."""
    if not OUT.exists():
        return set()
    with open(OUT, newline="") as f:
        c = Counter((r["dataset"], r["fit"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f))
    return {k for k, v in c.items() if v >= len(SCORES_LIST) * len(FUSIONS)}


def save(rows):
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def run_unit(bb, ds, n, seed, fits, train_all, labels, test):
    refs = E.sample_train(train_all, n, seed)
    grid, P = bb.grid, bb.grid ** 2
    n_ref, n_test = len(refs), test[TE.LAYERS[0]].shape[0]
    feats = {}
    for fit in fits:
        t0 = time.time()
        k, kappa, valid, aug, dim, skip, ppca, diag = parse_fit(fit)
        if aug not in feats:
            imgs = TE.ref_images(refs, "rot8") if aug == "rot8" else rotate_views(refs, AUG_ANGLES[aug])
            feats[aug] = bb.extract(imgs, list(TE.LAYERS))
        tr, nv = feats[aug], len(AUG_ANGLES[aug])
        unrot = np.concatenate([np.arange(i * nv * P, i * nv * P + P) for i in range(n_ref)])
        per_layer, ks = [], []
        chi_dim = dim
        for l in TE.LAYERS:
            X = tr[l].reshape(-1, tr[l].shape[-1])
            Xfit = X[torch.from_numpy(valid_patches(n_ref, grid, bb.patch))] if valid else X
            m = Layer(Xfit.to(DEV), k, kappa, seed, dim, skip, diag)
            del Xfit
            ks.append(m.k)
            x = test[l]
            xt = torch.from_numpy(np.ascontiguousarray(x).reshape(-1, x.shape[-1])).to(DEV)
            d2t, logdet = m.stats(xt)
            d2r, _ = m.stats(X[torch.from_numpy(unrot)].to(DEV))
            if ppca:
                d2t = d2t + m.residual(xt)[:, None]
                d2r = d2r + m.residual(X[torch.from_numpy(unrot)].to(DEV))[:, None]
            if ppca:
                chi_dim = m.dof
            del xt
            log_w = m.gmm.log_w.cpu().numpy()
            per_layer.append((d2t.numpy(), d2r.numpy(), logdet.numpy(), log_w))
            del m
            torch.cuda.empty_cache()
        rows, img = [], {}
        for score in SCORES_LIST:
            maps_t, maps_r = [], []
            for d2t, d2r, logdet, log_w in per_layer:
                lp = None
                if score.startswith("sp"):
                    zr = log_w - 0.5 * logdet[None] - 0.5 * d2r
                    resp = np.exp(zr - zr.max(1, keepdims=True))
                    resp /= resp.sum(1, keepdims=True)
                    lp = spatial_prior(resp, grid, log_w, int(score[2:]) / 10)
                kind = "nll" if score.startswith("sp") else score
                maps_t.append(smooth_grid(patch_scores(d2t, logdet, log_w, kind, lp, n_test, chi_dim), grid))
                maps_r.append(smooth_grid(patch_scores(d2r, logdet, log_w, kind, lp, n_ref, chi_dim), grid))
            for fusion in FUSIONS:
                if fusion == "z":
                    s = np.mean([(E.pool(mt, "p95") - mr.mean()) / mr.std() for mt, mr in zip(maps_t, maps_r)], 0)
                else:
                    s = np.mean([E.pool(mt, "p95") for mt in maps_t], 0)
                au, ap = E.metrics(labels, s)
                img[f"{score}__{fusion}"] = s.astype(np.float32)
                rows.append(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, fit=fit, score=score,
                                 fusion=fusion, variant=f"{fit}|{score}|{fusion}", n_train=n, seed=seed,
                                 img_ap=f"{ap:.4f}", img_auroc=f"{au:.4f}", k_used="/".join(map(str, ks)),
                                 seconds=f"{time.time() - t0:.1f}"))
        if a.post:
            raw = [patch_scores(d2t, logdet, log_w, "chi2", None, n_test, chi_dim) for d2t, _, logdet, log_w in per_layer]
            post = {f"chi2_sig{sg}": np.mean([E.pool(smooth_grid(r, grid, sg), "p95") for r in raw], 0) for sg in (0, 2)}
            post.update({f"chi2_{pl}": np.mean([E.pool(smooth_grid(r, grid), pl) for r in raw], 0) for pl in ("max", "p99")})
            post.update({f"chi2_L{-l}": E.pool(smooth_grid(r, grid), "p95") for l, r in zip(TE.LAYERS, raw)})
            for name, s in post.items():
                au, ap = E.metrics(labels, s)
                rows.append(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, fit=fit, score=name,
                                 fusion="raw", variant=f"{fit}|{name}|raw", n_train=n, seed=seed,
                                 img_ap=f"{ap:.4f}", img_auroc=f"{au:.4f}", k_used="/".join(map(str, ks)),
                                 seconds=f"{time.time() - t0:.1f}"))
        if a.post2:
            post = {}
            for name in ("chi2_wtail", "maha_det", "ent"):
                maps = []
                for d2t, _, logdet, log_w in per_layer:
                    if name == "chi2_wtail":
                        z = log_w[None] + chi2_logsf(d2t, chi_dim)
                        zm = z.max(1, keepdims=True)
                        s = -(zm[:, 0] + np.log(np.exp(z - zm).sum(1)))
                    elif name == "maha_det":
                        s = (d2t + (logdet - logdet.mean())[None]).min(1)
                    else:
                        z = log_w[None] - 0.5 * logdet[None] - 0.5 * d2t
                        r = np.exp(z - z.max(1, keepdims=True))
                        r /= r.sum(1, keepdims=True)
                        s = -(r * np.log(r + 1e-12)).sum(1)
                    maps.append(smooth_grid(s.reshape(n_test, -1), grid))
                post[name] = np.mean([E.pool(m_, "p95") for m_ in maps], 0)
            for name, s in post.items():
                au, ap = E.metrics(labels, s)
                rows.append(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", dataset=ds, fit=fit, score=name,
                                 fusion="raw", variant=f"{fit}|{name}|raw", n_train=n, seed=seed,
                                 img_ap=f"{ap:.4f}", img_auroc=f"{au:.4f}", k_used="/".join(map(str, ks)),
                                 seconds=f"{time.time() - t0:.1f}"))
        if a.save_maps:
            md = TE.OUT / "maps" / ds.replace("/", "__")
            md.mkdir(parents=True, exist_ok=True)
            mp = {}
            for l, (d2t, d2r, logdet, log_w) in zip(TE.LAYERS, per_layer):
                mp[f"t{-l}"] = patch_scores(d2t, logdet, log_w, "chi2", None, n_test, chi_dim).reshape(-1, grid, grid).astype(np.float32)
                mp[f"r{-l}"] = patch_scores(d2r, logdet, log_w, "chi2", None, n_ref, chi_dim).reshape(-1, grid, grid).astype(np.float32)
            np.savez_compressed(md / f"N{n}_s{seed}_{fit}.npz", labels=labels, **mp)
        sd = SCORES / ds.replace("/", "__")
        sd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sd / f"N{n}_s{seed}_{fit}.npz", labels=labels, **img)
        save(rows)
        base = next(r for r in rows if r["score"] == "nll" and r["fusion"] == "raw")
        E.log(f"  gauss {ds} N={n} seed {seed} {fit}: AP {base['img_ap']} K {base['k_used']} {base['seconds']}s")


def main():
    bb = E.Backbone("dinov3l")
    fin = done_fits()
    for ds in a.datasets:
        try:
            units = [(n, s) for n in a.ns for s in a.seeds]
            if all((ds, f, n, s) in fin for n, s in units for f in a.fits):
                continue
            train_all, test_info = G.load_dataset(ds)
            labels = np.array([t["label"] for t in test_info])
            test = {l: np.load(TE.cdir(ds) / f"test_orig_L{l}.npy", mmap_mode="r") for l in TE.LAYERS}
            for n, s in units:
                todo = [f for f in a.fits if (ds, f, n, s) not in fin and not (
                    a.fill_scores and (SCORES / ds.replace("/", "__") / f"N{n}_s{s}_{f}.npz").exists())]
                if todo:
                    run_unit(bb, ds, n, s, todo, train_all, labels, test)
            del test
        except Exception as e:  # log and continue; rerunning fills the gap
            E.log(f"ERROR {ds}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
