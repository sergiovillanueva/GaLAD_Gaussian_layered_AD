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
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import log_ndtr  # noqa: E402

OUT = TE.OUT / f"{a.out}.csv"
SCORES = TE.OUT / "scores"  # per-image scores of every variant, for threshold-transfer analyses
FIELDS = ["timestamp", "dataset", "fit", "score", "fusion", "variant", "n_train", "seed",
          "img_ap", "img_auroc", "k_used", "seconds"]
SCORES_LIST, FUSIONS = ["nll", "maha", "chi2", "sp5", "sp10"], ["raw", "z"]
DIM, SKIP, REG = 256, 2, 1e-2
DEV = torch.device(os.environ.get("GAUSS_GMM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
BATCHED = os.environ.get("GAUSS_BATCHED", "1") == "1"  # 0 = per-component loops (stage-1 code path)


def parse_fit(fit):
    m = re.fullmatch(r"k(\d+)(?:s(\d+|inf))?(v?)", fit)
    return int(m.group(1)), float(m.group(2) or 0), bool(m.group(3))  # "sinf": kappa = infinity


class ShrinkGMM(TE.TorchGMM):
    def __init__(self, k, reg, seed, prior, kappa):
        super().__init__(k, reg, seed)
        self.prior, self.kappa = prior, kappa

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(0) + 10 * torch.finfo(X.dtype).eps
        self.means = (resp.T @ X) / nk[:, None]
        if BATCHED:  # same algebra, all components in one batched product
            diff = X[None] - self.means[:, None]
            s = (resp.T[:, :, None] * diff).transpose(1, 2) @ diff
            del diff
            k = nk[:, None, None]
            if np.isinf(self.kappa):  # every component uses the pooled covariance
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

    def __init__(self, X, k, kappa, seed):
        X = X.double()
        self.mean = X.mean(0)
        Xc = X - self.mean
        evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
        order = torch.argsort(evals, descending=True)[: DIM + SKIP]
        self.comp = evecs[:, order][:, SKIP:]
        prior = torch.diag(evals[order][SKIP:])
        Z = self.project(X)
        del X, Xc
        self.k = min(k, max(2, Z.shape[0] // (DIM + 10)))
        self.gmm = ShrinkGMM(self.k, REG, seed, prior, kappa).fit(Z)

    def project(self, X):
        return (X.double() - self.mean) @ self.comp

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


def patch_scores(d2, logdet, log_w, kind, log_pi_u=None, n_img=None):
    """d2: [n_img * P, K] numpy float64. Returns [n_img, P] patch scores (higher = more anomalous)."""
    if kind in ("maha", "chi2"):
        m = d2.min(1)
        s = -chi2_logsf(m, DIM) if kind == "chi2" else m
        return s.reshape(n_img, -1)
    lw = np.broadcast_to(log_w, d2.shape) if log_pi_u is None else np.tile(log_pi_u, (n_img, 1))
    z = lw - 0.5 * logdet[None] - 0.5 * d2
    zmax = z.max(1, keepdims=True)
    return -(zmax[:, 0] + np.log(np.exp(z - zmax).sum(1))).reshape(n_img, -1)


def smooth_grid(s, grid):
    return np.stack([gaussian_filter(x, sigma=1) for x in s.reshape(-1, grid, grid)])


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
    tr = bb.extract(TE.ref_images(refs, "rot8"), list(TE.LAYERS))
    grid, P = bb.grid, bb.grid ** 2
    n_ref, n_test = len(refs), test[TE.LAYERS[0]].shape[0]
    unrot = np.concatenate([np.arange(i * 8 * P, i * 8 * P + P) for i in range(n_ref)])
    for fit in fits:
        t0 = time.time()
        k, kappa, valid = parse_fit(fit)
        per_layer, ks = [], []
        for l in TE.LAYERS:
            X = tr[l].reshape(-1, tr[l].shape[-1])
            Xfit = X[torch.from_numpy(valid_patches(n_ref, grid, bb.patch))] if valid else X
            m = Layer(Xfit.to(DEV), k, kappa, seed)
            del Xfit
            ks.append(m.k)
            x = test[l]
            d2t, logdet = m.stats(torch.from_numpy(np.ascontiguousarray(x).reshape(-1, x.shape[-1])).to(DEV))
            d2r, _ = m.stats(X[torch.from_numpy(unrot)].to(DEV))
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
                maps_t.append(smooth_grid(patch_scores(d2t, logdet, log_w, kind, lp, n_test), grid))
                maps_r.append(smooth_grid(patch_scores(d2r, logdet, log_w, kind, lp, n_ref), grid))
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
