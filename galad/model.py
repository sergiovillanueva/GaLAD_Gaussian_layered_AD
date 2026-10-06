"""GaLAD: Gaussian mixture with shrunk covariances and chi-square tail score (Section 3 of the paper).

Per layer: PCA of the reference patches (components 3 to 258, d = 256); Gaussian mixture with K = 12
full-covariance components fitted by EM, where the covariance update is replaced by
    Sigma_k = (n_k C_k + kappa Lambda) / (n_k + kappa),  kappa = 2000,  Lambda = diag(PCA variances)  (Eq. 1)
Patch score: s(x) = -log P(chi2_d >= min_k delta_k^2) (Eq. 2, Wilson-Hilferty). Map: Gaussian smoothing
(sigma = 1 patch). Image score: 95th percentile, mean over the two layers. All fitting in float64.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import log_ndtr

DIM, SKIP, RIDGE = 256, 2, 1e-2


def chi2_tail_score(x, d=DIM):
    """-log P(chi2_d >= x) with the Wilson-Hilferty approximation; log_ndtr keeps it finite for large x."""
    z = ((x / d) ** (1 / 3) - (1 - 2 / (9 * d))) / np.sqrt(2 / (9 * d))
    return -log_ndtr(-z)


def kmeans_labels(X, k, gen, iters=100):
    """k-means++ seeding followed by Lloyd iterations."""
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


class ShrunkMixture:
    """Gaussian mixture fitted by EM with the shrunk covariance update of Eq. 1."""

    def __init__(self, k, kappa, prior, seed, tol=1e-3, max_iter=200):
        self.k, self.kappa, self.prior, self.seed, self.tol, self.max_iter = k, kappa, prior, seed, tol, max_iter

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(0) + 10 * torch.finfo(X.dtype).eps
        self.means = (resp.T @ X) / nk[:, None]
        diff = X[None] - self.means[:, None]
        s = (resp.T[:, :, None] * diff).transpose(1, 2) @ diff  # n_k C_k for every component
        cov = (s + self.kappa * self.prior) / (nk[:, None, None] + self.kappa)
        cov = cov + RIDGE * torch.eye(d, dtype=X.dtype, device=X.device)
        self.chol = torch.linalg.cholesky(cov.cpu()).to(X.device)
        self.log_w = torch.log(nk / n)

    def mahalanobis(self, Z):
        """Squared Mahalanobis distance of every row of Z to every component: [n, K]; and log-determinants."""
        logdet = 2 * torch.log(torch.diagonal(self.chol, dim1=1, dim2=2)).sum(1)
        y = torch.linalg.solve_triangular(self.chol, (Z[None] - self.means[:, None]).transpose(1, 2), upper=False)
        return (y ** 2).sum(1).T, logdet

    def fit(self, X):
        gen = torch.Generator(device=X.device).manual_seed(self.seed)
        self._m_step(X, F.one_hot(kmeans_labels(X, self.k, gen), self.k).to(X.dtype))
        lb = -np.inf
        for _ in range(self.max_iter):
            d2, logdet = self.mahalanobis(X)
            lp = -0.5 * (X.shape[1] * np.log(2 * np.pi) + logdet[None] + d2) + self.log_w[None]
            norm = torch.logsumexp(lp, 1)
            self._m_step(X, torch.exp(lp - norm[:, None]))
            new = norm.mean().item()
            if abs(new - lb) < self.tol:
                break
            lb = new
        return self


class LayerModel:
    """PCA + shrunk mixture for one layer."""

    def __init__(self, X, k, kappa, seed):
        X = X.double()
        self.mean = X.mean(0)
        Xc = X - self.mean
        evals, evecs = torch.linalg.eigh(Xc.T @ Xc / (X.shape[0] - 1))
        order = torch.argsort(evals, descending=True)[: DIM + SKIP]
        self.comp = evecs[:, order][:, SKIP:]
        prior = torch.diag(evals[order][SKIP:])
        Z = self.project(X)
        self.k = min(k, max(2, Z.shape[0] // (DIM + 10)))  # at most one component per 266 patches
        self.gmm = ShrunkMixture(self.k, kappa, prior, seed).fit(Z)

    def project(self, X):
        return (X.double() - self.mean) @ self.comp

    def min_distance(self, X, chunk=32 * 784):
        return torch.cat([self.gmm.mahalanobis(self.project(X[i:i + chunk]))[0].min(1).values
                          for i in range(0, X.shape[0], chunk)]).cpu().numpy()


class GaLAD:
    def __init__(self, layers, k=12, kappa=2000.0, sigma=1.0, percentile=95, seed=0, device=None):
        self.layers, self.k, self.kappa, self.sigma, self.q, self.seed = layers, k, kappa, sigma, percentile, seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, ref_feats):
        """ref_feats: list of {layer: [P, D]} for the rotated reference views."""
        self.models = [LayerModel(torch.cat([f[l].float() for f in ref_feats]).to(self.device), self.k, self.kappa,
                                  self.seed) for l in self.layers]
        return self

    def score(self, test_feats, patch=16):
        """Returns image scores [n] and pixel maps (list of 448x448 arrays)."""
        layer_maps = []
        for m, l in zip(self.models, self.layers):
            s = chi2_tail_score(m.min_distance(torch.cat([f[l] for f in test_feats]).to(self.device)))
            p = test_feats[0][l].shape[0]
            g = int(round(p ** 0.5))
            layer_maps.append([gaussian_filter(s[i * p:(i + 1) * p].reshape(g, g), sigma=self.sigma)
                               for i in range(len(test_feats))])
        scores = np.mean([[np.percentile(x, self.q) for x in maps] for maps in layer_maps], 0)
        maps = [zoom(np.mean([lm[i] for lm in layer_maps], 0), patch, order=1) for i in range(len(test_feats))]
        return scores, maps
