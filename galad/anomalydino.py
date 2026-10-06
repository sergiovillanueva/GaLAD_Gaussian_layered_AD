"""AnomalyDINO scoring on a given backbone (the 'AnomalyDINO (DINOv3-L)' comparator of the paper):
cosine nearest-neighbor distance to the patches of the rotated references (last layer), image score =
mean of the top 1% patch distances, map = bilinear upsampling + Gaussian smoothing (sigma = 4 pixels)."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class AnomalyDINO:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, ref_feats):
        self.bank = F.normalize(torch.cat([f["last"] for f in ref_feats]).to(self.device), dim=-1)
        return self

    def score(self, test_feats, patch=16):
        scores, maps = [], []
        for f in test_feats:
            q = F.normalize(f["last"].float().to(self.device), dim=-1)
            d = (1 - (q @ self.bank.T).max(1).values).cpu().numpy()
            k = max(1, int(0.01 * d.size))
            scores.append(np.sort(d)[-k:].mean())
            g = int(round(d.size ** 0.5))
            maps.append(gaussian_filter(cv2.resize(d.reshape(g, g).astype(np.float32), (g * patch, g * patch),
                                                   interpolation=cv2.INTER_LINEAR), sigma=4))
        return np.array(scores), maps
