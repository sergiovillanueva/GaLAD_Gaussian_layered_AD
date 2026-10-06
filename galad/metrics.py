"""Image metrics and the standard pixel metrics of MVTec AD (AU-PRO up to a false-positive limit).

AU-PRO: per-region overlap against the false-positive rate over all normal pixels of all test images,
integrated up to the limit and divided by it. Scores are accumulated over 2^16 levels.
"""

import numpy as np
from scipy.ndimage import label as cc_label
from sklearn.metrics import average_precision_score, roc_auc_score


def image_metrics(labels, scores):
    """Returns (image AUROC, image AP)."""
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


def mask_regions(masks):
    """Precompute connected components once per category: list of (labels, sizes)."""
    out = []
    for k in masks:
        lab, num = cc_label(k)
        out.append((lab, np.bincount(lab.ravel()) if num else None))
    return out


def std_pixel_metrics_binned(maps, masks, regions=None, limit=0.3, bins=1 << 16):
    """Same quantities as std_pixel_metrics (+ pixel AP), accumulated image by image over
    `bins` score levels. Low memory and O(pixels); scores within one level count as a tie.
    Returns (aupro, pixel_auroc, pixel_ap)."""
    regions = regions or mask_regions(masks)
    lo = min(float(m.min()) for m in maps)
    hi = max(float(m.max()) for m in maps)
    scale = (bins - 1) / (hi - lo) if hi > lo else 0.0
    neg, pos, prow = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    n_regions = sum(int((lab.max() if sizes is not None else 0)) for lab, sizes in regions)
    for m, k, (lab, sizes) in zip(maps, masks, regions):
        idx = np.clip(((m - lo) * scale).astype(np.int64), 0, bins - 1).ravel()
        g = k.ravel().astype(bool)
        neg += np.bincount(idx[~g], minlength=bins)
        pos += np.bincount(idx[g], minlength=bins)
        if sizes is not None:
            l = lab.ravel()[g]
            prow += np.bincount(idx[g], weights=1.0 / sizes[l], minlength=bins)
    n_neg, n_pos = neg.sum(), pos.sum()
    if n_pos == 0 or n_neg == 0 or n_regions == 0:
        return None, None, None
    fp, tp = np.cumsum(neg[::-1]), np.cumsum(pos[::-1])
    fpr = np.r_[0.0, fp / n_neg]
    tpr = np.r_[0.0, tp / n_pos]
    pr = np.r_[0.0, np.cumsum(prow[::-1]) / n_regions]
    auroc = float(np.trapezoid(tpr, fpr))
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    ap = float(np.sum(np.diff(tpr) * prec))
    aupros = []
    for lim in np.atleast_1d(limit):  # several limits share one pass over the maps
        i = np.searchsorted(fpr, lim, side="right")
        x, y = fpr[:i], pr[:i]
        if i < fpr.size and fpr[i - 1] < lim:
            t = (lim - fpr[i - 1]) / (fpr[i] - fpr[i - 1])
            x, y = np.r_[x, lim], np.r_[y, pr[i - 1] + t * (pr[i] - pr[i - 1])]
        aupros.append(float(np.trapezoid(y, x) / lim))
    return (aupros[0] if np.isscalar(limit) else aupros), auroc, ap
