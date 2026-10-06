"""Standard pixel-level localization metrics (MVTec AD definition), computed exactly.

aupro_std: per-region overlap (PRO) vs false-positive rate, where the FPR counts ALL normal
pixels of ALL test images (including normal pixels of anomalous images), integrated with the
trapezoidal rule from FPR=0 up to exactly `limit` (linear interpolation at the limit) and
divided by `limit`. Every distinct score is a threshold (one global sort).
pixel_auroc: area under the pixel ROC curve, from the same sort.
"""

import numpy as np
from scipy.ndimage import label as cc_label


def std_pixel_metrics(maps, masks, limit=0.3):
    """maps, masks: lists of equally shaped 2-D arrays (masks binary). Returns (aupro, pixel_auroc)."""
    scores = np.concatenate([m.ravel() for m in maps]).astype(np.float64)
    gt = np.concatenate([k.ravel() for k in masks]).astype(bool)
    n_neg, n_pos = int((~gt).sum()), int(gt.sum())
    if n_pos == 0 or n_neg == 0:
        return None, None
    # Per-pixel PRO weight: 1 / (component size * number of components).
    pro_w = np.zeros(scores.shape[0])
    comps, offset, regions = [], 0, 0
    for k in masks:
        lab, num = cc_label(k)
        if num:
            sizes = np.bincount(lab.ravel())
            w = np.zeros_like(lab, dtype=np.float64)
            nz = lab > 0
            w[nz] = 1.0 / sizes[lab[nz]]
            comps.append((offset, w.ravel()))
            regions += num
        offset += k.size
    for off, w in comps:
        pro_w[off:off + w.size] = w
    pro_w /= regions

    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    neg = np.cumsum(~gt[order]) / n_neg          # FPR after each pixel
    pos = np.cumsum(gt[order]) / n_pos           # TPR after each pixel
    pro = np.cumsum(pro_w[order])                # PRO after each pixel
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]  # last index of each distinct score
    fpr = np.r_[0.0, neg[last]]
    tpr = np.r_[0.0, pos[last]]
    pr = np.r_[0.0, pro[last]]

    auroc = float(np.trapezoid(tpr, fpr))
    i = np.searchsorted(fpr, limit, side="right")
    x, y = fpr[:i], pr[:i]
    if i < fpr.size and fpr[i - 1] < limit:  # interpolate the curve at exactly `limit`
        t = (limit - fpr[i - 1]) / (fpr[i] - fpr[i - 1])
        x = np.r_[x, limit]
        y = np.r_[y, pr[i - 1] + t * (pr[i] - pr[i - 1])]
    return float(np.trapezoid(y, x) / limit), auroc


if __name__ == "__main__":
    # Self-checks with known answers.
    H = 10
    m1, k1 = np.zeros((H, H)), np.zeros((H, H), np.uint8)
    k1[2:4, 2:4] = 1
    m1[2:4, 2:4] = 1.0                             # perfect separation
    a, r = std_pixel_metrics([m1, np.zeros((H, H))], [k1, np.zeros((H, H), np.uint8)])
    print(f"perfect: aupro {a:.4f} (expected 1.0), auroc {r:.4f} (expected 1.0)")
    rng = np.random.default_rng(0)
    maps = [rng.random((64, 64)) for _ in range(20)]
    masks = []
    for _ in range(20):
        k = np.zeros((64, 64), np.uint8)
        y, x = rng.integers(5, 50, 2)
        k[y:y + 8, x:x + 8] = 1
        masks.append(k)
    a, r = std_pixel_metrics(maps, masks)
    print(f"random: aupro {a:.4f} (expected ~0.15 = 0.3/2), auroc {r:.4f} (expected ~0.5)")
    from sklearn.metrics import roc_auc_score
    print(f"sklearn pixel auroc on random: {roc_auc_score(np.concatenate([k.ravel() for k in masks]), np.concatenate([m.ravel() for m in maps])):.4f}")


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
