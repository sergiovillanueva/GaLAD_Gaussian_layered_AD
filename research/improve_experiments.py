"""Resumable experiments for the GaLAD revision (see IMPROVEMENT_PLAN.md).

Stages (run one at a time, each is resumable: finished rows are skipped):

  ablation  Re-run every ablation reported in the paper under ONE protocol
            (8 ablation categories, N=10, 5 seeds) and save every row.
  adino     Faithful AnomalyDINO re-implementation (k=1 cosine NN, mean of
            the top 1% patch distances, rotated references). Run with
            --backbone dinov2s to validate against the published numbers,
            and with --backbone dinov3l for the same-backbone comparison.
  augment   GaLAD with augmented reference images (flips / rotations) and
            plain GaLAD at the literature shot counts N in {1, 2, 4}.

Usage examples:
  uv run improve_experiments.py --stage ablation
  uv run improve_experiments.py --stage adino --backbone dinov2s
  uv run improve_experiments.py --stage adino --backbone dinov3l
  uv run improve_experiments.py --stage augment
  uv run improve_experiments.py --stage augment --plan      # print work left, no GPU
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--stage", required=True, choices=["ablation", "adino", "augment"])
parser.add_argument("--backbone", default="dinov3l", choices=["dinov3l", "dinov2s"],
                    help="Only used by --stage adino.")
parser.add_argument("--datasets", nargs="*", default=None, help="Override the stage's category list.")
parser.add_argument("--threads", type=int, default=4, help="CPU threads for numpy/sklearn.")
parser.add_argument("--device", type=int, default=0, help="GPU id.")
parser.add_argument("--plan", action="store_true", help="Only print what is left to run.")
parser.add_argument("--ns", nargs="*", type=int, default=None, help="Override shot counts (adino/augment).")
parser.add_argument("--pixel", action="store_true", help="adino: also compute AUsPRO (separate CSV, adino_rot only).")
args = parser.parse_args()

# Limit CPU threads before numpy/sklearn are imported.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(args.threads)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))  # a launcher may pin another GPU

# galad_train_all parses sys.argv at import time; hide our flags from it so we
# can reuse its loaders, metrics, feature extractor and GaLAD model unchanged.
_argv, sys.argv = sys.argv, [sys.argv[0]]
import galad_train_all as G  # noqa: E402
sys.argv = _argv

import csv  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter, label as cc_label, zoom  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

OUT_DIR = Path("output/improve")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
RES = 448

# Categories named in the paper's ablation protocol (Section 6, main.tex).
ABLATION_CATS = [
    "mvtec_AD/bottle", "mvtec_AD/carpet", "mvtec_AD/hazelnut", "mvtec_AD/leather",
    "VisA/capsules", "VisA/pcb1", "GoodsAD/drink_bottle", "GoodsAD/food_box",
]
MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
         "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
        "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
LIT_CATS = [f"mvtec_AD/{c}" for c in MVTEC] + [f"VisA/{c}" for c in VISA]

FIELDS = ["timestamp", "stage", "backbone", "dataset", "variant", "n_train", "seed",
          "img_auroc", "img_ap", "auspro", "seconds"]


# =============================================================================
# Bookkeeping
# =============================================================================

def csv_path():
    name = args.stage if args.stage != "adino" else f"adino_{args.backbone}" + ("_pix" if args.pixel else "")
    return OUT_DIR / f"{name}.csv"


def done_keys():
    p = csv_path()
    if not p.exists():
        return set()
    with open(p, newline="") as f:
        return {(r["dataset"], r["variant"], int(r["n_train"]), int(r["seed"]))
                for r in csv.DictReader(f)}


def save_row(row):
    p = csv_path()
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        row = dict(row, timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", stage=args.stage,
                   backbone=args.backbone if args.stage == "adino" else "dinov3l")
        w.writerow({k: row.get(k, "") for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(OUT_DIR / f"{csv_path().stem}_log.txt", "a") as f:
        f.write(line + "\n")


def sample_train(paths, n, seed):
    """Same draw as galad_train_all.py, so seeds map to identical reference sets."""
    rng = random.Random(seed)
    if n and len(paths) > n:
        return sorted(rng.sample(sorted(paths), n))
    return list(paths)


def fmt(v):
    return "" if v is None else f"{v:.4f}"


# =============================================================================
# Feature extraction from PIL images (needed for augmentation)
# =============================================================================

class Backbone:
    """Wraps DINOv3 ViT-L/16 (paper encoder) or DINOv2 ViT-S/14 (AnomalyDINO default)."""

    def __init__(self, kind):
        from transformers import AutoImageProcessor, AutoModel
        name = {"dinov3l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "dinov2s": "facebook/dinov2-small"}[kind]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Hard cap per process: exceeding it raises a visible CUDA OOM instead of letting Windows
        # page GPU memory to RAM, which stalled runs silently for hours (IMPROVEMENT_PLAN s. 20).
        frac = os.environ.get("GALAD_GPU_FRACTION")
        if frac and self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(frac))
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, output_hidden_states=True).to(self.device).eval()
        self.patch = self.model.config.patch_size
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0)
        self.grid = RES // self.patch
        self.n_patches = self.grid * self.grid
        log(f"Backbone {name} on {self.device}, grid {self.grid}x{self.grid}")

    def extract(self, images, layers, batch_size=16):
        """images: PIL images or paths (opened per batch to bound RAM).
        layers: hidden-state indices, or 'last' for the final normalized output."""
        out = {l: [] for l in layers}
        start = 1 + self.n_reg
        for i in range(0, len(images), batch_size):
            batch = [(Image.open(im).convert("RGB") if isinstance(im, (str, Path)) else im)
                     .resize((RES, RES), Image.BICUBIC) for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt",
                                    do_resize=False, do_center_crop=False).to(self.device)
            with torch.inference_mode(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
                o = self.model(**inputs)
            for l in layers:
                h = o.last_hidden_state if l == "last" else o.hidden_states[l]
                out[l].append(h[:, start:start + self.n_patches].float().cpu())
        return {l: torch.cat(v) for l, v in out.items()}


def open_rgb(paths):
    return [Image.open(p).convert("RGB") for p in paths]


def dihedral(img):
    """The 8 flips/rotations by multiples of 90 degrees (no padding artefacts)."""
    rots = [img, img.transpose(Image.ROTATE_90), img.transpose(Image.ROTATE_180),
            img.transpose(Image.ROTATE_270)]
    return rots + [r.transpose(Image.FLIP_LEFT_RIGHT) for r in rots]


def hflip(img):
    return [img, img.transpose(Image.FLIP_LEFT_RIGHT)]


def rot45(img):
    """AnomalyDINO rotations as in its official code (src/utils.py): 0..315 in 45-degree steps,
    bilinear, cv2.BORDER_DEFAULT (reflected) borders."""
    import cv2
    a = np.asarray(img)
    h, w = a.shape[:2]
    out = []
    for angle in range(0, 360, 45):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(Image.fromarray(cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_DEFAULT)))
    return out


def augment(images, fn):
    return [a for im in images for a in fn(im)]


# =============================================================================
# Scoring helpers
# =============================================================================

def fit_galad(train_feats, layers, n_comp=3, skip=2, dim=256, seed=0, force_k=False):
    """GaLADConfig.fit caps K at n_patches // (dim + 10) (K=2 for a single image).
    force_k=True refits each layer's GMM with exactly n_comp components (N=1 control)."""
    m = G.GaLADConfig(name="galad", layers=layers, pca_skip=skip, n_comp=n_comp,
                      reduction="pca", dim=dim, grid_size=None, random_state=seed)
    m.fit({l: train_feats[l] for l in layers})
    if force_k:
        from sklearn.mixture import GaussianMixture
        for l in layers:
            if m.gmms[l].n_components == n_comp:
                continue
            f = train_feats[l]
            red = m.reducers[l].transform(f.reshape(-1, f.shape[-1]).numpy().astype(np.float32))[:, skip:]
            m.gmms[l] = GaussianMixture(n_components=n_comp, covariance_type="full", reg_covar=1e-2,
                                        random_state=seed, n_init=1, max_iter=200).fit(red)
    return m


def raw_maps(model, test_feats, grid):
    """Unsmoothed per-layer NLL maps: {layer: array [n_test, grid, grid]}."""
    maps = {}
    for l in model.layers:
        f = test_feats[l]
        flat = f.reshape(-1, f.shape[-1]).numpy().astype(np.float32)
        red = model.reducers[l].transform(flat)[:, model.pca_skip:]
        nll = -model.gmms[l].score_samples(red)
        maps[l] = nll.reshape(f.shape[0], grid, grid)
    return maps


def smooth(maps, sigma):
    if sigma <= 0:
        return maps
    return {l: np.stack([gaussian_filter(m, sigma=sigma) for m in v]) for l, v in maps.items()}


def pool(a, how):
    flat = a.reshape(a.shape[0], -1)
    if how == "mean":
        return flat.mean(1)
    if how == "max":
        return flat.max(1)
    if how == "top1pct":
        k = max(1, int(0.01 * flat.shape[1]))  # floor, as AnomalyDINO's mean_top1p
        return np.sort(flat, 1)[:, -k:].mean(1)
    return np.percentile(flat, int(how[1:]), axis=1)  # "p90", "p95", "p99"


def image_scores(maps, how="p95", rule="layer_mean"):
    """rule='layer_mean': mean over layers of per-layer pooled scores (what the code does).
    rule='fused': pool the averaged map (what Section 3.5 of the paper states)."""
    if rule == "fused":
        return pool(np.mean(list(maps.values()), axis=0), how)
    return np.mean([pool(v, how) for v in maps.values()], axis=0)


def fused_map(maps):
    return np.mean(list(maps.values()), axis=0)


def fast_auspro(anomaly_maps, gt_masks, image_labels, alpha=0.3):
    """Same result as galad_train_all.compute_auspro (checked to 1e-16 on synthetic data),
    ~70x faster: connected components are labelled once instead of once per threshold."""
    labels = np.asarray(image_labels)
    normal_idx, abnormal_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return None, None
    normal = np.concatenate([anomaly_maps[i].ravel() for i in normal_idx])
    thresholds = np.linspace(normal.max(), normal.min(), 100)
    fprs = (len(normal) - np.searchsorted(np.sort(normal), thresholds, side="left")) / len(normal)
    regions = []
    for i in abnormal_idx:
        lab, num = cc_label(gt_masks[i])
        for cc in range(1, num + 1):
            vals = np.sort(anomaly_maps[i][lab == cc])
            if vals.size:
                regions.append(vals)
    pros = np.zeros(len(thresholds))
    for vals in regions:
        pros += (vals.size - np.searchsorted(vals, thresholds, side="left")) / vals.size
    if regions:
        pros /= len(regions)
    valid = fprs <= alpha
    if not np.any(valid):
        return None, None
    order = np.argsort(fprs[valid])
    xf, yf = fprs[valid][order], pros[valid][order]
    return float(yf[-1]), float(np.trapezoid(yf, xf) / alpha)


def masks_448(test_info):
    shape = (RES, RES)
    masks = []
    for e in test_info:
        mask = G.load_mask(e)
        if mask is not None:
            gt = (mask > 0).astype(np.uint8)
            if gt.shape != shape:
                gt = (zoom(gt.astype(float), (shape[0] / gt.shape[0], shape[1] / gt.shape[1]),
                           order=0) > 0.5).astype(np.uint8)
        else:
            gt = np.zeros(shape, np.uint8)
        masks.append(gt)
    return masks


def adino_maps_448(d, grid):
    """AnomalyDINO's dists2map: bilinear resize of the patch-distance grid, then Gaussian sigma=4."""
    import cv2
    return [gaussian_filter(cv2.resize(x.reshape(grid, grid).astype(np.float32), (RES, RES),
                                       interpolation=cv2.INTER_LINEAR), sigma=4) for x in d]


def auspro_448(maps_grid, test_info):
    """Same pixel protocol as galad_train_all.py (maps and masks at 448x448)."""
    if not any(e.get("mask_path") or e.get("mask_dir") for e in test_info):
        return None
    shape = (RES, RES)
    maps2d, masks2d, labels = [], [], []
    for am, e in zip(maps_grid, test_info):
        mask = G.load_mask(e)
        if mask is not None:
            gt = (mask > 0).astype(np.uint8)
            if gt.shape != shape:
                gt = (zoom(gt.astype(float), (shape[0] / gt.shape[0], shape[1] / gt.shape[1]),
                           order=0) > 0.5).astype(np.uint8)
        else:
            gt = np.zeros(shape, np.uint8)
        maps2d.append(zoom(am, (shape[0] / am.shape[0], shape[1] / am.shape[1]), order=1))
        masks2d.append(gt)
        labels.append(int(e.get("label", 0)))
    return fast_auspro(maps2d, masks2d, labels)[1]


def metrics(labels, scores):
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


# =============================================================================
# Stage definitions: each returns the list of (variant, n_train) it will produce
# =============================================================================

ABL_LAYER_SETS = {"layers_-1": [-1], "layers_-6": [-6], "layers_-12": [-12], "default": [-6, -12]}
ABL_SIGMAS = [0, 2, 3]            # sigma=1 is the default row
ABL_K = [1, 5, 10]                # K=3 is the default row
ABL_SKIP = [0, 1, 3]              # skip=2 is the default row
ABL_POOL = ["mean", "max", "p90", "p99", "top1pct"]


def ablation_variants():
    v = list(ABL_LAYER_SETS)
    v += [f"sigma_{s}" for s in ABL_SIGMAS] + [f"K_{k}" for k in ABL_K]
    v += [f"skip_{k}" for k in ABL_SKIP] + [f"pool_{p}" for p in ABL_POOL] + ["rule_fused_p95"]
    return [(x, 10) for x in v]


ADINO_N = args.ns or [1, 2, 4]
ADINO_VARIANTS = ["adino_rot"] if args.pixel else ["adino_rot", "adino_norot"]


def adino_variants():
    return [(v, n) for n in ADINO_N for v in ADINO_VARIANTS]


AUG_N = [1, 2, 4]
AUG_VARIANTS = ["galad", "galad_hflip", "galad_d4"]


def augment_variants():
    # galad_k3: no augmentation, K forced to 3 at N=1, to separate the augmentation
    # effect from the automatic K=3 -> K=2 reduction that GaLADConfig applies to one image.
    return [(v, n) for n in AUG_N for v in AUG_VARIANTS] + [("galad_k3", 1)]


def stage_spec():
    if args.stage == "ablation":
        return ABLATION_CATS, ablation_variants()
    if args.stage == "adino":
        return LIT_CATS, adino_variants()
    return LIT_CATS, augment_variants()


# =============================================================================
# Per-category runners
# =============================================================================

def run_ablation(ds, bb, todo):
    train_all, test_info = G.load_dataset(ds)
    labels = np.array([t["label"] for t in test_info])
    layers = [-1, -6, -12]
    test_feats = bb.extract([t["path"] for t in test_info], layers)
    for seed in SEEDS:
        need = [v for v, n in todo if (v, n, seed) in todo_set(ds, todo)]
        if not need:
            continue
        t0 = time.time()
        tr = bb.extract(open_rgb(sample_train(train_all, 10, seed)), layers)
        base = {l: fit_galad(tr, [l], seed=seed) for l in layers}
        maps = {}
        for l in layers:
            maps.update(raw_maps(base[l], test_feats, bb.grid))
        rows = {}

        def emit(variant, scores, pmaps=None):
            if (variant, 10, seed) not in todo_set(ds, todo):
                return
            au, ap = metrics(labels, scores)
            pro = auspro_448(pmaps, test_info) if pmaps is not None else None
            rows[variant] = (au, ap, pro)

        for name, ls in ABL_LAYER_SETS.items():
            m = smooth({l: maps[l] for l in ls}, 1)
            emit(name, image_scores(m), fused_map(m))
        d_raw = {l: maps[l] for l in (-6, -12)}
        for s in ABL_SIGMAS:
            m = smooth(d_raw, s)
            emit(f"sigma_{s}", image_scores(m), fused_map(m))
        d1 = smooth(d_raw, 1)
        for p in ABL_POOL:
            emit(f"pool_{p}", image_scores(d1, p))
        emit("rule_fused_p95", image_scores(d1, "p95", rule="fused"))
        for k in ABL_K:
            if (f"K_{k}", 10, seed) in todo_set(ds, todo):
                m = {}
                for l in (-6, -12):
                    m.update(raw_maps(fit_galad(tr, [l], n_comp=k, seed=seed), test_feats, bb.grid))
                emit(f"K_{k}", image_scores(smooth(m, 1)))
        for k in ABL_SKIP:
            if (f"skip_{k}", 10, seed) in todo_set(ds, todo):
                m = {}
                for l in (-6, -12):
                    m.update(raw_maps(fit_galad(tr, [l], skip=k, seed=seed), test_feats, bb.grid))
                emit(f"skip_{k}", image_scores(smooth(m, 1)))
        secs = time.time() - t0
        for v, (au, ap, pro) in rows.items():
            save_row(dict(variant=v, n_train=10, seed=seed, img_auroc=fmt(au), img_ap=fmt(ap),
                          auspro=fmt(pro), seconds=f"{secs:.1f}", dataset=ds))
        log(f"  {ds} seed {seed}: {len(rows)} rows, default AUROC "
            f"{rows.get('default', (float('nan'),))[0]:.4f}, {secs:.0f}s")


def knn_scores(ref, test, chunk=4096):
    """Cosine distance to the nearest reference patch, per test patch."""
    ref = torch.nn.functional.normalize(ref.to(torch.float32), dim=-1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = ref.to(dev)
    out = []
    for img in test:
        q = torch.nn.functional.normalize(img.to(dev, torch.float32), dim=-1)
        d = []
        for i in range(0, q.shape[0], chunk):
            d.append(1 - (q[i:i + chunk] @ ref.T).max(1).values)
        out.append(torch.cat(d).cpu().numpy())
    return np.stack(out)


def run_adino(ds, bb, todo):
    train_all, test_info = G.load_dataset(ds)
    labels = np.array([t["label"] for t in test_info])
    test = bb.extract([t["path"] for t in test_info], ["last"])["last"]
    has_px = any(e.get("mask_path") or e.get("mask_dir") for e in test_info)
    gts = masks_448(test_info) if (args.pixel and has_px) else None
    for n in ADINO_N:
        for seed in SEEDS:
            ts = todo_set(ds, todo)
            if not any((v, n, seed) in ts for v in ADINO_VARIANTS):
                continue
            imgs = open_rgb(sample_train(train_all, n, seed))
            for v in ADINO_VARIANTS:
                if (v, n, seed) not in ts:
                    continue
                t0 = time.time()
                refs = augment(imgs, rot45) if v == "adino_rot" else imgs
                ref = bb.extract(refs, ["last"])["last"].reshape(-1, test.shape[-1])
                d = knn_scores(ref, test)
                au, ap = metrics(labels, pool(d, "top1pct"))
                pro = None
                if args.pixel and has_px:
                    pro = fast_auspro(adino_maps_448(d, bb.grid), gts, [int(t.get("label", 0)) for t in test_info])[1]
                save_row(dict(variant=v, n_train=n, seed=seed, img_auroc=fmt(au), img_ap=fmt(ap),
                              auspro=fmt(pro), seconds=f"{time.time() - t0:.1f}", dataset=ds))
            log(f"  {ds} N={n} seed {seed} done")


def run_augment(ds, bb, todo):
    train_all, test_info = G.load_dataset(ds)
    labels = np.array([t["label"] for t in test_info])
    layers = [-6, -12]
    test_feats = bb.extract([t["path"] for t in test_info], layers)
    fns = {"galad": None, "galad_hflip": hflip, "galad_d4": dihedral, "galad_k3": None}
    for n in AUG_N:
        for seed in SEEDS:
            ts = todo_set(ds, todo)
            imgs = None
            for v, fn in fns.items():
                if (v, n, seed) not in ts:
                    continue
                imgs = imgs or open_rgb(sample_train(train_all, n, seed))
                t0 = time.time()
                tr = bb.extract(augment(imgs, fn) if fn else imgs, layers)
                model = fit_galad(tr, layers, seed=seed, force_k=(v == "galad_k3"))
                m = smooth(raw_maps(model, test_feats, bb.grid), 1)
                au, ap = metrics(labels, image_scores(m))
                pro = auspro_448(fused_map(m), test_info)
                save_row(dict(variant=v, n_train=n, seed=seed, img_auroc=fmt(au), img_ap=fmt(ap),
                              auspro=fmt(pro), seconds=f"{time.time() - t0:.1f}", dataset=ds))
            log(f"  {ds} N={n} seed {seed} done")


_TODO_CACHE = {}


def todo_set(ds, todo):
    """Set of (variant, n, seed) still missing for this dataset (refreshed per call)."""
    done = _TODO_CACHE.setdefault("done", done_keys())
    return {(v, n, s) for v, n in todo for s in SEEDS if (ds, v, n, s) not in done}


# =============================================================================
# Main
# =============================================================================

def main():
    cats, variants = stage_spec()
    if args.datasets:
        cats = args.datasets
    done = done_keys()
    left = {ds: sum((ds, v, n, s) not in done for v, n in variants for s in SEEDS) for ds in cats}
    total = len(cats) * len(variants) * len(SEEDS)
    log(f"Stage {args.stage} ({args.backbone if args.stage == 'adino' else 'dinov3l'}): "
        f"{total - sum(left.values())}/{total} rows done -> {csv_path()}")
    for ds, k in left.items():
        if k:
            log(f"  pending {ds}: {k} rows")
    if args.plan or not any(left.values()):
        return

    bb = Backbone(args.backbone if args.stage == "adino" else "dinov3l")
    runner = {"ablation": run_ablation, "adino": run_adino, "augment": run_augment}[args.stage]
    t_start = time.time()
    for i, ds in enumerate(cats):
        if not left[ds]:
            continue
        if not Path(f"data/{ds}").exists():
            log(f"MISSING data/{ds}, skipped")
            continue
        log(f"[{i + 1}/{len(cats)}] {ds}")
        try:
            runner(ds, bb, variants)
        except Exception as e:  # keep going; the row stays pending for the next run
            log(f"ERROR {ds}: {type(e).__name__}: {e}")
        _TODO_CACHE.clear()
        torch.cuda.empty_cache()
        elapsed = (time.time() - t_start) / 60
        log(f"  elapsed {elapsed:.1f} min")
    log("Stage finished.")


if __name__ == "__main__":
    main()
