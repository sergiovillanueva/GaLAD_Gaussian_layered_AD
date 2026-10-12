"""Server evaluation of the frozen GaLAD against AnomalyDINO on the same backbone
(IMPROVEMENT_PLAN.md section 28). One fixed method in several settings; nothing is tuned here.

GaLAD (frozen in section 27.4): references augmented with eight 45-degree rotations; two hidden
layers at 1/4 and 1/2 of the depth from the top (-6/-12 for ViT-L); PCA to 256 dims skipping 2;
Gaussian mixture with K=12 and covariance shrinkage kappa=2000; patch score = chi-square tail of
the smallest Mahalanobis distance; Gaussian smoothing sigma=1; image score = mean over layers of
the 95th percentile; pixel map = mean of the smoothed layer maps, bilinear to pixels.
AnomalyDINO (matched): references rotated at original resolution, last layer, cosine 1-NN, mean of
the top 1%; maps bilinear + Gaussian sigma=4.

Settings: --backbone dinov3s|dinov3b|dinov3l|dinov3h|dinov2s|dinov2l|dinov2rs|dinov2rl (DINOv2: patch 14, grid 32x32;
'r' = with 4 register tokens); --resize square (448x448) | aspect (shorter
side 448, aspect ratio kept, cropped to a multiple of the patch size; categories whose images are
all square are skipped because the result equals square); --mask none | official (AnomalyDINO's
PCA background mask computed with DINOv2-S, only on the categories its code masks, applied to both
methods: masked patches score 0).
knnw / knnwc: the same control with whitened PCA coordinates, without / with GaLAD's chi-square tail.
knn (control): 1-NN on GaLAD's own representation (same rotated references, layers and PCA of the
fitted GaLAD), Euclidean distance to the nearest reference patch, then GaLAD's smoothing and pooling.
--perturb: one benign change applied to every test image after resizing (and, for shift, to its mask):
bright/dark (brightness x1.3 / x0.7), blur (Gaussian, radius 1.5 px), noise (Gaussian, std 10 grey
levels, seeded per image), jpeg (quality 30), shift (8 px right and down, half a patch, reflected border).
subspace (comparator, SubspaceAD, Lendering et al., CVPR 2026, re-implemented from the paper on our
backbone): patch features averaged over the blocks at 55-70% of the depth (22-28 of the 40 blocks of
DINOv2-G; 13-17 for ViT-L), each reference rotated by 30 random angles in [0, 345] degrees (seeded; the
paper skips rotations for transistor, here every category is treated alike), PCA keeping 99% of the
variance, patch score = squared residual outside that subspace, image score = mean of the top 1% of
patches (TVaR), map = bilinear to the input size + Gaussian sigma = 4.
galadm (exploratory): the same fitted GaLAD, but the patch score is the smallest squared Mahalanobis distance
(no chi-square tail), same smoothing and pooling.
galadb (exploratory): GaLAD fitted on the SubspaceAD layer band (features averaged over the blocks at
55-70% of the depth) instead of the two layers; same rotations, PCA, mixture, score and post-processing.
--res: input side for square resizing (default 448); with another value, every pixel map is resized to
448 x 448 (area interpolation) before the pixel metrics, so the ground truth is the same.
--level sets the severity; scale (zoom about the image centre, reflected borders) and rotate (about the centre,
reflected borders) also transform the ground-truth mask, as shift does.
--save-maps: per-image patch-grid maps of every method (GaLAD/knn: fused smoothed map; AnomalyDINO: raw
distances) in output/server/scores/<tag>/<ds>/maps_N{n}_s{seed}_{method}.npz.
Metrics: image AP and AUROC; standard AU-PRO up to FPR 0.3, 0.05 and 0.01, pixel AUROC and AP.
Rows: output/server/<tag>.csv; per-image scores: output/server/scores/<tag>/<ds>/.
--bench: efficiency instead of accuracy (seed 0, 50 test images, batch 1): output/server/bench_<tag>.csv.

Usage: uv run server_eval.py --tag l_square --backbone dinov3l --datasets VisA/pcb1 --ns 1 --seeds 0
"""

import argparse
import sys

BACKBONES = {"dinov3s": "facebook/dinov3-vits16-pretrain-lvd1689m",
             "dinov3b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
             "dinov3l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
             "dinov3h": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
             "dinov2s": "facebook/dinov2-small", "dinov2l": "facebook/dinov2-large",
             "dinov2rs": "facebook/dinov2-with-registers-small", "dinov2rl": "facebook/dinov2-with-registers-large"}

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--backbone", default="dinov3l", choices=list(BACKBONES))
p.add_argument("--resize", default="square", choices=["square", "aspect"])
p.add_argument("--mask", default="none", choices=["none", "official"])
p.add_argument("--datasets", nargs="+", required=True)
p.add_argument("--ns", nargs="+", type=int, default=[1, 2, 4, 5])
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
p.add_argument("--methods", nargs="+", default=["galad", "adino"], choices=["galad", "adino", "knn", "knnw", "knnwc", "subspace", "galadb", "galadm"])
p.add_argument("--perturb", default="none",
               choices=["none", "bright", "dark", "blur", "noise", "jpeg", "shift", "scale", "rotate"])
p.add_argument("--ref-gain", type=float, nargs="+", default=None,
               help="Brightness gain of each reference image (cycled), applied before the rotations; all methods.")
p.add_argument("--level", type=float, default=None,
               help="Severity of --perturb (default = the original single level): bright/dark brightness change "
                    "(0.3 = x1.3 / x0.7), blur radius px, noise std, jpeg quality, shift px, scale factor, rotate degrees.")
p.add_argument("--save-maps", action="store_true")
p.add_argument("--res", type=int, default=448, help="Input side (square only); maps are evaluated at 448.")
p.add_argument("--no-pixel", action="store_true", help="Image metrics only.")
p.add_argument("--pool", default="p95", choices=["p95", "p99", "max", "top1pct"],
               help="GaLAD image score: this statistic of each smoothed layer map, averaged over layers. "
                    "Every choice is also stored per layer in the score files.")
p.add_argument("--bench", action="store_true", help="Measure efficiency instead of accuracy.")
p.add_argument("--threads", type=int, default=4)
a = p.parse_args()

_argv, sys.argv = sys.argv, [sys.argv[0], "--datasets", "none", "--threads", str(a.threads)]
import gauss_screen as GS  # noqa: E402  (frozen GMM code: PCA + shrinkage EM + chi-square tail)
if a.mask == "official":
    import adino_official as AO  # noqa: E402  (official background mask, DINOv2-S)
sys.argv = _argv
E, G = GS.E, GS.G

import csv  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from scipy.ndimage import gaussian_filter, zoom  # noqa: E402

from pixel_metrics_std import mask_regions, std_pixel_metrics_binned  # noqa: E402

RES, K, KAPPA = a.res, 12, 2000.0
assert RES == 448 or a.resize == "square", "--res other than 448 needs --resize square"
DEVICE = GS.DEV
OUT = Path("output/server")
CSV = OUT / f"{a.tag}.csv"
SCORES = OUT / "scores" / a.tag
FIELDS = ["timestamp", "tag", "backbone", "resize", "mask", "dataset", "method", "n_train", "seed",
          "img_ap", "img_auroc", "aupro_30", "aupro_05", "pix_auroc", "pix_ap", "k_used", "fit_s", "aupro_01", "perturb"]
BENCH = OUT / f"bench_{a.tag}.csv"
BFIELDS = ["backbone", "resize", "dataset", "method", "n_train", "state_mib", "ref_extract_s", "fit_s",
           "forward_ms", "score_ms", "score_ms_fp32", "fp32_max_rel_diff", "peak_gpu_mib", "gpu"]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Net:
    def __init__(self, kind):
        from transformers import AutoImageProcessor, AutoModel
        name = BACKBONES[kind]
        frac = os.environ.get("GALAD_GPU_FRACTION")
        if frac and DEVICE.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(float(frac))
        try:  # local cache first: DINOv3 is gated and the server may have no Hugging Face login
            self.processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=True)
        except OSError:
            self.processor = AutoImageProcessor.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True)
        self.model = self.model.to(DEVICE).eval()
        cfg = self.model.config
        self.patch, self.n_reg = cfg.patch_size, getattr(cfg, "num_register_tokens", 0)
        depth = cfg.num_hidden_layers
        self.layers = [-(depth // 4), -(depth // 2)]  # -6/-12 for the 24 blocks of ViT-L
        # SubspaceAD layer band (hidden_states index = block number): 22-28 of 40 -> 13-17 of 24
        self.sub_layers = list(range(round(0.55 * depth), round(0.70 * depth) + 1)) if {"subspace", "galadb"} & set(a.methods) else []
        log(f"Backbone {name}: {depth} blocks, layers {self.layers}, patch {self.patch}, dim {cfg.hidden_size}")

    def prep(self, img):
        if a.resize == "square":
            return img.resize((RES, RES), Image.BICUBIC)
        w, h = img.size
        s = RES / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = img.size
        return img.crop((0, 0, w - w % self.patch, h - h % self.patch))

    def extract(self, imgs, dtype=torch.float16, batch_size=None):
        """Prepared PIL images (any sizes) -> list of ({layer: [P, D] cpu, 'last': [P, D]}, (gh, gw))."""
        bs = batch_size or (16 if self.model.config.hidden_size <= 1024 else 4)
        out, groups = [None] * len(imgs), {}
        for i, im in enumerate(imgs):
            groups.setdefault(im.size, []).append(i)
        start = 1 + self.n_reg
        for (w, h), idx in groups.items():
            gh, gw = h // self.patch, w // self.patch
            j = 0
            while j < len(idx):
                part = idx[j:j + bs]
                try:
                    inputs = self.processor(images=[imgs[i] for i in part], return_tensors="pt",
                                            do_resize=False, do_center_crop=False).to(DEVICE)
                    with torch.inference_mode(), torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                        o = self.model(**inputs)
                except torch.OutOfMemoryError:
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)  # retry the same images with a smaller batch
                    torch.cuda.empty_cache()
                    log(f"  out of GPU memory: batch size -> {bs}")
                    continue
                for t, i in enumerate(part):
                    f = {l: (o.last_hidden_state if l == "last" else o.hidden_states[l])[t, start:start + gh * gw]
                         .float().cpu().to(dtype) for l in self.layers + ["last"]}
                    if self.sub_layers:
                        f["sub"] = torch.stack([o.hidden_states[l][t, start:start + gh * gw].float()
                                                for l in self.sub_layers]).mean(0).cpu().to(dtype)
                    out[i] = (f, (gh, gw))
                del o, inputs
                j += len(part)
        return out


def all_square(paths):
    for q in paths:
        with Image.open(q) as im:
            if im.size[0] != im.size[1]:
                return False
    return True


def eval_masks(test_info, sizes):
    """Ground truth at the evaluation size of each prepared test image."""
    if a.resize == "square":
        return E.masks_448(test_info)
    out = []
    for e, (orig, (w, h)) in zip(test_info, sizes):
        m = G.load_mask(e)
        if m is None:
            out.append(np.zeros((h, w), np.uint8))
            continue
        s = RES / min(orig)
        m = cv2.resize((m > 0).astype(np.uint8), (round(orig[0] * s), round(orig[1] * s)), interpolation=cv2.INTER_NEAREST)
        out.append(m[:h, :w])
    return out


def background_keep(paths, grids, dino2):
    """Official AnomalyDINO foreground mask (DINOv2-S, threshold 10) mapped to our patch grids."""
    keep = []
    for q, (gh, gw) in zip(paths, grids):
        tok, grid2 = dino2.tokens(Image.open(q).convert("RGB"))
        m = AO.background_mask(tok, grid2).reshape(grid2).astype(np.float32)
        keep.append(cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA).ravel() >= 0.5)
    return keep


CENTER = os.environ.get("GAUSS_CENTER") == "1"  # exploratory: subtract each image's mean patch token
POOL3 = os.environ.get("GAUSS_POOL3") == "1"    # exploratory: 3x3 mean of the patch tokens before PCA


def prep_tokens(x, grid):
    """Exploratory per-image transforms of one image's [P, D] tokens (GaLAD and knn only)."""
    if not (CENTER or POOL3):
        return x
    x = x.float()
    if CENTER:
        x = x - x.mean(0, keepdim=True)
    if POOL3:
        gh, gw = grid
        g = x.T.reshape(1, -1, gh, gw)
        x = F.avg_pool2d(g, 3, stride=1, padding=1, count_include_pad=False).reshape(x.shape[1], -1).T
    return x


def galad_fit(views, net, seed, layers=None):
    t0 = time.time()
    models = [GS.Layer(torch.cat([prep_tokens(f[l], g).float() for f, g in views]).to(DEVICE), K, KAPPA, seed)
              for l in layers or net.layers]
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return models, time.time() - t0


def galad_score(models, test, net, keep=None, layers=None, chi2=True):
    """Image scores and per-image fused maps (patch grid) for the frozen GaLAD."""
    layer_maps = []
    for m, l in zip(models, layers or net.layers):
        d2, _ = m.stats(torch.cat([prep_tokens(f[l], g) for f, g in test]).to(DEVICE))
        s = -GS.chi2_logsf(d2.numpy().min(1), GS.DIM) if chi2 else d2.numpy().min(1)
        maps, i = [], 0
        for t, (_, (gh, gw)) in enumerate(test):
            x = s[i:i + gh * gw].copy()
            if keep is not None:
                x[~keep[t]] = 0.0
            maps.append(gaussian_filter(x.reshape(gh, gw), sigma=1))
            i += gh * gw
        layer_maps.append(maps)
    per = {f"{pl}_L{j}": np.array([E.pool(x[None], pl)[0] for x in maps]) for pl in ("p95", "p99", "max", "top1pct")
           for j, maps in enumerate(layer_maps)}
    scores = np.mean([per[f"{a.pool}_L{j}"] for j in range(len(layer_maps))], 0)
    fused = [np.mean([lm[t] for lm in layer_maps], 0) for t in range(len(test))]
    return scores, fused, per


def galad_score_fp32(models, test, net):
    """Same image scores computed in float32 (fitting stays in float64); used only by --bench."""
    layer_maps = []
    for m, l in zip(models, net.layers):
        mean, comp = m.mean.float(), m.comp.float()
        mu, chol = m.gmm.means.float(), m.gmm.chol.float()
        z = (torch.cat([f[l] for f, _ in test]).to(DEVICE).float() - mean) @ comp
        y = torch.linalg.solve_triangular(chol, (z[None] - mu[:, None]).transpose(1, 2), upper=False)
        s = -GS.chi2_logsf((y ** 2).sum(1).min(0).values.double().cpu().numpy(), GS.DIM)
        maps, i = [], 0
        for _, (gh, gw) in test:
            maps.append(gaussian_filter(s[i:i + gh * gw].reshape(gh, gw), sigma=1))
            i += gh * gw
        layer_maps.append(maps)
    return np.mean([[E.pool(x[None], a.pool)[0] for x in maps] for maps in layer_maps], 0)


def adino_score(bank, test, keep=None):
    scores, grids = [], []
    for t, (f, (gh, gw)) in enumerate(test):
        q = F.normalize(f["last"].float().to(DEVICE), dim=-1)
        d = (1 - (q @ bank.T).max(1).values).cpu().numpy()
        if keep is not None:
            d[~keep[t]] = 0.0
        k = max(1, int(0.01 * d.size))  # floor, as AnomalyDINO's mean_top1p
        scores.append(np.sort(d)[-k:].mean())
        grids.append(d.reshape(gh, gw))
    return np.array(scores), grids


LEVEL = {"bright": 0.3, "dark": 0.3, "blur": 1.5, "noise": 10, "jpeg": 30, "shift": 8, "scale": 1.1, "rotate": 10}


def lev():
    return LEVEL.get(a.perturb) if a.level is None else a.level


def geom_matrix(w, h):
    """Affine matrix of the geometric perturbations (None for photometric ones)."""
    if a.perturb == "shift":
        return np.float32([[1, 0, lev()], [0, 1, lev()]])
    if a.perturb == "scale":
        return cv2.getRotationMatrix2D((w / 2, h / 2), 0, lev())
    if a.perturb == "rotate":
        return cv2.getRotationMatrix2D((w / 2, h / 2), lev(), 1.0)
    return None


def perturb_image(img, i):
    """Benign test-time change (--perturb, --level) of a prepared PIL image; i seeds the noise."""
    if a.perturb in ("bright", "dark"):
        return ImageEnhance.Brightness(img).enhance(1 + lev() if a.perturb == "bright" else 1 - lev())
    if a.perturb == "blur":
        return img.filter(ImageFilter.GaussianBlur(lev()))
    if a.perturb == "noise":
        x = np.asarray(img, np.float32) + np.random.default_rng(i).normal(0, lev(), (img.size[1], img.size[0], 3))
        return Image.fromarray(np.clip(np.round(x), 0, 255).astype(np.uint8))
    if a.perturb == "jpeg":
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(lev()))
        return Image.open(buf).convert("RGB")
    x = np.asarray(img)
    m = geom_matrix(x.shape[1], x.shape[0])
    if m is not None:
        return Image.fromarray(cv2.warpAffine(x, m, (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT_101))
    return img


def perturb_mask(g):
    m = geom_matrix(g.shape[1], g.shape[0])
    if m is None:
        return g
    return cv2.warpAffine(g.astype(np.uint8), m, (g.shape[1], g.shape[0]), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def knn_score(models, ref, test, net, keep=None, whiten=False, chi2=False):
    """1-NN on GaLAD's representation: Euclidean distance in each layer's PCA space, then GaLAD's post-processing.
    whiten: coordinates divided by the PCA standard deviations (Mahalanobis distance under the global covariance);
    chi2: patch score -log chi2_256 tail of the squared distance, as in GaLAD."""
    layer_maps = []
    for m, l in zip(models, net.layers):
        w = m.evals.rsqrt().float().to(DEVICE) if whiten else 1.0
        bank = m.project(torch.cat([f[l].float() for f, _ in ref]).to(DEVICE)).float() * w
        x = torch.cat([f[l] for f, _ in test])  # stays on the CPU; projected 2048 patches at a time
        s = torch.cat([torch.cdist(m.project(x[j:j + 2048].to(DEVICE)).float() * w, bank).min(1).values.cpu()
                       for j in range(0, x.shape[0], 2048)]).numpy().astype(np.float64)
        if chi2:
            s = -GS.chi2_logsf(s ** 2, GS.DIM)
        del bank, x
        maps, i = [], 0
        for t, (_, (gh, gw)) in enumerate(test):
            x = s[i:i + gh * gw].astype(np.float64)
            if keep is not None:
                x[~keep[t]] = 0.0
            maps.append(gaussian_filter(x.reshape(gh, gw), sigma=1))
            i += gh * gw
        layer_maps.append(maps)
    scores = np.mean([[E.pool(x[None], a.pool)[0] for x in maps] for maps in layer_maps], 0)
    fused = [np.mean([lm[t] for lm in layer_maps], 0) for t in range(len(test))]
    return scores, fused


def pixel_metrics(maps, masks, regions):
    pro, pau, pap = std_pixel_metrics_binned(maps, masks, regions, limit=(0.3, 0.05, 0.01))
    return (None,) * 5 if pro is None else (pro[0], pro[1], pau, pap, pro[2])


def done():
    if not CSV.exists():
        return set()
    with open(CSV, newline="") as f:
        return {(r["dataset"], r["method"], int(r["n_train"]), int(r["seed"])) for r in csv.DictReader(f)}


def save(row):
    new = not CSV.exists()
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def subspace_views(net, paths, seed):
    """SubspaceAD references: each prepared image plus 30 rotations by random angles in [0, 345] (reflected borders)."""
    rng = np.random.default_rng(seed)
    out = []
    for q in paths:
        x = np.asarray(net.prep(Image.open(q).convert("RGB")))
        h, w = x.shape[:2]
        out.append(Image.fromarray(x))
        for ang in rng.uniform(0, 345, 30):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), 1.0)
            out.append(Image.fromarray(cv2.warpAffine(x, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)))
    return out


def subspace_fit_score(ref, test, var=0.99):
    """PCA subspace of the reference patches (99% variance); squared residual per test patch."""
    X = torch.cat([f["sub"].float() for f, _ in ref]).to(DEVICE).double()
    mu = X.mean(0)
    X -= mu
    evals, evecs = torch.linalg.eigh(X.T @ X / (X.shape[0] - 1))
    del X
    evals, evecs = evals.flip(0), evecs.flip(1)
    r = int(torch.searchsorted(torch.cumsum(evals, 0) / evals.sum(), torch.tensor(var, dtype=evals.dtype, device=evals.device))) + 1
    C = evecs[:, :r].float()
    scores, grids = [], []
    for f, (gh, gw) in test:
        z = f["sub"].float().to(DEVICE) - mu.float()
        res = ((z - (z @ C) @ C.T) ** 2).sum(1).cpu().numpy()
        k = max(1, int(0.01 * res.size))
        scores.append(np.sort(res)[-k:].mean())
        grids.append(res.reshape(gh, gw))
    return np.array(scores), grids, r


def refs_for(net, paths):
    """Rotated reference views: GaLAD rotates the prepared image, AnomalyDINO the original one."""
    imgs = [Image.open(q).convert("RGB") for q in paths]
    if a.ref_gain:  # reference-diversity experiment: brightness gain of reference i = ref_gain[i % len]
        imgs = [ImageEnhance.Brightness(im).enhance(a.ref_gain[i % len(a.ref_gain)]) for i, im in enumerate(imgs)]
    galad = [v for im in imgs for v in E.rot45(net.prep(im))]
    adino = [net.prep(v) for im in imgs for v in E.rot45(im)]
    return galad, adino


def run_category(net, ds, dino2, fin):
    train_all, test_info = G.load_dataset(ds)
    paths = [t["path"] for t in test_info]
    if a.resize == "aspect" and all_square(paths + list(train_all)):
        log(f"skip {ds}: all images square (aspect = square)")
        return
    todo = [(n, s, m) for n in a.ns for s in a.seeds for m in a.methods if (ds, m, n, s) not in fin]
    if not todo:
        return
    labels = np.array([int(t["label"]) for t in test_info])
    sizes, prepped = [], []
    for i, q in enumerate(paths):
        im = Image.open(q).convert("RGB")
        pi = perturb_image(net.prep(im), i)
        sizes.append((im.size, pi.size))
        prepped.append(pi)
    test = net.extract(prepped)
    del prepped
    keep = None
    if a.mask == "official" and AO.masking_for(ds):
        keep = background_keep(paths, [g for _, g in test], dino2)
    gts = regions = None
    if not a.no_pixel:
        gts = [perturb_mask(g) for g in eval_masks(test_info, sizes)]
        regions = mask_regions(gts)
    sd = SCORES / ds.replace("/", "__")
    sd.mkdir(parents=True, exist_ok=True)
    for n, seed in sorted({(n, s) for n, s, _ in todo}):
        methods = [m for nn, ss, m in todo if (nn, ss) == (n, seed)]
        g_views, a_views = refs_for(net, E.sample_train(train_all, n, seed))
        fitted = {}  # GaLAD fit shared by galad and knn
        for method in sorted(methods, key=lambda m: m != "galad"):
            fit_s, k_used, per = 0.0, "", {}
            if method in ("galad", "knn", "knnw", "knnwc", "galadm") and not fitted:
                ref = net.extract(g_views, dtype=torch.float32)
                fitted["models"], fitted["fit_s"] = galad_fit(ref, net, seed)
                fitted["ref"] = ref
            if method == "galad":
                models, fit_s = fitted["models"], fitted["fit_s"]
                k_used = "/".join(str(m.k) for m in models)
                scores, grids, per = galad_score(models, test, net, keep)
                maps = [zoom(x, net.patch, order=1) for x in grids]
            elif method in ("knn", "knnw", "knnwc"):
                scores, grids = knn_score(fitted["models"], fitted["ref"], test, net, keep,
                                          whiten=method != "knn", chi2=method == "knnwc")
                maps = [zoom(x, net.patch, order=1) for x in grids]
            elif method == "galadm":
                scores, grids, per = galad_score(fitted["models"], test, net, keep, chi2=False)
                maps = [zoom(x, net.patch, order=1) for x in grids]
            elif method == "galadb":
                ref = fitted.get("ref") or net.extract(g_views, dtype=torch.float32)
                models, fit_s = galad_fit(ref, net, seed, layers=["sub"])
                k_used = str(models[0].k)
                scores, grids, per = galad_score(models, test, net, keep, layers=["sub"])
                maps = [zoom(x, net.patch, order=1) for x in grids]
                del models
            elif method == "subspace":
                t0 = time.time()
                sref = net.extract(subspace_views(net, E.sample_train(train_all, n, seed), seed), dtype=torch.float32)
                scores, grids, r = subspace_fit_score(sref, test)
                fit_s, k_used = time.time() - t0, f"r={r}"
                maps = [gaussian_filter(cv2.resize(x.astype(np.float32), (x.shape[1] * net.patch, x.shape[0] * net.patch),
                                                   interpolation=cv2.INTER_LINEAR), sigma=4) for x in grids]
                del sref
            else:
                feats = net.extract(a_views, dtype=torch.float32)
                bank = F.normalize(torch.cat([f["last"] for f, _ in feats]).to(DEVICE), dim=-1)
                scores, grids = adino_score(bank, test, keep)
                maps = [gaussian_filter(cv2.resize(x.astype(np.float32), (x.shape[1] * net.patch, x.shape[0] * net.patch),
                                                   interpolation=cv2.INTER_LINEAR), sigma=4) for x in grids]
                del bank, feats
            au, ap = E.metrics(labels, scores)
            if RES != 448:
                maps = [cv2.resize(np.asarray(m, np.float32), (448, 448), interpolation=cv2.INTER_AREA) for m in maps]
            pro30 = pro05 = pau = pap = pro01 = None
            if not a.no_pixel and any(g.any() for g in gts):
                pro30, pro05, pau, pap, pro01 = pixel_metrics(maps, gts, regions)
            if a.save_maps:
                np.savez_compressed(sd / f"maps_N{n}_s{seed}_{method}.npz",
                                    maps=np.stack([np.asarray(x, np.float32) for x in grids]))
            np.savez_compressed(sd / f"N{n}_s{seed}_{method}.npz", test=np.asarray(scores, np.float32), labels=labels,
                                **{k: v.astype(np.float32) for k, v in per.items()})
            save(dict(timestamp=f"{datetime.now():%Y-%m-%d %H:%M:%S}", tag=a.tag, backbone=a.backbone, resize=a.resize,
                      mask=a.mask, dataset=ds, method=method, n_train=n, seed=seed, img_ap=f"{ap:.4f}",
                      img_auroc=f"{au:.4f}", aupro_30=E.fmt(pro30), aupro_05=E.fmt(pro05), pix_auroc=E.fmt(pau),
                      pix_ap=E.fmt(pap), k_used=k_used, fit_s=f"{fit_s:.2f}", aupro_01=E.fmt(pro01), perturb=a.perturb if a.perturb == "none" else f"{a.perturb}{lev():g}"))
            log(f"  {a.tag} {ds} N={n} s{seed} {method}: AP {ap:.4f} AUROC {au:.4f} AU-PRO30 {E.fmt(pro30)}")
            del maps, grids
            torch.cuda.empty_cache()
        del fitted


def state_mib(models, tri=True):
    """float32 size of the GaLAD state: PCA mean + components, mixture weights, means, Cholesky factors."""
    n = 0
    for m in models:
        d, k = m.comp.shape[1], m.k
        n += m.mean.numel() + m.comp.numel() + k + k * d + k * (d * (d + 1) // 2 if tri else d * d)
    return 4 * n / 2 ** 20


def bench_category(net, ds):
    train_all, test_info = G.load_dataset(ds)
    imgs = [net.prep(Image.open(t["path"]).convert("RGB")) for t in test_info[:50]]
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"

    def timed(fn):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return r, time.perf_counter() - t0

    for _ in range(3):
        net.extract(imgs[:1])  # warm-up
    test, fw = [], []
    for im in imgs:
        r, t = timed(lambda: net.extract([im], batch_size=1))
        test += r
        fw.append(t * 1000)
    have = set()
    if BENCH.exists():
        with open(BENCH, newline="") as f:
            have = {(r["dataset"], r["method"], int(r["n_train"])) for r in csv.DictReader(f)}
    for n in a.ns:
        g_views, a_views = refs_for(net, E.sample_train(train_all, n, 0))
        for method in a.methods:
            if (ds, method, n) in have:
                continue
            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            views, ext = timed(lambda: net.extract(g_views if method == "galad" else a_views, dtype=torch.float32))
            if method == "galad":
                (models, _), fit = timed(lambda: galad_fit(views, net, 0))
                state = state_mib(models)
                sc = [timed(lambda: galad_score(models, [x], net))[1] * 1000 for x in test]
                sc32 = [timed(lambda: galad_score_fp32(models, [x], net))[1] * 1000 for x in test]
                ref64 = galad_score(models, test, net)[0]
                diff = float(np.max(np.abs(galad_score_fp32(models, test, net) - ref64) / np.abs(ref64).clip(1e-9)))
            else:
                bank, fit = timed(lambda: F.normalize(torch.cat([f["last"] for f, _ in views]).to(DEVICE), dim=-1))
                state = bank.numel() * 4 / 2 ** 20
                sc = [timed(lambda: adino_score(bank, [x]))[1] * 1000 for x in test]
                sc32, diff = sc, 0.0
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else 0
            row = (dict(backbone=a.backbone, resize=a.resize, dataset=ds, method=method, n_train=n,
                             state_mib=f"{state:.2f}", ref_extract_s=f"{ext:.2f}", fit_s=f"{fit:.2f}",
                             forward_ms=f"{np.median(fw):.1f}", score_ms=f"{np.median(sc):.2f}",
                             score_ms_fp32=f"{np.median(sc32):.2f}", fp32_max_rel_diff=f"{diff:.2e}",
                             peak_gpu_mib=f"{peak:.0f}", gpu=gpu))
            log(f"  bench {ds} N={n} {method}: state {state:.2f} MiB fit {fit:.2f}s forward {np.median(fw):.1f}ms "
                f"score {np.median(sc):.2f}ms (fp32 {np.median(sc32):.2f}ms, max rel diff {diff:.1e})")
            new = not BENCH.exists()
            with open(BENCH, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=BFIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            del views
            models = bank = None
            torch.cuda.empty_cache()



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    net = Net(a.backbone)
    dino2 = AO.DinoV2S() if a.mask == "official" else None
    fin = done()
    for ds in a.datasets:
        try:
            if a.bench:
                bench_category(net, ds)
            else:
                run_category(net, ds, dino2, fin)
        except Exception as e:  # log and continue; rerunning fills the gap
            log(f"ERROR {ds}: {type(e).__name__}: {e}")
        torch.cuda.empty_cache()
    log(f"{a.tag} finished")


if __name__ == "__main__":
    main()
