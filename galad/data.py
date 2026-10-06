"""Dataset loading (MVTec AD layout, MVTec LOCO AD, VisA one-class split), reference draws and masks."""

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

EXT = (".png", ".jpg", ".jpeg", ".bmp")
RES = 448


def _images(folder):
    return sorted(str(p) for p in Path(folder).glob("*.*") if p.suffix.lower() in EXT)


def load_dataset(dataset, root="data"):
    """Returns (train_paths, test_info); test_info entries have 'path', 'label' and mask fields.
    `dataset` is '<benchmark>/<category>', e.g. 'mvtec_AD/bottle', 'VisA/pcb1', 'btad/01'."""
    path = Path(root) / dataset
    if not path.exists():
        raise FileNotFoundError(path)
    if dataset.startswith("VisA/"):
        return _load_visa(path)
    return _load_folder(path, loco=dataset.startswith("mvtec_loco_AD/"))


def _load_folder(path, loco):
    train = _images(path / "train" / "good")
    gt_dir, test = path / "ground_truth", []
    for sub in sorted(d for d in (path / "test").iterdir() if d.is_dir()):
        normal = sub.name == "good"
        for img in _images(sub):
            e = {"path": img, "label": 0 if normal else 1, "mask_path": None, "mask_dir": None}
            if not normal and gt_dir.exists():
                stem = Path(img).stem
                if loco:  # one folder of masks per image
                    d = gt_dir / sub.name / stem
                    if d.exists() and list(d.glob("*.png")):
                        e["mask_path"], e["mask_dir"] = str(next(d.glob("*.png"))), str(d)
                else:
                    for suffix in (".png", "_mask.png"):
                        m = gt_dir / sub.name / f"{stem}{suffix}"
                        if m.exists():
                            e["mask_path"] = str(m)
                            break
            test.append(e)
    return train, test


def _load_visa(path):
    root, train, test = path.parent, [], []
    with open(root / "split_csv" / "1cls.csv") as f:
        for row in csv.DictReader(f):
            if row["object"] != path.name or not (root / row["image"]).exists():
                continue
            img = str(root / row["image"])
            if row["split"] == "train":
                train.append(img)
            elif row["split"] == "test":
                mask = root / row["mask"] if row.get("mask", "").strip() else None
                test.append({"path": img, "label": 0 if row["label"] == "normal" else 1,
                             "mask_path": str(mask) if mask is not None and mask.exists() else None, "mask_dir": None})
    return train, test


def sample_references(train_paths, n, seed):
    """The reference draw of the paper: draw `seed` gives the same images for every method."""
    rng = random.Random(seed)
    if n and len(train_paths) > n:
        return sorted(rng.sample(sorted(train_paths), n))
    return list(train_paths)


def load_mask(entry):
    if not entry.get("mask_path"):
        return None
    if entry.get("mask_dir"):  # MVTec LOCO: union of all masks of the image
        out = None
        for m in Path(entry["mask_dir"]).glob("*.png"):
            a = np.array(Image.open(m).convert("L"))
            out = a if out is None else np.maximum(out, a)
        return out
    return np.array(Image.open(entry["mask_path"]).convert("L"))


def masks_448(test_info):
    """Binary ground-truth masks at the evaluation resolution (nearest-neighbor resize)."""
    out = []
    for e in test_info:
        m = load_mask(e)
        if m is None:
            out.append(np.zeros((RES, RES), np.uint8))
            continue
        g = (m > 0).astype(np.uint8)
        if g.shape != (RES, RES):
            g = (zoom(g.astype(float), (RES / g.shape[0], RES / g.shape[1]), order=0) > 0.5).astype(np.uint8)
        out.append(g)
    return out
