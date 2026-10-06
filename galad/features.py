"""Frozen DINOv3 patch features and the 45-degree rotations of the reference images."""

import cv2
import numpy as np
import torch
from PIL import Image

RES = 448
BACKBONES = {"dinov3s": "facebook/dinov3-vits16-pretrain-lvd1689m",
             "dinov3b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
             "dinov3l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
             "dinov3h": "facebook/dinov3-vith16plus-pretrain-lvd1689m"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numerical settings of the experiments in the paper (TF32 matmuls in float32, fp16 autocast forward).
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def rotations(img):
    """Eight views rotated by 0, 45, ..., 315 degrees (bilinear, reflected borders), as in AnomalyDINO."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    out = []
    for angle in range(0, 360, 45):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out.append(Image.fromarray(cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)))
    return out


class Backbone:
    """DINOv3 ViT; returns the patch tokens of two hidden layers (1/4 and 1/2 of the depth from the top,
    i.e. -6 and -12 for ViT-L/16) and of the last layer."""

    def __init__(self, kind="dinov3l"):
        from transformers import AutoImageProcessor, AutoModel
        name = BACKBONES[kind]
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, output_hidden_states=True).to(DEVICE).eval()
        cfg = self.model.config
        self.patch, self.n_reg = cfg.patch_size, getattr(cfg, "num_register_tokens", 0)
        self.layers = [-(cfg.num_hidden_layers // 4), -(cfg.num_hidden_layers // 2)]
        self.batch = 16 if cfg.hidden_size <= 1024 else 4

    @staticmethod
    def prep(img):
        return img.resize((RES, RES), Image.BICUBIC)

    def extract(self, imgs, dtype=torch.float16):
        """Prepared 448x448 PIL images -> list of {layer: [784, D] tensor on CPU, 'last': [784, D]}."""
        out, start, g = [], 1 + self.n_reg, RES // self.patch
        for i in range(0, len(imgs), self.batch):
            inputs = self.processor(images=imgs[i:i + self.batch], return_tensors="pt", do_resize=False,
                                    do_center_crop=False).to(DEVICE)
            with torch.inference_mode(), torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                o = self.model(**inputs)
            for t in range(len(imgs[i:i + self.batch])):
                out.append({l: (o.last_hidden_state if l == "last" else o.hidden_states[l])[t, start:start + g * g]
                            .float().cpu().to(dtype) for l in self.layers + ["last"]})
        return out

    def references(self, paths):
        """GaLAD rotates the resized image; AnomalyDINO rotates the original image and then resizes it."""
        imgs = [Image.open(p).convert("RGB") for p in paths]
        galad = [v for im in imgs for v in rotations(self.prep(im))]
        adino = [self.prep(v) for im in imgs for v in rotations(im)]
        return self.extract(galad, torch.float32), self.extract(adino, torch.float32)
