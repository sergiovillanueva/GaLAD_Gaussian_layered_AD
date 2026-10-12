"""Backbone time of DINOv3 ViT-L/16 at 448 x 448 on the laptop GPU, with the numerics of the evaluation (12 Oct).

  pipeline       server_eval Net.extract, batch 1: HF image processor on the CPU, autocast, features copied to the
                 CPU (the backbone time of Table 7)
  model, b1/b16  the model alone on GPU tensors, autocast (as in the evaluation), batch 1 and batches of 16 images
Float16 weights give NaN in the intermediate layers and bfloat16 weights change the features, so only autocast
is reported. Each setting: warm-up, then the mean of the timed runs with torch.cuda.synchronize.
Usage: uv run bench_backbone.py   -> output/bench_backbone.csv
"""

import sys
import time

sys.argv = [sys.argv[0], "--tag", "bench_backbone", "--datasets", "none"]
import server_eval as SE  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_, info = SE.G.load_dataset("mvtec_AD/bottle")
net = SE.Net("dinov3l")
imgs = [net.prep(Image.open(e["path"]).convert("RGB")) for e in info[:16]]
x = net.processor(images=imgs, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to("cuda")


def timed(fn, reps=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


rows = [dict(setting="pipeline", batch=1, ms_per_image=timed(lambda: net.extract([imgs[0]], batch_size=1), 20, 3))]
with torch.inference_mode(), torch.autocast("cuda"):
    for b in (1, 16):
        rows.append(dict(setting="model", batch=b,
                         ms_per_image=timed(lambda: net.model(pixel_values=x[:b]), 30, 5) / b))
R = pd.DataFrame(rows).assign(gpu=torch.cuda.get_device_name(0))
R.to_csv("output/bench_backbone.csv", index=False, float_format="%.2f")
print(R.to_string(index=False))
