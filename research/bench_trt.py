"""Feature-extractor timings of DINOv3 ViT-L/16 at 448 x 448: PyTorch and TensorRT (isolated test bench, 12 Oct).

Two extractors: GaLAD (hidden states of blocks 13 and 19; the network stops after block 19) and AnomalyDINO
(last block and final norm; the full network). Settings:
  pytorch_eval   PyTorch, float32 weights + autocast fp16, full model with all hidden states (as in the evaluation)
  pytorch_opt    PyTorch, bfloat16 weights, SDPA, the extractor above, CUDA graph
  tensorrt_bf16  ONNX export (fp32, static shapes) -> TensorRT engine built with the BF16 flag
  tensorrt_fp16  same with the FP16 flag (kept only if the outputs are finite)
Batch 1 and 16, inputs already on the GPU. Protocol: 50 warm-up runs, then 3 repetitions of the timed loop with
torch.cuda.synchronize; the median repetition is reported, per image.
Usage (isolated env): .venv/Scripts/python.exe bench_trt.py   -> results/trt_timings.csv
"""

import time
from pathlib import Path

import pandas as pd
import tensorrt as trt
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModel

NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
ONNX, ENG, RES = Path("onnx"), Path("engines"), Path("results")
for d in (ONNX, ENG, RES):
    d.mkdir(exist_ok=True)
dev = torch.device("cuda")
TRT_LOG = trt.Logger(trt.Logger.WARNING)


class Extractor(torch.nn.Module):
    def __init__(self, m, depth, keep, final_norm):
        super().__init__()
        self.m, self.depth, self.keep, self.final_norm = m, depth, keep, final_norm

    def forward(self, x):
        h = self.m.embeddings(x)
        rope = self.m.rope_embeddings(x)
        outs = []
        for i, blk in enumerate(self.m.layer[:self.depth], 1):
            h = blk(h, position_embeddings=rope)
            if i in self.keep:
                outs.append(self.m.norm(h) if self.final_norm else h)
        return tuple(outs)


VARIANTS = {"galad": dict(depth=19, keep=(13, 19), final_norm=False),
            "anomalydino": dict(depth=24, keep=(24,), final_norm=True)}


def timed(fn, sync, reps, warm=50, rounds=3):
    for _ in range(warm):
        fn()
    sync()
    res = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        res.append((time.perf_counter() - t0) / reps * 1000)
    return sorted(res)[1]


def build_engine(onnx_path, flag):
    plan = ENG / f"{onnx_path.stem}_{flag}.plan"
    if not plan.exists():
        builder = trt.Builder(TRT_LOG)
        net = builder.create_network(0)
        parser = trt.OnnxParser(net, TRT_LOG)
        assert parser.parse_from_file(str(onnx_path)), [parser.get_error(i) for i in range(parser.num_errors)]
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
        cfg.set_flag(getattr(trt.BuilderFlag, flag.upper()))
        t0 = time.time()
        data = builder.build_serialized_network(net, cfg)
        assert data is not None, f"build failed: {plan}"
        plan.write_bytes(bytes(data))
        print(f"  built {plan.name} in {time.time() - t0:.0f} s", flush=True)
    return trt.Runtime(TRT_LOG).deserialize_cuda_engine(plan.read_bytes())


def trt_runner(engine):
    ctx = engine.create_execution_context()
    bufs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16}[engine.get_tensor_dtype(name)]
        t = (torch.randn if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else torch.empty)(
            tuple(engine.get_tensor_shape(name)), dtype=dt, device=dev)
        bufs[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.Stream()
    return (lambda: ctx.execute_async_v3(stream.cuda_stream)), stream.synchronize, bufs


rows = []
for var, cfg in VARIANTS.items():
    for b in (1, 16):
        reps = 200 if b == 1 else 30
        x = torch.randn(b, 3, 448, 448, device=dev)
        # PyTorch as evaluated (full model, autocast) and optimized (bf16 extractor + CUDA graph)
        m32 = AutoModel.from_pretrained(NAME, attn_implementation="sdpa", local_files_only=True).to(dev).eval()
        with torch.inference_mode():
            def ev():
                with torch.autocast("cuda"):
                    return m32(pixel_values=x, output_hidden_states=True)
            rows.append(dict(extractor=var, setting="pytorch_eval", batch=b, ms_per_image=timed(ev, torch.cuda.synchronize, reps) / b, finite=True))
        # ONNX export from the fp32 extractor (attention exported as plain matmul/softmax)
        onnx_path = ONNX / f"{var}_b{b}.onnx"
        if not onnx_path.exists():
            ext32 = Extractor(m32, **cfg).eval()
            names = [f"h{k}" for k in cfg["keep"]]
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                torch.onnx.export(ext32, (x,), str(onnx_path), dynamo=False, opset_version=18,
                                  input_names=["pixels"], output_names=names)
            print(f"  exported {onnx_path}", flush=True)
        del m32
        torch.cuda.empty_cache()
        mbf = AutoModel.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                        local_files_only=True).to(dev).eval()
        ext = Extractor(mbf, **cfg).eval()
        xb = x.bfloat16()
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    ext(xb)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = ext(xb)
            ok = all(torch.isfinite(o).all().item() for o in out)
            rows.append(dict(extractor=var, setting="pytorch_opt", batch=b, ms_per_image=timed(g.replay, torch.cuda.synchronize, reps) / b, finite=ok))
            del g, out
        del mbf, ext, xb
        torch.cuda.empty_cache()
        for flag in ("bf16", "fp16"):
            try:
                eng = build_engine(onnx_path, flag)
                run, sync, bufs = trt_runner(eng)
                run(); sync()
                ok = all(torch.isfinite(t).all().item() for n, t in bufs.items() if n != "pixels")
                rows.append(dict(extractor=var, setting=f"tensorrt_{flag}", batch=b, ms_per_image=timed(run, sync, reps) / b, finite=ok))
                del eng, bufs
            except Exception as e:  # noqa: BLE001
                print(f"  TensorRT {flag} failed for {onnx_path.name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()
        print(pd.DataFrame(rows).tail(4).to_string(index=False), flush=True)
        del x
R = pd.DataFrame(rows).assign(gpu=torch.cuda.get_device_name(0), tensorrt=trt.__version__, torch=torch.__version__)
R.to_csv(RES / "trt_timings.csv", index=False, float_format="%.2f")
print(R.to_string(index=False))
