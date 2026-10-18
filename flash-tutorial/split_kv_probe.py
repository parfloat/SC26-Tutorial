#!/usr/bin/env python3
"""Which flash kernels PyTorch's SDPA launches for a decoding-shaped call
(1 query per head) as the number of keys grows, with grid and time.
The thread-block grid of flash_fwd_splitkv_kernel is (query tiles, splits, batch*heads).
Numbers for the deck's split-KV slides:  python3 split_kv_probe.py"""
import json, os, tempfile

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, ProfilerActivity


def run(B, H, Lq, Lk, d=32):
    q = torch.randn(B, H, Lq, d, device="cuda", dtype=torch.half)
    k = torch.randn(B, H, Lk, d, device="cuda", dtype=torch.half)
    v = torch.randn_like(k)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v)            # warm-up
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
    path = os.path.join(tempfile.mkdtemp(), "trace.json")
    p.export_chrome_trace(path)
    ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel" and "flash" in e["name"]]
    return [(e["name"].split("<")[0].replace("void pytorch_flash::", ""), e["args"].get("grid"), e["dur"]) for e in ev]


print(f"{torch.cuda.get_device_name()}: {torch.cuda.get_device_properties(0).multi_processor_count} SMs, torch {torch.__version__}")
for B, Lk in [(1, 7), (1, 256), (1, 320), (1, 512), (1, 1024), (1, 2048), (1, 8192), (100, 8192)]:
    print(f"batch {B:3d}, 4 heads, 1 query, {Lk:5d} keys:", run(B, 4, 1, Lk))
