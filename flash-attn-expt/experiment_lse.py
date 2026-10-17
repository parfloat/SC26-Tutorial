#!/usr/bin/env python3
"""Step 9: controlled A/B experiment on flash-attention's split-KV combine guard

    lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    (PyTorch 2.6 flash_fwd_kernel.h:1157 == flash-attention flash_fwd_kernel.h:1204)

The guard lives in flash_fwd_splitkv_combine_kernel. PyTorch launches that
kernel only when its heuristic picks num_splits > 1, which for head_dim 32
needs seqlen_k > 256 (one key block = 256). So the prompt is a fixed
297-digit context followed by "312": A + 300 digits = 301 keys -> 2 splits.

The external input is a vector added to the token embedding of the last
prompt position (as a soft prompt or a vision encoder's output would be):
  A: all zeros                      -> clean
  B: +inf in coordinate 0           -> an upstream FP16 overflow
Same model, same tokens, same kernels; only the external input differs.

    python3 experiment_lse.py --input A|B           # plain run (use under nixnan)
    python3 experiment_lse.py --input A|B --oracle  # no nixnan: flash LSE per
          call (+inf = combine guard fired), NaN census of every block output,
          and the list of flash kernels launched (torch profiler)
"""
import argparse
import random
from collections import Counter

import torch

import tinygpt
from tinygpt import load, complete

p = argparse.ArgumentParser()
p.add_argument("--input", choices=("A", "B"), required=True)
p.add_argument("--context", type=int, default=297)
p.add_argument("--oracle", action="store_true")
p.add_argument("--ckpt", default="tinygpt.pt")
a = p.parse_args()

rng = random.Random(7)
prompt = "".join(rng.choice("0123456789") for _ in range(a.context)) + "312"
T = 1 + len(prompt)                                    # A + digits
model = load(a.ckpt)
ext = torch.zeros(T + 8, model.cfg["d"], device="cuda")
if a.input == "B":
    ext[T - 1, 0] = float("inf")                       # last prompt position
model.ext_embed = ext

if not a.oracle:
    out = complete(model, prompt, len(prompt) + 3)
    print(f"input {a.input}: ...{prompt[-6:]} -> ...{out[-6:]}  (seqlen_k {T}..{T + 2})")
    raise SystemExit

# ---------------------------------------------------------------- oracle
tinygpt.ORACLE = []
nan_rows = []                                          # (layer, call, #non-finite rows)
calls = Counter()


def census(mod, inp, out):
    calls[mod.idx] += 1
    bad = (~torch.isfinite(out)).any(-1)[0]
    nan_rows.append((mod.idx, calls[mod.idx], int(bad.sum()), out.shape[1]))


for b in model.blocks:
    b.register_forward_hook(census)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    out = complete(model, prompt, len(prompt) + 3)
print(f"input {a.input}: ...{prompt[-6:]} -> ...{out[-6:]}")

kern = Counter()
for e in prof.events():
    if "flash" in e.name:
        n = e.name
        kern["flash_fwd_splitkv_combine_kernel" if "combine" in n else
             "flash_fwd_splitkv_kernel" if "splitkv" in n else "flash_fwd_kernel"] += 1
print("flash kernels launched (model + oracle calls):", dict(kern))

# Each model attention call has a twin oracle call, in order: 4 layers x (prefill + 2 decode steps).
step_names = ["prefill"] + [f"decode{i}" for i in range(1, 3)]
print(f"{'call':9s} {'layer':5s} {'q x k':>9s}  {'+inf LSE rows':>13s} {'-inf':>5s} {'NaN':>4s}   "
      f"block output rows non-finite")
for i, (layer, q, k, lse) in enumerate(tinygpt.ORACLE):
    _, _, nbad, nrows = nan_rows[i]
    print(f"{step_names[i // 4]:9s} {layer:5d} {q:4d}x{k:<4d}  {int(torch.isposinf(lse).sum()):13d} "
          f"{int(torch.isneginf(lse).sum()):5d} {int(torch.isnan(lse).sum()):4d}   {nbad}/{nrows}")
fired = sum(int(torch.isposinf(l).sum()) for *_, l in tinygpt.ORACLE)
nan_lse = sum(int(torch.isnan(l).sum()) for *_, l in tinygpt.ORACLE)
print(f"combine guard fired on {fired} (head,row) LSE entries; NaN LSE entries returned: {nan_lse}")
