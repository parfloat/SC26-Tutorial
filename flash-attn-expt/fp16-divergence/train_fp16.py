#!/usr/bin/env python3
"""TinyGPT trained in pure FP16 (parameters, activations, gradients), no loss
scaling, at a learning rate high enough to diverge. Stops at the first
non-finite loss.

Same model, data and initialisation as ../train.py, except model.half() and
plain SGD (momentum 0.9) instead of AdamW: SGD's update is lr * gradient, so a
large lr makes the weights, activations and gradients grow until FP16's
maximum (65504) is exceeded.

    python train_fp16.py --lr 2.0                 # loss per step, stop at first inf/NaN
    python train_fp16.py --lr 2.0 --oracle        # + per-step max|value| of every layer
                                                  #   output, gradient and parameter, and
                                                  #   the first tensor that went non-finite
Under nixnan, run without --oracle (the oracle launches its own kernels).
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygpt import TinyGPT, EOS, encode  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=2.0)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--oracle", action="store_true")
a = p.parse_args()

torch.manual_seed(a.seed)
here = os.path.dirname(os.path.abspath(__file__))
words = open(os.path.join(here, "..", "training-set.txt")).read().split()
seqs = [encode(w) + [EOS] for w in words]
T = max(map(len, seqs))
x = torch.full((len(seqs), T), EOS, dtype=torch.long)
y = torch.full((len(seqs), T), -100, dtype=torch.long)
for i, s in enumerate(seqs):
    x[i, :len(s)] = torch.tensor(s)
    y[i, :len(s) - 1] = torch.tensor(s[1:])
x, y = x[:, :-1].cuda(), y[:, :-1].cuda()

model = TinyGPT().cuda().half()                    # FP16 everything, no master weights
opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9)

FP16_MAX = 65504.0
acts = {}
if a.oracle:
    def hook(name):
        def f(mod, inp, out):
            acts[name] = out.detach()
        return f
    for name, mod in model.named_modules():
        if name and not any(True for _ in mod.children()):      # leaf modules, in forward order
            mod.register_forward_hook(hook(name))


def amax(t):
    t = t.float()
    fin = t[torch.isfinite(t)]
    return (fin.abs().max().item() if fin.numel() else float("nan"),
            int(torch.isinf(t).sum()), int(torch.isnan(t).sum()))


def report(step):
    """One line per tensor group, plus the first non-finite tensor in execution order."""
    rows = [("act " + n, t) for n, t in acts.items()]
    rows += [("grad " + n, q.grad) for n, q in model.named_parameters() if q.grad is not None]
    rows += [("param " + n, q) for n, q in model.named_parameters()]
    big = max(rows, key=lambda r: amax(r[1])[0] if amax(r[1])[0] == amax(r[1])[0] else -1)
    first_bad = next(((n, amax(t)) for n, t in rows if amax(t)[1] or amax(t)[2]), None)
    act_max = max(amax(t)[0] for _, t in rows if _.startswith("act"))
    grad_max = max((amax(t)[0] for _, t in rows if _.startswith("grad")), default=float("nan"))
    par_max = max(amax(t)[0] for _, t in rows if _.startswith("param"))
    print(f"   oracle: max|act| {act_max:10.4g}  max|grad| {grad_max:10.4g}  max|param| {par_max:10.4g}"
          f"   largest: {big[0]} ({amax(big[1])[0]:.4g})", flush=True)
    if first_bad:
        n, (m, ninf, nnan) = first_bad
        print(f"   oracle: FIRST NON-FINITE tensor (forward, then grads, then params): {n}: "
              f"{ninf} inf, {nnan} NaN", flush=True)


print(f"TinyGPT FP16, SGD lr {a.lr} momentum 0.9, no loss scaling", flush=True)
for step in range(1, a.steps + 1):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    lv = loss.item()
    print(f"step {step:4d}  loss {lv:.4f}", flush=True)
    if a.oracle:
        report(step)
    if lv != lv or abs(lv) == float("inf"):
        print(f"STOP: non-finite loss at step {step}", flush=True)
        break
    opt.step()
