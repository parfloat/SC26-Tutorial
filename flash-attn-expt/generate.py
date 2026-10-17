#!/usr/bin/env python3
"""Step 5: autoregressive completion with the trained TinyGPT.

    python3 generate.py [--prompt 312] [--length 6] [--ckpt tinygpt.pt]
                        [--linear fp32|fp16|nf4]

--linear selects how the Linear layers run (step 10):
  fp32 (default) -- the trained weights, SIMT FFMA, no tensor cores
  fp16           -- weights cast to FP16 (cuBLAS HMMA GEMMs)
  nf4            -- bitsandbytes NF4 Linear4bit, compute dtype FP16
                    (the 4-bit setting BioMistral was loaded with)
"""
import argparse

import torch
import torch.nn as nn

from tinygpt import load, complete, encode, decode

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="312")
p.add_argument("--length", type=int, default=6)
p.add_argument("--ckpt", default="tinygpt.pt")
p.add_argument("--linear", choices=("fp32", "fp16", "nf4"), default="fp32")
a = p.parse_args()

model = load(a.ckpt)


class Half(nn.Module):
    """Run a Linear in FP16 (input cast down, output cast back to FP32)."""
    def __init__(self, lin):
        super().__init__()
        self.lin = lin.half()

    def forward(self, x):
        return self.lin(x.half()).float()


def swap_linears(module, make):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, make(child))
        else:
            swap_linears(child, make)


if a.linear == "fp16":
    swap_linears(model, Half)
elif a.linear == "nf4":
    import bitsandbytes as bnb

    class NF4(nn.Module):
        """bitsandbytes 4-bit (NF4) Linear, compute dtype FP16."""
        def __init__(self, lin):
            super().__init__()
            self.q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=False,
                                       compute_dtype=torch.float16, quant_type="nf4")
            self.q.weight = bnb.nn.Params4bit(lin.weight.data.cpu(), requires_grad=False,
                                              quant_type="nf4")
            self.q.cuda()                       # quantizes the weights on the move

        def forward(self, x):
            return self.q(x.half()).float()

    swap_linears(model, NF4)

out = complete(model, a.prompt, a.length)
print(f"completion ({a.linear} linears): {a.prompt} -> {out}   (tokens {decode(encode(a.prompt))} -> {decode(encode(out))})")
