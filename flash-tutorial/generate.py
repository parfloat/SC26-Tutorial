#!/usr/bin/env python3
"""Autoregressive completion with the trained TinyGPT.

    python3 generate.py [--prompt 312] [--length 6] [--ckpt tinygpt.pt]
"""
import argparse

from tinygpt import complete, decode, encode, load

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="312")
p.add_argument("--length", type=int, default=6)
p.add_argument("--ckpt", default="tinygpt.pt")
a = p.parse_args()

out = complete(load(a.ckpt), a.prompt, a.length)
print(f"completion: {a.prompt} -> {out}   (tokens {decode(encode(a.prompt))} -> {decode(encode(out))})")
