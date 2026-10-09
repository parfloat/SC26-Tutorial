#!/usr/bin/env python3
"""Dump what flash attention's two NaN guards clean up, for experiment_lse.py's inputs A and B.

Same model, same 300-digit prompt as ../experiment_lse.py (2 KV splits, so the
split-KV kernel and the combine kernel run). Every attention call records:

  flash_fwd_splitkv_kernel, softmax.h:177, per (split, head, row):
      lse = (sum == 0 || sum != sum) ? -inf : row_max * scale + log(sum)
  flash_fwd_splitkv_combine_kernel, flash_fwd_kernel.h:1157, per row:
      lse_logsum = (lse_sum == 0 || lse_sum != lse_sum) ? +inf : log(lse_sum) + lse_max
      with lse_sum, both disjuncts, the value the guard replaced, and the
      number of non-finite values in the combined output row.

    python dump_lse.py --input A|B       # -> traces/lse-A/, traces/lse-B/
"""
import argparse
import os
import random
from collections import Counter

import numpy as np
import torch

import instr_flash as IF
from tinygpt import complete, load

p = argparse.ArgumentParser()
p.add_argument("--input", choices=("A", "B"), required=True)
p.add_argument("--ckpt", default=os.path.join(IF.HERE, "..", "tinygpt.pt"))
p.add_argument("--keep", type=int, default=4, help="per-row listings for the last KEEP calls")
a = p.parse_args()
outdir = os.path.join(IF.HERE, "traces", f"lse-{a.input}")
os.makedirs(outdir, exist_ok=True)

rng = random.Random(7)                                     # as in ../experiment_lse.py
prompt = "".join(rng.choice("0123456789") for _ in range(297)) + "312"
T = 1 + len(prompt)
IF.install()
model = load(a.ckpt)
ext = torch.zeros(T + 8, model.cfg["d"], device="cuda")
if a.input == "B":
    ext[T - 1, 0] = float("inf")
model.ext_embed = ext
IF.CALLS.clear()
text = complete(model, prompt, len(prompt) + 3)
calls = IF.CALLS

GUARD = {0: "none", 1: "sum==0", 2: "sum!=sum (NaN)", 3: "no key block (early exit)"}
rep = open(os.path.join(outdir, "REPORT.txt"), "w")


def out(s=""):
    print(s)
    rep.write(s + "\n")


out(f"input {a.input}: ...{prompt[-6:]} -> ...{text[-6:]}   ({len(calls)} flash calls, 4 heads)")
out("Instrumented copy of PyTorch 2.6 flash (bitwise-equal outputs, see check_equivalence.py).\n")
tot = Counter()
arrays = {}
for n, c in enumerate(calls):
    sq, sk = c["q_len"], c["k_len"]
    out(f"=== call {n}: layer {c['layer']}, seqlen_q {sq}, seqlen_k {sk}, causal {c['causal']}, "
        f"num_splits {c['num_splits']}")
    if c["split"] is None or c["num_splits"] < 2:
        out("  (not split: no combine kernel)")
        continue
    s = c["split"][c["split"][:, 0] == 1]                  # written records
    for valid in (1, 0):
        sel = s[s[:, 4] == valid]
        cnt = Counter((int(r[1]), GUARD[int(r[8])]) for r in sel)
        out(f"  splitkv softmax.h guard, {'valid rows' if valid else 'pad rows  '}: " +
            ", ".join(f"split {k[0]} {k[1]}: {v}" for k, v in sorted(cnt.items())))
        for (sp, g), v in cnt.items():
            tot[("split", valid, g)] += v
    cb = c["comb"][c["comb"][:, 0] == 1]
    for valid in (1, 0):
        sel = cb[cb[:, 2] == valid]
        fired = sel[sel[:, 9] == 1]
        out(f"  combine guard, {'valid rows' if valid else 'pad rows  '}: {len(sel)} rows, fired {len(fired)} "
            f"(lse_sum == 0: {int(fired[:, 7].sum())}, lse_sum != lse_sum: {int(fired[:, 8].sum())}); "
            f"rows with non-finite O: {int((sel[:, 12] > 0).sum())}")
        tot[("comb", valid, "rows")] += len(sel)
        tot[("comb", valid, "fired")] += len(fired)
        tot[("comb", valid, "nan_disjunct")] += int(fired[:, 8].sum())
        tot[("comb", valid, "zero_disjunct")] += int(fired[:, 7].sum())
        tot[("comb", valid, "nonfinite_O_rows")] += int((sel[:, 12] > 0).sum())
    arrays[f"call{n}_split"] = c["split"].numpy()
    arrays[f"call{n}_comb"] = c["comb"].numpy()
    arrays[f"call{n}_meta"] = np.array([c["layer"], sq, sk, int(c["causal"]), c["num_splits"]])

    if n >= len(calls) - a.keep or (a.input == "B" and n == 0):
        # Per-row listing of every combine row the guard cleaned (valid rows), and a sample of pad rows.
        with open(os.path.join(outdir, f"call{n}_layer{c['layer']}_combine.txt"), "w") as f:
            f.write(f"call {n}: layer {c['layer']}, seqlen_q {sq}, seqlen_k {sk}; combine row = head * {sq} + row\n"
                    "Rows marked * are padding rows of the last 16-row combine block (not stored).\n")
            f.write(f"{'':1s}{'head':>4s} {'row':>4s} {'lse_split0':>11s} {'lse_split1':>11s} {'lse_max':>9s} "
                    f"{'lse_sum':>9s} {'sum==0':>6s} {'sum!=sum':>8s} {'fired':>5s} {'replaced':>10s} "
                    f"{'stored':>8s} {'nonfin_O':>8s}\n")
            for r in cb.tolist():
                idx = int(r[1])
                if not (r[9] or (r[12] and r[2]) or n >= len(calls) - a.keep):
                    continue
                f.write(f"{' ' if r[2] else '*'}{idx // sq:4d} {idx % sq:4d} {r[3]:11.5g} {r[4]:11.5g} {r[5]:9.4g} "
                        f"{r[6]:9.4g} {int(r[7]):6d} {int(r[8]):8d} {int(r[9]):5d} {r[10]:10.5g} {r[11]:8.5g} "
                        f"{int(r[12]):8d}\n")
            f.write("\nsplitkv softmax.h guard records: split head row valid row_max row_sum lse guard\n")
            for r in s.tolist():
                if r[8] or n >= len(calls) - a.keep:
                    f.write(f"{int(r[1]):5d} {int(r[2]):4d} {int(r[3]):4d} {int(r[4]):5d} {r[5]:10.5g} "
                            f"{r[6]:10.5g} {r[7]:10.5g}  {GUARD[int(r[8])]}\n")
    out()

out("=== totals over all calls")
for valid, lab in ((1, "valid rows"), (0, "pad rows")):
    out(f"  splitkv softmax.h guard, {lab}: " +
        ", ".join(f"{g}: {v}" for (k, vv, g), v in sorted(tot.items()) if k == "split" and vv == valid))
    out(f"  combine guard, {lab}: {tot[('comb', valid, 'rows')]} rows, fired {tot[('comb', valid, 'fired')]} "
        f"(lse_sum == 0: {tot[('comb', valid, 'zero_disjunct')]}, lse_sum != lse_sum: "
        f"{tot[('comb', valid, 'nan_disjunct')]}), rows with non-finite O: {tot[('comb', valid, 'nonfinite_O_rows')]}")
np.savez_compressed(os.path.join(outdir, "records.npz"), **arrays)
rep.close()
