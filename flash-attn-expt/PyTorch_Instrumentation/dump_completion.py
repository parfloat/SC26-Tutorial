#!/usr/bin/env python3
"""Dump the padding lanes of flash_fwd_kernel during TinyGPT's completion of 312.

The completion (312 -> 6 digits) makes 3 model forward passes x 4 layers = 12
flash calls, all through the non-split flash_fwd_kernel (128 x 128 tile,
head dim 32). seqlen_q is 4 (prefill) or 1 (decode) and seqlen_k 4..6, so
almost the whole tile lies outside the sequence. For the last 4 calls (the 4
layers of the last forward pass) this script writes, per head, the whole tile:
the Q, K, V shared-memory tiles, S = QK^T before and after masking, P, O and
the per-row softmax guard (softmax.h: sum == 0 || sum != sum).

    python dump_completion.py            # -> traces/completion/
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch

import instr_flash as IF
from tinygpt import complete, load

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=os.path.join(IF.HERE, "..", "tinygpt.pt"))
p.add_argument("--prompt", default="312")
p.add_argument("--length", type=int, default=6)
p.add_argument("--keep", type=int, default=4, help="dump the last KEEP flash calls")
p.add_argument("--out", default=os.path.join(IF.HERE, "traces", "completion"))
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

IF.install()
model = load(a.ckpt)
runs = []
for r in range(2):                       # run twice: stale lanes change from run to run, valid lanes don't
    IF.CALLS.clear()
    text = complete(model, a.prompt, a.length)
    runs.append((text, list(IF.CALLS)))
text, calls = runs[0]


# ---------------------------------------------------------------- classification
def kinds(x, fp16):
    """Counter of value kinds in a float tensor (FP16 or FP32 subnormal threshold)."""
    x = x.float().flatten()
    tiny = 2.0 ** -14 if fp16 else 2.0 ** -126
    c = Counter()
    c["NaN"] = int(torch.isnan(x).sum())
    c["+inf"] = int(torch.isposinf(x).sum())
    c["-inf"] = int(torch.isneginf(x).sum())
    fin = x[torch.isfinite(x)]
    c["zero"] = int((fin == 0).sum())
    c["subnormal"] = int(((fin != 0) & (fin.abs() < tiny)).sum())
    c["normal"] = int((fin.abs() >= tiny).sum())
    return c


def fmt(c):
    n = sum(c.values())
    return f"{n:6d} values: " + ", ".join(f"{k} {c[k]}" for k in ("NaN", "+inf", "-inf", "subnormal", "zero", "normal") if c[k])


def regions(call, h):
    sq, sk = call["q_len"], call["k_len"]
    t = {n: IF.tile(call, h, n) for n in ("Q", "K", "V", "S_RAW", "S_MASK", "P", "O", "ROW")}
    return [
        ("Q  rows < seqlen_q  (loaded)", t["Q"][:sq], True),
        ("Q  rows >= seqlen_q (never loaded)", t["Q"][sq:], True),
        ("K  rows < seqlen_k  (loaded)", t["K"][:sk], True),
        ("K  rows >= seqlen_k (never loaded)", t["K"][sk:], True),
        ("V  rows >= seqlen_k (cleared)", t["V"][sk:], True),
        ("S = QK^T  valid (r < sq, c < sk)", t["S_RAW"][:sq, :sk], False),
        ("S = QK^T  pad cols (r < sq, c >= sk)", t["S_RAW"][:sq, sk:], False),
        ("S = QK^T  pad rows (r >= sq)", t["S_RAW"][sq:], False),
        ("S masked  pad rows (r >= sq)", t["S_MASK"][sq:], False),
        ("P         valid rows", t["P"][:sq], True),
        ("P         pad rows", t["P"][sq:], True),
        ("O         valid rows", t["O"][:sq], False),
        ("O         pad rows", t["O"][sq:], False),
    ]


# ---------------------------------------------------------------- provenance of the stale rows
def row_keys(bits):
    return [tuple(r) for r in bits.tolist()]


def provenance(calls, n):
    """For call n, match every never-loaded Q/K row (bitwise, 32 FP16 values) against the rows
    every earlier flash call of the same run left in shared memory: its Q, K, V tiles and its O
    tile (acc_o rounded to FP16, which the epilogue writes into the sQ region)."""
    seen = {}
    for m in range(n):
        c = calls[m]
        if c["tile"] is None:
            continue
        for h in range(c["tile"].shape[0]):
            for name in ("Q", "K", "V", "O"):
                if name == "O":
                    bits = IF.tile(c, h, "O").half().view(torch.int16).to(torch.int32) & 0xFFFF
                else:
                    bits = IF.tile(c, h, name, bits=True)
                for r, key in enumerate(row_keys(bits)):
                    seen.setdefault(key, (m, h, name, r))
    c = calls[n]
    res = Counter()
    for h in range(c["tile"].shape[0]):
        for name, lo in (("Q", c["q_len"]), ("K", c["k_len"])):
            for key in row_keys(IF.tile(c, h, name, bits=True)[lo:]):
                hit = seen.get(key)
                res[(name, "zero row" if not any(key) else
                     f"= {hit[2]} tile of an earlier call" if hit else "no match")] += 1
    return res


def fp32_pairs(bits):
    """Fraction of adjacent FP16 pairs that, read as one FP32 word, are a plausible FP32 value
    (0 or 1e-6 <= |x| <= 1e6). Random bits give about 0.16."""
    b = bits.flatten().to(torch.int64)
    w = (b[0::2] | (b[1::2] << 16)).to(torch.int32).view(torch.float32)
    ok = (w == 0) | ((w.abs() >= 1e-6) & (w.abs() <= 1e6))
    return float(ok.float().mean()) if w.numel() else float("nan")


# ---------------------------------------------------------------- write
first = len(calls) - a.keep
rep = open(os.path.join(a.out, "REPORT.txt"), "w")


def out(s=""):
    print(s)
    rep.write(s + "\n")


out(f"TinyGPT {a.prompt!r} -> {text!r}  (second run: {runs[1][0]!r}); {len(calls)} flash calls, "
    f"dumping calls {first}..{len(calls) - 1}")
out("Instrumented copy of PyTorch 2.6 flash_fwd_kernel (bitwise-equal outputs, see check_equivalence.py).")
out("Tile 128 x 128 (kBlockM x kBlockN), head dim 32; batch 0, m_block 0; all 4 heads summed.\n")
arrays = {}
for n in range(first, len(calls)):
    c = calls[n]
    assert c["num_splits"] == 1 and c["tile"] is not None
    sq, sk = c["q_len"], c["k_len"]
    out(f"=== call {n}: layer {c['layer']}, seqlen_q {sq}, seqlen_k {sk}, causal {c['causal']}")
    agg = {}
    for h in range(c["tile"].shape[0]):
        for name, x, fp16 in regions(c, h):
            agg.setdefault(name, [Counter(), fp16])[0].update(kinds(x, fp16))
    for name, (cnt, _) in agg.items():
        out(f"  {name:38s} {fmt(cnt)}")
    rows = torch.stack([IF.tile(c, h, "ROW") for h in range(c["tile"].shape[0])])
    g = rows[:, :, 3]
    out(f"  softmax.h guard (sum==0 || sum!=sum), rows < seqlen_q:  fired {int((g[:, :sq] > 0).sum())} "
        f"of {g[:, :sq].numel()}")
    out(f"  softmax.h guard, rows >= seqlen_q:                     fired {int((g[:, sq:] > 0).sum())} "
        f"of {g[:, sq:].numel()}  (sum NaN: {int((g[:, sq:] == 2).sum())}, sum 0: {int((g[:, sq:] == 1).sum())})"
        f"  -> lse +inf, never stored")
    out(f"  stale-row provenance: {dict(provenance(calls, n))}")
    qb = torch.cat([IF.tile(c, h, "Q", bits=True)[sq:] for h in range(4)])
    out(f"  stale Q lanes read as FP32 pairs, plausible-FP32 fraction: {fp32_pairs(qb):.2f}")
    other = runs[1][1][n]
    same_valid = all(torch.equal(IF.tile(c, h, "O")[:sq], IF.tile(other, h, "O")[:sq]) for h in range(4))
    same_stale = all(torch.equal(IF.tile(c, h, "Q", bits=True)[sq:], IF.tile(other, h, "Q", bits=True)[sq:])
                     for h in range(4))
    out(f"  second run: valid O rows identical: {same_valid}; stale Q rows identical: {same_stale}")
    out()
    for name in ("Q", "K", "V", "S_RAW", "S_MASK", "P", "O", "ROW"):
        arrays[f"call{n}_{name}"] = torch.stack([IF.tile(c, h, name) for h in range(4)]).float().numpy()
        if name in ("Q", "K", "V"):
            arrays[f"call{n}_{name}_bits"] = torch.stack([IF.tile(c, h, name, bits=True)
                                                          for h in range(4)]).numpy().astype(np.uint16)
    arrays[f"call{n}_meta"] = np.array([c["layer"], sq, sk, int(c["causal"])])

    # Readable per-call dump: every head, Q/K/V/O tiles in full, S/P for the rows that matter.
    with open(os.path.join(a.out, f"call{n}_layer{c['layer']}.txt"), "w") as f:
        f.write(f"call {n}: layer {c['layer']}, seqlen_q {sq}, seqlen_k {sk}, causal {c['causal']}\n"
                f"Rows marked * are outside the sequence (padding lanes).\n")
        for h in range(4):
            for name in ("Q", "K", "V", "O"):
                t = IF.tile(c, h, name).float()
                lim = sq if name in ("Q", "O") else sk
                f.write(f"\n--- head {h} {name} tile {tuple(t.shape)}\n")
                for r in range(t.shape[0]):
                    f.write(f"{'*' if r >= lim else ' '}{r:3d} " + " ".join(f"{v:10.3g}" for v in t[r].tolist()) + "\n")
            for name in ("S_RAW", "S_MASK", "P"):
                t = IF.tile(c, h, name).float()
                f.write(f"\n--- head {h} {name}: rows 0..{sq + 7}, cols 0..15 (seqlen_k {sk})\n")
                for r in range(min(sq + 8, t.shape[0])):
                    f.write(f"{'*' if r >= sq else ' '}{r:3d} " + " ".join(f"{v:10.3g}" for v in t[r, :16].tolist()) + "\n")
            t = IF.tile(c, h, "ROW")
            f.write(f"\n--- head {h} per-row softmax state: row_max row_sum lse guard(1: sum 0, 2: sum NaN)\n")
            for r in range(t.shape[0]):
                f.write(f"{'*' if r >= sq else ' '}{r:3d} " + " ".join(f"{v:12.5g}" for v in t[r].tolist()) + "\n")
np.savez_compressed(os.path.join(a.out, "dumps.npz"), **arrays)
rep.close()
print("wrote", a.out)
