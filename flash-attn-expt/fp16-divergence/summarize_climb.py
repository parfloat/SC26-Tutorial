#!/usr/bin/env python3
"""Condense the nixnan logs of the FP16 divergence into one timeline.

    python summarize_climb.py nixnan-climb.log [nixnan-exceptions.log] > SUMMARY.txt

For each training step (the stretch of log after the previous "step N" line, i.e. the
forward/backward pass that produced step N's loss) it lists, per kernel and format, the
highest binade any histogram bin reached and the instructions that put values there,
then the exception reports (inf / NaN, subnormals omitted) in the order nixnan printed them.
"""
import re
import sys
from collections import defaultdict

STEP = re.compile(r"\] step +(\d+) +loss (\S+)")
BIN = re.compile(r"#nixnan: (f16|f32) bin has reached threshold: function=(.*?) range=\[(-?\d+),(-?\d+)\] "
                 r"count=(\d+)(?: instruction=(.*?) ; in function)?")
ERR = re.compile(r"#nixnan: error \[([^\]]*)\] detected in operand (\d+) of instruction (.*?) ; in function (.*?) of type (\w+)")


def kernel(fn):
    if "layer_norm" in fn or fn.startswith("void at::native::(a"):
        return "LayerNorm"
    if "CUDAFunctor_add" in fn or "vectorized_elementwise_kernel" in fn:
        return "add (FP16 residual)"
    return fn[:40]


for path in sys.argv[1:]:
    print(f"==================== {path}")
    seg = 1                                    # log lines before "step 1" belong to step 1
    climb = defaultdict(lambda: defaultdict(lambda: [-99, set(), 0]))
    errors = []
    loss = {}
    for line in open(path, errors="replace"):
        m = STEP.search(line)
        if m:
            loss[int(m.group(1))] = m.group(2)
            seg = int(m.group(1)) + 1
            continue
        m = BIN.search(line)
        if m:
            fmt, fn, lo, hi, cnt, ins = m.groups()
            c = climb[seg][(kernel(fn), fmt)]
            if int(hi) > c[0]:
                c[0], c[1] = int(hi), set()
            if int(hi) == c[0] and ins:
                c[1].add(ins.split(" ")[0])
            c[2] += 1
            continue
        m = ERR.search(line)
        if m and m.group(1) != "subnormal" and "infinity" in m.group(1) or (m and "NaN" in m.group(1)):
            errors.append((seg, m.group(1), m.group(3), kernel(m.group(4)) if m.group(4).strip() != "void at::native::"
                           else "LayerNorm (anonymous-namespace name)", m.group(5)))
        if "WATCHER" in line:
            errors.append((seg, "WATCHER", "first NaN report -> run killed", "", ""))
    steps = sorted(set(climb) | {e[0] for e in errors})
    for s in steps:
        tag = f"loss {loss[s]}" if s in loss else "(killed before the loss was computed)"
        print(f"\n--- step {s}: {tag}")
        for (k, fmt), (hi, ins, n) in sorted(climb[s].items()):
            print(f"    {k:22s} {fmt}: highest binade 2^{hi} (|x| >= {2.0 ** hi:,.0f})  via {', '.join(sorted(ins))}"
                  f"   [{n} bin reports]")
        for e in [e for e in errors if e[0] == s]:
            print(f"    !! {e[1]:10s} {e[2]:45s} {e[3]} ({e[4]})")
