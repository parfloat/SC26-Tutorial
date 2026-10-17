#!/usr/bin/env python3
"""logs/NAME.log + logs/NAME.stdout -> a Markdown summary per run: the command,
time, the kernels instrumented, and nixnan's exception reports grouped by
kernel, instruction and kind.

    python3 summarize.py select smoke gen312 train3 > logs/SUMMARY.md
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = re.compile(r"^#nixnan: running kernel \[(.*)\] \.\.\.$", re.M)
ERR = re.compile(r"#nixnan: error \[([^\]]*)\] detected in operand (\d+) of instruction (.*?) ; "
                 r"in function (.*?) of type (\w+)")
REP = re.compile(r"#nixnan: (NaN|Infinity|-Infinity|Subnormal|Division by 0):\s+(\d+) \((\d+) repeats\)")
SEC = re.compile(r"#nixnan: --- (\w+) (Memory  )?Operations ---")
WL = re.compile(r"FUNCTION_WHITELIST = (\S+)")
SAMP = re.compile(r"^\s*SAMPLING = (\d+)", re.M)


def label(kernel):
    """Short name; flash_fwd_kernel also gets causal / non-causal from its
    template arguments <traits, Is_dropout, Is_causal, ...>."""
    short = re.sub(r"^void (pytorch_flash::)?", "", kernel).split("<")[0]
    if short == "flash_fwd_kernel":
        args = re.sub(r".*>\s*>,\s*", "", kernel.split("(")[0]).rstrip(">").split(", ")
        if len(args) >= 2:
            short += " (causal)" if args[1] == "true" else " (non-causal)"
    return short


for name in sys.argv[1:]:
    log = open(os.path.join(HERE, "logs", name + ".log"), errors="replace").read()
    out = open(os.path.join(HERE, "logs", name + ".stdout"), errors="replace").read()
    wl_path = WL.search(out)                     # nixnan prints its settings to stdout
    wl = set()
    if wl_path and os.path.exists(wl_path.group(1)):
        wl = {l.strip() for l in open(wl_path.group(1)) if l.strip()}
    launched = sorted(set(RUN.findall(log)))
    instr = [k for k in launched if k in wl]
    print(f"## {name}\n")
    cmd = re.search(r"^nixnan: exit (\d+) after (\d+) s", out, re.M)
    samp = SAMP.search(out)
    print(f"* whitelist: `{os.path.basename(wl_path.group(1)) if wl_path else 'none (all kernels)'}`, "
          f"SAMPLING={samp.group(1) if samp else '?'}; exit {cmd.group(1) if cmd else '?'}, "
          f"{cmd.group(2) if cmd else '?'} s")
    for line in out.splitlines():
        if line.startswith(("completion:", "flash output:", "step ", "train next-token")):
            print(f"* program: `{line.strip()}`")
    print(f"* kernels launched: {len(launched)}; **instrumented ({len(instr)})**: "
          + ", ".join(f"`{label(k)}`" for k in instr))
    notin = sorted({label(k) for k in launched if "pytorch_flash" in k and k not in wl})
    if notin:
        print(f"* flash kernels launched but not instrumented: " + ", ".join(f"`{k}`" for k in notin))
    errs = Counter()
    for kind, opnd, ins, fn, ty in ERR.findall(log):
        op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0]   # drop a predicate guard
        errs[(label(fn), op, kind, ty)] += 1
    print(f"* exception reports: {sum(errs.values())}\n")
    if errs:
        print("| kernel | instruction | value | type | reports |\n|---|---|---|---|---:|")
        for (k, op, kind, ty), n in sorted(errs.items(), key=lambda x: (x[0][0], -x[1])):
            print(f"| `{k}` | `{op}` | {kind} | {ty} | {n} |")
        print()
    sec, rows = None, []
    for line in log.splitlines():
        m = SEC.search(line)
        if m:
            sec = m.group(1) + (" memory" if m.group(2) else "")
            continue
        m = REP.search(line)
        if m and sec and (int(m.group(2)) or int(m.group(3))):
            rows.append(f"{sec} {m.group(1)}: {m.group(2)} sites ({m.group(3)} repeats)")
    print("nixnan final report (nonzero): " + ("; ".join(rows) if rows else "all zero") + "\n")
