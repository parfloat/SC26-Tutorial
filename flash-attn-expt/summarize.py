#!/usr/bin/env python3
"""Summarize nixnan logs: the final report totals, and the distinct
exception sites grouped by kernel (flash-attention kernels named apart).

    python3 summarize.py logs/*.log
"""
import re
import sys
from collections import Counter, defaultdict

ERR = re.compile(r"#nixnan: error \[([^\]]*)\] detected in (.*?) of instruction (.*?) ; in function (.*?) of type (\w+)")
REP = re.compile(r"#nixnan: (NaN|Infinity|-Infinity|Subnormal|Division by 0):\s+(\d+) \((\d+) repeats\)")
SEC = re.compile(r"#nixnan: --- (\w+) (Memory  )?Operations ---")
RUN = re.compile(r"#nixnan: running kernel \[(.*?)\]")


def short(fn):
    for k in ("flash_fwd_splitkv_combine_kernel", "flash_fwd_splitkv_kernel",
              "flash_bwd", "flash_fwd_kernel"):
        if k in fn:
            return k
    fn = re.sub(r"^void ", "", fn)
    return re.sub(r"<.*", "", fn).split("(")[0]


def kernel_name(run):
    return short(re.sub(r"^void ", "", run))


for path in sys.argv[1:]:
    text = open(path, errors="replace").read()
    print(f"==================== {path}")
    if "nixnan Report" not in text:
        print("  (no final report: run incomplete)")
    launches = Counter(kernel_name(m) for m in RUN.findall(text))
    flash = {k: v for k, v in launches.items() if "flash" in k}
    print(f"  kernel launches logged: {sum(launches.values())}  (flash: {flash or 'none'})")
    # final report
    sec, rows = None, []
    for line in text.splitlines():
        m = SEC.search(line)
        if m:
            sec = m.group(1) + (" mem" if m.group(2) else "")
            continue
        m = REP.search(line)
        if m and sec and (int(m.group(2)) or int(m.group(3))):
            rows.append(f"{sec} {m.group(1)}: {m.group(2)} sites ({m.group(3)} repeats)")
    print("  report (nonzero):", "; ".join(rows) if rows else "all zero")
    # distinct exception sites by kernel
    sites = defaultdict(set)
    kinds = defaultdict(Counter)
    for m in ERR.finditer(text):
        kind, opnd, ins, fn, ty = m.groups()
        k = short(fn)
        sites[k].add((ins, opnd))
        for x in kind.split(","):
            kinds[k][x.strip()] += 1
    for k in sorted(sites, key=lambda k: -len(sites[k])):
        print(f"  {k:40s} {len(sites[k]):5d} sites  {dict(kinds[k])}")
