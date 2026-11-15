#!/usr/bin/env python3
"""Tables from logs/: E0 budget and E1 RESULT lines with nixnan's non-zero counters."""
import glob, os, re

def nixnan(path):
    sec, out = None, []
    for line in open(path, errors="replace"):
        m = re.match(r"#nixnan: --- (.*?) ---", line)
        if m:
            sec = re.sub(r"\s+", " ", m.group(1)).replace(" Operations", "").replace("Memory ", "mem")
            continue
        m = re.match(r"#nixnan: ([A-Za-z0-9\- ]+?):\s+(\d+) \((\d+) repeats\)", line)
        if m and sec and int(m.group(2)) and m.group(1) in ("NaN", "Infinity", "-Infinity", "Subnormal", "Division by 0"):
            k = {"Infinity": "Inf", "-Infinity": "-Inf", "Subnormal": "Sub", "Division by 0": "Div0"}.get(m.group(1), m.group(1))
            out.append("%s %s %s(%s)" % (sec, k, m.group(2), m.group(3)))
    return "; ".join(out) or "no exceptions"

def has_report(path):
    return "nixnan Report" in open(path, errors="replace").read()

if os.path.exists("logs/e0/budget.tsv"):
    print("E0 budget (limits: native 60 s, nixnan 120 s)")
    print("%-24s %9s %9s %8s %s" % ("run", "native_s", "nixnan_s", "kernels", "verdict"))
    print(open("logs/e0/budget.tsv").read().rstrip())
    if os.path.exists("logs/e0/mpir_gpu.time"):
        print(open("logs/e0/mpir_gpu.time").read().rstrip())
    print()

def result(path):
    m = re.search(r"RESULT matrix=(\S+) config=(\S+) path=(\S+) n=(\d+) status=(\S+) fact_err=(\S+) resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)",
                  open(path, errors="replace").read())
    return m.groups() if m else None

def kappas():
    k = {}
    if os.path.exists("matrices/matrices.tsv"):
        for line in open("matrices/matrices.tsv").read().splitlines()[1:]:
            f = line.split("\t")
            if f[1] == "256":
                k[f[0]] = f[4]
    return k

runs = {}   # (config, path) -> {matrix: (result tuple, nixnan summary)}
for f in sorted(glob.glob("logs/e1/*_native.log")):
    r = result(f)
    if not r:
        continue
    nx = f.replace("_native", "_nixnan")
    nxs = (nixnan(nx) if has_report(nx) else "NO REPORT") if os.path.exists(nx) else "-"
    name = os.path.basename(f).split("_")[0]
    runs.setdefault((r[1], r[2]), {})[name] = (r, nxs)

if runs:
    kp = kappas()
    mats = sorted({m for v in runs.values() for m in v})
    print("E1 reference, n=256   (" + ", ".join("%s: kappa2 = %s" % (m, kp.get(m, "?")) for m in mats) + ")")
    # four column headings: for each matrix, factorization error and forward error
    cols = "".join(" | %-14s %-14s" % ("%s: A-LL'/A" % m, "%s: x-1/1" % m) for m in mats)
    print("%-16s%s" % ("config path", cols))
    for key in sorted(runs):
        line = "%-16s" % (key[0] + " " + key[1])
        for m in mats:
            r = runs[key].get(m)
            line += " | %-14s %-14s" % ((r[0][5], r[0][7]) if r else ("-", "-"))
        print(line)
    print("\nstatus, first bad pivot, nixnan counters (non-zero only):")
    for key in sorted(runs):
        for m in mats:
            r = runs[key].get(m)
            if r:
                print("  %-3s %-9s %-10s %-6s bad_piv=%-3s %s" % (r[0][1], r[0][2], m, r[0][4], r[0][8], r[1]))
