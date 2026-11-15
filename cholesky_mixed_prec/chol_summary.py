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

rows = sorted(glob.glob("logs/e1/*_native.log"))
if rows:
    print("E1 reference: A_delta, n=256, delta=2^-12")
    print("%-18s %-7s %-10s %-10s %-10s %-9s %s" % ("run", "status", "fact_err", "resid", "fwd_err", "bad_piv", "nixnan"))
    for f in rows:
        tag = os.path.basename(f)[:-len("_native.log")]
        m = re.search(r"status=(\S+) fact_err=(\S+) resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)", open(f).read())
        nx = f.replace("_native", "_nixnan")
        nxs = (nixnan(nx) if has_report(nx) else "NO REPORT") if os.path.exists(nx) else "-"
        print("%-18s %-7s %-10s %-10s %-10s %-9s %s" % ((tag,) + (m.groups() if m else ("?",) * 5) + (nxs,)))
