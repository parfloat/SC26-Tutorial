#!/usr/bin/env python3
"""Summarise logs/clamp_story/*: per run, the GPU pivot, residual, finite-x count and nixnan's non-zero counters."""
import glob, os, re, sys
d = sys.argv[1] if len(sys.argv) > 1 else "logs/clamp_story"

def native(path):
    t = open(path, errors="replace").read()
    piv = re.search(r"min \|U\[i\]\[i\]\| = (\S+)", t)
    res = re.search(r"Residual norm \|\|Ax - b\|\|: (\S+)", t)
    xs = re.findall(r"^x\[\d+\]: (\S+)", t, re.M)
    fin = [float(v) for v in xs if "nan" not in v and "inf" not in v]
    return (piv.group(1) if piv else "-", res.group(1) if res else "-",
            "%d/%d" % (len(fin), len(xs)), max((abs(v) for v in fin), default=float("nan")))

def nixnan(path):
    sec, out = None, []
    for line in open(path, errors="replace"):
        m = re.match(r"#nixnan: --- (.*?) ---", line)
        if m:
            sec = re.sub(r"\s+", " ", m.group(1)).replace(" Operations", "").replace("Memory ", "mem")
            continue
        m = re.match(r"#nixnan: ([A-Za-z0-9\- ]+?):\s+(\d+) \((\d+) repeats\)", line)
        if m and sec and int(m.group(2)):
            k = {"NaN": "NaN", "Infinity": "Inf", "-Infinity": "-Inf", "Subnormal": "Sub", "Division by 0": "Div0"}[m.group(1)]
            out.append("%s %s %s(%s)" % (sec, k, m.group(2), m.group(3)))
    return "; ".join(out) or "no exceptions"

print("%-30s %-11s %-11s %-8s %-10s %s" % ("run", "min pivot", "residual", "x finite", "max|x|", "nixnan (unique(repeats))"))
for f in sorted(glob.glob(os.path.join(d, "*_native.log")),
                key=lambda p: (("128" in p), "fp32" in p, "clampoff" not in p, "bad" in p)):
    tag = os.path.basename(f)[:-len("_native.log")]
    nx = f.replace("_native", "_nixnan")
    p, r, fin, mx = native(f)
    print("%-30s %-11s %-11s %-8s %-10.3g %s" % (tag, p, r, fin, mx, nixnan(nx) if os.path.exists(nx) else "(no nixnan log)"))
