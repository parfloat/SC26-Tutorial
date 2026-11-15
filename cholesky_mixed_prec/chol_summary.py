#!/usr/bin/env python3
"""Tables from logs/: E0 budget and E1 RESULT lines with nixnan's non-zero counters."""
import glob, os, re, sys

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

# ---------------------------------------------------------------- E2
CODE = {"ok": "ok", "inaccurate": "inacc", "breakdown": "BRK", "nan": "NaN"}

def sweep(path):
    out = {}   # (store, path) -> {e: (status, cause, fact_err, fwd_err, kappa)}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = re.match(r"RESULT .*? config=(\S+) store=(\S+) path=(\S+) n=\d+ log2delta=(-?[\d.]+) kappa=(\S+) status=(\S+) cause=(\S+) "
                     r"stored_changed=\d fact_err=(\S+) fact_err_stored=(\S+) resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)", line)
        if m:
            g = m.groups()
            out.setdefault((g[1], g[2]), {})[-int(float(g[3]))] = (g[5], g[6], g[7], g[10], g[4], g[11])
    return out

def first_fail(d):
    """first e (smallest delta ... in increasing e) with a non-ok status; sweeps are ordered by increasing e"""
    for e in sorted(d):
        if d[e][0] != "ok":
            return e
    return None

def e2_points():
    pts = []
    s32 = sweep("logs/e2/sweep_f32.log")
    for store in ("f32", "tf32", "f16", "bf16"):
        for path in ("unblocked", "blocked"):
            es = {12}
            ff = first_fail(s32.get((store, path), {}))
            if ff is not None:
                es.add(ff)
            for e in sorted(es):
                pts.append((store, e, path))
    return pts

if "--e2-points" in sys.argv:
    for p in e2_points():
        print(*p)
    raise SystemExit

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


def e2_report():
    s32, s64 = sweep("logs/e2/sweep_f32.log"), sweep("logs/e2/sweep_f64.log")
    if not (s32 or s64):
        return
    print("\nE2: sweep of delta = 2^-e on A_delta (n=256).  ok / inacc (error > 1000 eps or forward error > 0.1) / BRK (pivot <= 0 seen) / NaN")
    cols = [(st, pa) for st in ("f32", "tf32", "f16", "bf16") for pa in ("unblocked", "blocked")]
    print("%3s %10s |%s" % ("e", "kappa", "".join(" %-8s" % (st + "-" + pa[0]) for st, pa in cols)) + "   (-u unblocked, -b blocked; FP32 arithmetic)")
    for e in sorted({e for d in s32.values() for e in d}):
        k = next(iter(s32.values()))[e][4]
        print("%3d %10s |%s" % (e, k, "".join(" %-8s" % CODE.get(s32.get(c, {}).get(e, ("?",))[0], "?") for c in cols)))
    print("\nThresholds (first failing e; the failure is at storage if the stored matrix differs from A, else in the arithmetic):")
    print("%-14s %-10s %-9s %-14s %-11s %s" % ("arithmetic", "storage", "path", "first fail e", "kappa", "status / cause"))
    for tag, s in (("FP32", s32), ("FP64", s64)):
        for key in sorted(s):
            ff = first_fail(s[key])
            r = s[key].get(ff) if ff is not None else None
            last_ok = max([e for e in s[key] if e < ff and s[key][e][0] == "ok"], default=None) if ff is not None else None
            print("%-14s %-10s %-9s %-14s %-11s %s" % (tag, key[0], key[1], "none in range" if ff is None else "%d (last ok %s)" % (ff, last_ok),
                                                         r[4] if r else "-", "%s / %s" % (r[0], r[1]) if r else "-"))
    em = {}
    if os.path.exists("logs/e2/emulate.tsv"):
        for line in open("logs/e2/emulate.tsv").read().splitlines()[1:]:
            e, st, r = line.split("\t")
            em.setdefault(st, {})[int(e)] = r
    if em:
        print("\nCPU cross-check (LAPACK float32 on the rounded matrix) vs GPU kernels: e where they disagree on 'pivot <= 0'")
        for st in ("f32", "tf32", "f16", "bf16"):
            dis = []
            for pa in ("unblocked", "blocked"):
                for e, r in em.get(st, {}).items():
                    g = s32.get((st, pa), {}).get(e)
                    if g and ((g[0] == "breakdown") != (r == "NOT-PD")):
                        dis.append("%d(%s: LAPACK %s, GPU %s)" % (e, pa[0], r, g[0]))
            print("  %-5s %s" % (st, ", ".join(dis) if dis else "agree at every e"))
    pts = sorted(glob.glob("logs/e2/confirm_*_native.log"))
    if pts:
        print("\nnixnan on chosen points (FP32 arithmetic, n=256):")
        print("  %-6s %-3s %-10s %-10s %-9s %s" % ("store", "e", "path", "status", "bad_piv", "nixnan (non-zero counters; unique(repeats))"))
        for f in pts:
            m = re.match(r"confirm_(\w+?)_e(\d+)_(\w+)_native", os.path.basename(f))
            r = re.search(r"status=(\S+) cause=\S+ stored_changed=\d fact_err=\S+ fact_err_stored=\S+ resid=\S+ fwd_err=\S+ first_bad_pivot=(\S+)", open(f).read())
            nx = f.replace("_native", "_nixnan")
            nxs = (nixnan(nx) if has_report(nx) else "NO REPORT") if os.path.exists(nx) else "-"
            print("  %-6s %-3s %-10s %-10s %-9s %s" % (m.group(1), m.group(2), m.group(3), r.group(1) if r else "?", r.group(2) if r else "?", nxs))

e2_report()
