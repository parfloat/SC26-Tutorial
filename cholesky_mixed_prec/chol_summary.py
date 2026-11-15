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


# ---------------------------------------------------------------- E3
def e3_points():
    """true-half-arithmetic points: the demo delta 2^-12 (breakdown) and the last finite-but-wrong delta before the first breakdown"""
    pts = []
    for exe, f in (("chol_h16", "logs/e3/sweep_h16.log"), ("chol_hbf16", "logs/e3/sweep_hbf16.log")):
        d = sweep(f).get(("f16" if exe == "chol_h16" else "bf16", "unblocked"), {})
        brk = [e for e in sorted(d) if d[e][0] == "breakdown"]
        silent = [e for e in sorted(d) if d[e][0] == "inaccurate" and (not brk or e < brk[0])]
        if silent:
            pts.append((exe, max(silent), "silent"))
        pts.append((exe, 12, "breakdown"))
    return pts

if "--e3-points" in sys.argv:
    for p in e3_points():
        print(*p)
    raise SystemExit


# ---------------------------------------------------------------- E4
def e4_points():
    """tensor-core points per format: the demo delta 2^-12 (breakdown), the last delta below the first breakdown, and M0's delta 2^-6"""
    pts = []
    d = {}
    if os.path.exists("logs/e4/sweep_tc.log"):
        for line in open("logs/e4/sweep_tc.log", errors="replace"):
            m = re.search(r"path=blocked-tc-(\w+) n=\d+ log2delta=(-?\d+) kappa=\S+ status=(\S+)", line)
            if m:
                d.setdefault(m.group(1), {})[-int(m.group(2))] = m.group(3)
    for f in ("f16", "bf16", "tf32"):
        brk = [e for e in sorted(d.get(f, {})) if d[f][e] == "breakdown"]
        pts.append((f, 12, "breakdown"))
        if brk and brk[0] > 2:
            pts.append((f, brk[0] - 1, "lastok"))
        pts.append((f, 6, "M0"))
    return sorted(set(pts), key=lambda p: (p[0], p[1]))

if "--e4-points" in sys.argv:
    for p in e4_points():
        print(*p)
    raise SystemExit

# ---------------------------------------------------------------- E5
E5_CONFIGS = ["F32", "mixed-F16", "mixed-BF16", "mixed-TF32", "naive-F16"]

def e5_cases(path="logs/e5/sweep.log"):
    """{(config, kappa, seed): dict(status, fact_err, resid, fwd_err, first_bad_pivot)}"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        m = re.match(r"CASE config=(\S+) kappa=(\S+) seed=(\d+) RESULT .*? status=(\S+) cause=\S+ stored_changed=\d "
                     r"fact_err=(\S+) fact_err_stored=\S+ resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)(?: t_ms=\S+ min_rel_pivot=(\S+))?", line)
        if m:
            g = m.groups()
            out[(g[0], g[1], int(g[2]))] = dict(status=g[3], fact_err=g[4], resid=g[5], fwd_err=g[6], bad=g[7], minrel=g[8])
    return out

def e5_points():
    """nixnan points: per configuration the first (lowest kappa, lowest seed) case of every status that occurs; the all-ok
    configurations contribute one case. mixed-BF16 and mixed-TF32 contribute only their non-ok statuses."""
    cases = e5_cases(); pts = []
    kap = sorted({k for (_, k, _) in cases}, key=float)
    for c in E5_CONFIGS:
        seen = set()
        for k in kap:
            for sd in sorted({s for (_, _, s) in cases}):
                r = cases.get((c, k, sd))
                if r and r["status"] not in seen:
                    seen.add(r["status"])
                    if r["status"] != "ok" or c in ("F32", "mixed-F16", "naive-F16"):
                        pts.append((c, k, sd))
    return pts

if "--e5-points" in sys.argv:
    for p in e5_points():
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


def detail(path):
    """first exception, and kernels in order of first appearance with counts, from a PRINT_ILL_INSTR log"""
    first, order, kinds, types = None, [], {}, {}
    pat = re.compile(r"error \[([^\]]+)\] detected in operand (\d+) of instruction (.*?) ; in function (?:void )?(\w+)(?: at (\S+):(\d+))? of type (\w+)")
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        kind, op, ins, fn, f, ln, ty = m.groups()
        if first is None:
            first = (kind, ins.strip(), fn, "%s:%s" % (os.path.basename(f), ln) if f else "no line info", ty)
        order.append(fn)
        kinds[kind] = kinds.get(kind, 0) + 1
        types[ty] = types.get(ty, 0) + 1
    chain, seen = [], {}
    for fn in order:
        if fn not in seen:
            seen[fn] = len(chain); chain.append([fn, 0])
        chain[seen[fn]][1] += 1
    return first, chain, kinds, types, len(order)

def e3_report():
    runs = sorted(glob.glob("logs/e3/*_native.log"))
    if not runs:
        return
    print("\nE3: naive reduced precision.  a = FP16/BF16 storage, FP32 arithmetic;  b = true FP16/BF16 arithmetic (unblocked).")
    for tag, f in (("FP16 arithmetic sweep (A_delta, n=256)", "logs/e3/sweep_h16.log"), ("BF16 arithmetic sweep", "logs/e3/sweep_hbf16.log")):
        d = sweep(f)
        if d:
            (key, v), = d.items()
            print("  %-40s %s" % (tag, " ".join("%d:%s" % (e, CODE[v[e][0]]) for e in sorted(v))))
    print("  (e: delta = 2^-e; ok / inacc / BRK)")
    for f in runs:
        tag = os.path.basename(f)[:-len("_native.log")]
        r = re.search(r"config=(\S+) store=(\S+) path=(\S+) n=\d+ log2delta=(-?\d+) kappa=\S+ status=(\S+) cause=(\S+) stored_changed=\d fact_err=(\S+) "
                      r"fact_err_stored=\S+ resid=\S+ fwd_err=(\S+) first_bad_pivot=(\S+)", open(f).read())
        print("\n  %s" % tag)
        if r:
            print("    %s store=%s %s delta=2^%s: status %s (%s), fact_err %s, fwd_err %s, first bad pivot %s" % r.groups())
        nx = f.replace("_native", "_nixnan")
        print("    nixnan report: %s" % (nixnan(nx) if os.path.exists(nx) and has_report(nx) else "-"))
        dt = f.replace("_native", "_detail")
        if os.path.exists(dt):
            first, chain, kinds, types, total = detail(dt)
            if first:
                print("    first exception: [%s] %s in %s (%s) type %s" % first)
                print("    %d detail lines; by kind %s; by type %s" % (total, dict(sorted(kinds.items())), dict(sorted(types.items()))))
                print("    chain (kernel: lines): " + " -> ".join("%s: %d" % (k, n) for k, n in chain))
            else:
                print("    no exception lines")

e3_report()


def e4_report():
    if not os.path.exists("logs/e4/sweep_tc.log"):
        return
    print("\nE4: tensor-core trailing update (WMMA inputs FP16 / BF16 / TF32, FP32 accumulate; FP32 panel), A_delta n=256, NB=32")
    if os.path.exists("logs/e4/sass_hmma.txt"):
        print("  SASS: " + "; ".join(l.split(None, 1)[1] + " x" + l.split()[0] for l in open("logs/e4/sass_hmma.txt").read().splitlines()))
    d = {}
    for line in open("logs/e4/sweep_tc.log", errors="replace"):
        m = re.search(r"path=blocked-tc-(\w+) n=\d+ log2delta=(-?\d+) kappa=(\S+) status=(\S+) .*fact_err=(\S+) fact_err_stored=\S+ resid=\S+ fwd_err=(\S+) first_bad_pivot=(\S+)", line)
        if m:
            f, e, k, st, fe, fw, bp = m.groups()
            d.setdefault(-int(e), {})["kappa"] = k
            d[-int(e)][f] = (st, fe, fw, bp)
    print("  %3s %10s | %-22s | %-22s | %-22s   (status, factorization error, forward error; BRK = first bad pivot)" % ("e", "kappa", "f16", "bf16", "tf32"))
    for e in sorted(d):
        cells = []
        for f in ("f16", "bf16", "tf32"):
            st, fe, fw, bp = d[e][f]
            cells.append("%-5s %-8s %-8s" % (CODE.get(st, st), "BRK@" + bp if st == "breakdown" else fe, "" if st == "breakdown" else fw))
        print("  %3d %10s | %s" % (e, d[e]["kappa"], " | ".join("%-22s" % c for c in cells)))
    print("\n  nixnan on chosen points:")
    for f in sorted(glob.glob("logs/e4/*_native.log")):
        tag = os.path.basename(f)[:-len("_native.log")]
        r = re.search(r"path=blocked-tc-(\w+) n=\d+ log2delta=(-?\d+) kappa=(\S+) status=(\S+) .*fact_err=(\S+) fact_err_stored=\S+ resid=\S+ fwd_err=(\S+) first_bad_pivot=(\S+)", open(f).read())
        nx = f.replace("_native", "_nixnan"); dt = f.replace("_native", "_detail")
        print("  %s: status %s, fwd %s, bad pivot %s%s" % (tag, r.group(4), r.group(6), r.group(7),
              " (block step k=%d)" % (int(r.group(7)) // 32) if r.group(7) != "-1" else ""))
        print("      report: %s" % (nixnan(nx) if os.path.exists(nx) and has_report(nx) else "-"))
        if os.path.exists(dt):
            first, chain, kinds, types, total = detail(dt)
            if first:
                hm = sum(1 for l in open(dt, errors="replace") if "error [" in l and "HMMA" in l)
                print("      first exception: [%s] %s in %s; chain: %s; HMMA lines: %d" % (first[0], first[1], first[2], " -> ".join("%s:%d" % (k, n) for k, n in chain), hm))

e4_report()


def sqrt_report():
    if not glob.glob("logs/sqrt/*_detail.log"):
        return
    print("\nsqrt study: IEEE sqrtf (MUFU.RSQ + range test + slow path) vs -prec-sqrt=false (MUFU.SQRT)")
    for k, what in (("e3a", "E3a: FP16 storage, pivot 1 is exactly 0 -> sqrt(0)"), ("e4", "E4: tensor-core update, a Schur diagonal is negative -> sqrt(<0)")):
        for b in ("ieee", "fast"):
            dt = "logs/sqrt/%s_%s_detail.log" % (k, b)
            if not os.path.exists(dt):
                continue
            first, chain, kinds, types, total = detail(dt)
            sq = []
            for line in open(dt, errors="replace"):
                m = re.search(r"error \[(\w+)\] detected in operand (\d) of instruction ((?:@\S+ )?(?:MUFU\.(?:RSQ|SQRT)|FADD\.FTZ R\d+, R\d+, 1)[^;]*) ; in function (?:void )?(pivot_kernel|potrf_tile_kernel)", line)
                if m:
                    sq.append("%s/op%s %s" % (m.group(1), m.group(2), m.group(3).strip()))
            print("  %-4s %-5s first: [%s] %s in %s; %d lines; sqrt-site lines: %s" % (k, b, first[0], first[1], first[2], total, "; ".join(sq[:6]) or "none"))
    for b in ("ieee", "fast"):
        f = "logs/sqrt/sass_pivot_%s.txt" % b
        if os.path.exists(f):
            print("  pivot_kernel SASS (%s): %s" % (b, ", ".join(sorted({re.sub(r"^@!?P\d ", "", l.split(";")[0].split("*/")[1].strip()).split()[0] for l in open(f) if "*/" in l}))))

sqrt_report()


def e5_report():
    cases = e5_cases()
    if not cases:
        return
    import statistics
    kap = sorted({k for (_, k, _) in cases}, key=float)
    ab = {"ok": "ok", "inaccurate": "inacc", "breakdown": "BRK", "nan": "NaN"}
    print("\nE5: random Q diag(sigma) Q^T, n=256, 5 seeds per cell. Cell = status counts; second line = median forward error (finite runs)")
    print("%-11s" % "config" + "".join("%-22s" % ("kappa " + k) for k in kap))
    for c in E5_CONFIGS:
        row1, row2 = "%-11s" % c, "%-11s" % ""
        for k in kap:
            rs = [cases[(c, k, s)] for s in range(5) if (c, k, s) in cases]
            cnt = {}
            for r in rs:
                cnt[r["status"]] = cnt.get(r["status"], 0) + 1
            row1 += "%-22s" % " ".join("%d %s" % (n, ab.get(st, st)) for st, n in sorted(cnt.items(), key=lambda t: -t[1]))
            fw = [float(r["fwd_err"]) for r in rs if r["fwd_err"] not in ("nan", "inf")]
            row2 += "%-22s" % ("%.1e" % statistics.median(fw) if fw else "-")
        print(row1); print(row2)
    print("E5 nixnan on chosen points (counters unique(repeats); native status in brackets)")
    for c, k, s in e5_points():
        f = "logs/e5/%s_k%s_s%d_nixnan.log" % (c, k, s)
        if os.path.exists(f):
            print("  %-11s kappa %-4s seed %d [%s]: %s" % (c, k, s, cases[(c, k, s)]["status"], nixnan(f)))

e5_report()


def e6_report():
    if not glob.glob("logs/e6/*_hist.log"):
        return
    print("\nE6: nixnan histogram, danger-zone view. Cell = approx. occurrences (threshold reports x count, count=%s) of values with that exponent" % "64")
    for sc, what, order in (("s1", "FP32 arithmetic on FP16-rounded A (chol_f32 --store f16, unblocked)", "f32"),
                            ("s2", "tensor-core update (chol_tc --fmt f16)", "f32"),
                            ("s3", "true FP16 arithmetic (chol_h16)", "f16")):
        print(" %s: %s" % (sc, what))
        for e in (6, 12):
            f = "logs/e6/%s_e%d_hist.log" % (sc, e)
            if not os.path.exists(f):
                continue
            tab = {}
            for line in open(f, errors="replace"):
                m = re.search(r"(f32|f16) bin has reached threshold: function=(?:void )?([a-z_0-9]+).*? range=\[([-\w]+),([-\w]+)\] count=(\d+)", line)
                if m:
                    key = (m.group(2), "%s..%s" % (m.group(3), m.group(4)) if m.group(3) != m.group(4) else m.group(3))
                    tab[key] = tab.get(key, 0) + 1
            kern = sorted({k for k, _ in tab})
            print("   delta=2^-%d" % e)
            for k in kern:
                print("     %-14s %s" % (k, "  ".join("[%s]:%d" % (b, n * 64) for (kk, b), n in sorted(tab.items(), key=lambda t: t[0][1]) if kk == k)))
    if os.path.exists("logs/e6/options.tsv"):
        print(" cost of the options (s1, delta=2^-12, wall s): " + "; ".join(l.replace(" ", "=") for l in open("logs/e6/options.tsv").read().split("\n") if l))
        for lab in ("report", "histogram", "hist_sampling4", "whitelist_rank1", "hist_whitelist", "instr_mem"):
            f = "logs/e6/opt_%s.log" % lab
            if os.path.exists(f):
                t = open(f, errors="replace").read()
                print("   %-16s threshold reports %-6d exception lines %d" % (lab, t.count("reached threshold"), t.count("error [")))

e6_report()


def e7_report():
    if not glob.glob("logs/e7/*_plain_native.log"):
        return
    U = {"F32": 2.0 ** -24, "mixed-F16": 2.0 ** -11, "mixed-TF32": 2.0 ** -11, "naive-F16": 2.0 ** -11, "mixed-BF16": 2.0 ** -8}
    cases = e5_cases()
    print("\nE7a: guard threshold tau = c u, evaluated on the 150 E5 runs from min_rel_pivot (a run is flagged if min_rel_pivot <= c u)")
    print("  Cell = non-ok runs flagged / non-ok runs, and ok runs flagged / ok runs (false alarms)")
    cs = (1, 10, 30, 100, 300)
    print("  %-11s" % "config" + "".join("%-22s" % ("c=%d" % c) for c in cs))
    for cf in E5_CONFIGS:
        rows = [r for (c, k, s), r in cases.items() if c == cf and r.get("minrel")]
        row = "  %-11s" % cf
        for c in cs:
            tau = c * U[cf]
            bad = [r for r in rows if r["status"] != "ok"]; ok = [r for r in rows if r["status"] == "ok"]
            row += "%-22s" % ("%d/%d, %d/%d" % (sum(float(r["minrel"]) <= tau for r in bad), len(bad), sum(float(r["minrel"]) <= tau for r in ok), len(ok)))
        print(row)
    print("E7b: live runs, tau = 100 u. Native status without / with the guard; nixnan counters without / with the guard")
    for plain in sorted(glob.glob("logs/e7/*_plain_native.log")):
        tag = os.path.basename(plain)[:-len("_plain_native.log")]
        txt = open(plain, errors="replace").read(); g = open(plain.replace("_plain_", "_guard_"), errors="replace").read()
        pm = re.search(r"status=(\S+).*?fwd_err=(\S+) first_bad_pivot=(\S+)", txt); gm = re.search(r"GUARD pivot (\d+): relative pivot (\S+) <= tau ([^\s;]+)", g)
        gs = re.search(r"status=(\S+)", g)
        def nx(f):
            return nixnan(f) if os.path.exists(f) and has_report(f) else "?"
        print("  %-22s plain: %s fwd=%s bad_pivot=%s | guard: %s | nixnan plain: %s | nixnan guard: %s" % (
            tag, pm.group(1) if pm else "?", pm.group(2) if pm else "?", pm.group(3) if pm else "?",
            ("status=%s, pivot %s rel %s <= %s" % (gs.group(1), gm.group(1), gm.group(2), gm.group(3))) if gm else "status=%s (no trip)" % (gs.group(1) if gs else "?"),
            nx(plain.replace("_native", "_nixnan")), nx(plain.replace("_plain_native", "_guard_nixnan"))))
    if os.path.exists("logs/e7/heldout.log"):
        print("E7c: held-out kappa (150..700, seeds 10-12): status, forward error, min_rel_pivot / u, and whether tau = 100 u / 300 u would trip")
        for line in open("logs/e7/heldout.log", errors="replace"):
            m = re.match(r"HELD config=(\S+) kappa=(\S+) seed=(\d+) RESULT .*? status=(\S+) .*fwd_err=(\S+) first_bad_pivot=\S+ t_ms=\S+ min_rel_pivot=(\S+)", line)
            if m:
                c, k, sd, st, fw, mr = m.groups(); r = float(mr) / U[c]
                print("  %-10s kappa %-4s seed %s: %-10s fwd %-9s min_rel/u = %6.0f   trip@100u: %-3s trip@300u: %s" % (c, k, sd, st, fw, r, "yes" if r <= 100 else "no", "yes" if r <= 300 else "no"))
    if os.path.exists("logs/e7/cost.tsv"):
        print("  cost of the per-step check: " + "; ".join(open("logs/e7/cost.tsv").read().split("\n")))

e7_report()
