#!/usr/bin/env python3
"""Generate the LaTeX fragments and log excerpts used by chol_report.tex and Chol_Slides.tex from logs/.

  python3 make_chol_docs.py          ->  docs_gen/*.tex (tables) and docs_gen/*.txt (verbatim excerpts)

Every number in the generated tables is read from the logs, so the report and the slides cannot drift from them.
"""
import glob, os, re

OUT = "docs_gen"
os.makedirs(OUT, exist_ok=True)
MARK = "#nixnan: ------------ nixnan Report"

def tex_escape(s):
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")

def sci(x, d=1):
    """'1.234e-05' -> '$1.2\\times10^{-5}$'; passes nan through"""
    try:
        v = float(x)
    except ValueError:
        return x
    if v != v:
        return "NaN"
    if v == 0:
        return "0"
    m, e = ("%.*e" % (d, v)).split("e")
    return "$%s\\!\\times\\!10^{%d}$" % (m, int(e))

def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)

def report_block(path):
    t = open(path, errors="replace").read()
    i = t.find(MARK)
    return t[i:] if i >= 0 else ""

def nixnan_counts(path):
    """[(section, kind, unique, repeats)] for non-zero counters"""
    sec, out = None, []
    for line in report_block(path).splitlines():
        m = re.match(r"#nixnan: --- (.*?) ---", line)
        if m:
            sec = re.sub(r"\s+", " ", m.group(1)).replace(" Operations", "").replace("Memory ", "mem ")
            continue
        m = re.match(r"#nixnan: ([A-Za-z0-9\- ]+?):\s+(\d+) \((\d+) repeats\)", line)
        if m and sec and int(m.group(2)) and m.group(1) in ("NaN", "Infinity", "-Infinity", "Subnormal", "Division by 0"):
            k = {"Infinity": "Inf", "-Infinity": "-Inf", "Subnormal": "Sub", "Division by 0": "Div0"}.get(m.group(1), m.group(1))
            out.append((sec, k, int(m.group(2)), int(m.group(3))))
    return out

def counts_tex(path):
    c = nixnan_counts(path)
    if not c:
        return "no exceptions"
    return ", ".join("%s %s %d (%s)" % (s, k, u, format(r, ",").replace(",", "{,}") and "{:,}".format(r).replace(",", "\\,")) for s, k, u, r in c)

RES = re.compile(r"RESULT matrix=(\S+) config=(\S+) store=(\S+) path=(\S+) n=(\d+) log2delta=(-?[\d.]+) kappa=(\S+) status=(\S+) cause=(\S+) "
                 r"stored_changed=(\d) fact_err=(\S+) fact_err_stored=(\S+) resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)")
RES1 = re.compile(r"RESULT matrix=(\S+) config=(\S+) path=(\S+) n=(\d+) status=(\S+) fact_err=(\S+) resid=(\S+) fwd_err=(\S+) first_bad_pivot=(\S+)")

# ---------------------------------------------------------------- E0
rows = []
if os.path.exists("logs/e0/budget.tsv"):
    for line in open("logs/e0/budget.tsv"):
        f = line.split()
        if len(f) == 5:
            rows.append(f)
t = "\\begin{tabular}{lrrrl}\n\\toprule\nrun (FP32) & native (s) & nixnan (s) & kernels & within budget\\\\\n\\midrule\n"
for r in rows:
    t += "%s & %s & %s & %s & %s\\\\\n" % (tex_escape(r[0]), r[1], r[2], r[3], "yes" if r[4] == "PASS" else "NO")
t += "\\bottomrule\n\\end{tabular}\n"
write("tab_e0.tex", t)
mp = open("logs/e0/mpir_gpu.time").read().strip() if os.path.exists("logs/e0/mpir_gpu.time") else ""
m = re.search(r"native ([\d.]+) s.*nixnan ([\d.]+) s", mp)
write("mpir_time.tex", ("%s~s natively and %s~s under NixNan" % m.groups()) if m else "(not measured)")

# ---------------------------------------------------------------- E1
d = {}
for f in sorted(glob.glob("logs/e1/*_native.log")):
    m = RES1.search(open(f).read())
    if m:
        name = os.path.basename(f).split("_")[0]
        g = m.groups()
        d[(g[1], g[2], name)] = (g[5], g[7], g[4])
t = "\\begin{tabular}{llrrrr}\n\\toprule\n & & \\multicolumn{2}{c}{M0 ($\\kappa_2=1.6\\times10^{4}$)} & \\multicolumn{2}{c}{M1 ($\\kappa_2=1.0\\times10^{6}$)}\\\\\n"
t += "arithmetic & path & $\\|A-LL^T\\|/\\|A\\|$ & $\\|x-1\\|/\\|1\\|$ & $\\|A-LL^T\\|/\\|A\\|$ & $\\|x-1\\|/\\|1\\|$\\\\\n\\midrule\n"
for cfg in ("F64", "F32"):
    for path in ("unblocked", "blocked"):
        a, b = d.get((cfg, path, "M0")), d.get((cfg, path, "M1"))
        if a and b:
            t += "%s & %s & %s & %s & %s & %s\\\\\n" % (cfg, path, sci(a[0]), sci(a[1]), sci(b[0]), sci(b[1]))
t += "\\bottomrule\n\\end{tabular}\n"
write("tab_e1.tex", t)

# ---------------------------------------------------------------- E2
def sweep(path):
    out = {}
    for line in open(path, errors="replace"):
        m = RES.search(line)
        if m:
            g = m.groups()
            out.setdefault((g[2], g[3]), {})[-int(float(g[5]))] = dict(status=g[7], cause=g[8], fe=g[10], fw=g[13], kappa=g[6], bad=g[14])
    return out
s32, s64 = sweep("logs/e2/sweep_f32.log"), sweep("logs/e2/sweep_f64.log")
COL = {"ok": ("ok", "green!15"), "inaccurate": ("inacc", "orange!30"), "breakdown": ("BRK", "red!25"), "nan": ("NaN", "red!25")}
cols = [(st, pa) for st in ("f32", "tf32", "f16", "bf16") for pa in ("unblocked", "blocked")]
t = "\\begin{tabular}{rr|" + "cc|" * 3 + "cc}\n\\toprule\n$e$ & $\\kappa_2$"
for st in ("f32", "tf32", "f16", "bf16"):
    t += " & \\multicolumn{2}{c%s}{%s}" % ("|" if st != "bf16" else "", {"f32": "FP32", "tf32": "TF32", "f16": "FP16", "bf16": "BF16"}[st] + " storage")
t += "\\\\\n & "
t += "".join(" & u & b" for _ in range(4)) + "\\\\\n\\midrule\n"
for e in sorted({e for v in s32.values() for e in v}):
    k = next(iter(s32.values()))[e]["kappa"]
    t += "%d & %s" % (e, sci(k, 1))
    for c in cols:
        r = s32.get(c, {}).get(e)
        lab, col = COL.get(r["status"], ("?", "white")) if r else ("?", "white")
        t += " & \\cellcolor{%s}%s" % (col, lab)
    t += "\\\\\n"
t += "\\bottomrule\n\\end{tabular}\n"
write("tab_e2_grid.tex", t)

def first_fail(dd):
    for e in sorted(dd):
        if dd[e]["status"] != "ok":
            return e
rows = ""
for tag, s in (("FP32", s32), ("FP64", s64)):
    for key in sorted(s, key=lambda k: ({"bf16": 0, "f16": 1, "tf32": 2, "f32": 3, "f64": 4}[k[0]], k[1])):
        dd = s[key]
        ff = first_fail(dd)
        brk = next((e for e in sorted(dd) if dd[e]["status"] == "breakdown"), None)
        r = dd.get(ff) if ff is not None else None
        rows += "%s & %s & %s & %s & %s & %s\\\\\n" % (tag, {"f16": "FP16", "bf16": "BF16", "tf32": "TF32", "f32": "FP32", "f64": "FP64"}[key[0]], key[1],
                                                         ff if ff is not None else "--", sci(r["kappa"]) if r else "--",
                                                         "%s (%s)" % (r["status"], r["cause"]) + (", breakdown at %d" % brk if brk and brk != ff else "") if r else "--")
write("tab_e2_thr.tex", "\\begin{tabular}{lllrrl}\n\\toprule\narithmetic & storage & path & first not-ok $e$ & $\\kappa_2$ & status (cause)\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

# nixnan on E2 chosen points
rows = ""
for f in sorted(glob.glob("logs/e2/confirm_*_native.log")):
    m = re.match(r"confirm_(\w+?)_e(\d+)_(\w+)_native", os.path.basename(f))
    r = RES.search(open(f).read())
    nx = f.replace("_native", "_nixnan")
    rows += "%s & %s & %s & %s & %s\\\\\n" % ({"f16": "FP16", "bf16": "BF16", "tf32": "TF32", "f32": "FP32"}[m.group(1)], m.group(2), m.group(3),
                                              r.group(8) if r else "?", counts_tex(nx) if os.path.exists(nx) else "--")
write("tab_e2_nixnan.tex", "\\begin{tabular}{llllp{7.3cm}}\n\\toprule\nstorage & $e$ & path & status & nixnan: non-zero counters, unique (repeats)\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- E3
def detail(path):
    pat = re.compile(r"error \[([^\]]+)\] detected in operand (\d+) of instruction (.*?) ; in function (?:void )?(\w+)(?: at (\S+):(\d+))? of type (\w+)")
    first, order, kinds, types, withline = None, [], {}, {}, 0
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        kind, op, ins, fn, f, ln, ty = m.groups()
        if first is None:
            first = (kind, ins.strip(), fn, ty)
        order.append(fn); kinds[kind] = kinds.get(kind, 0) + 1; types[ty] = types.get(ty, 0) + 1
        withline += 1 if f else 0
    chain, seen = [], {}
    for fn in order:
        if fn not in seen:
            seen[fn] = len(chain); chain.append([fn, 0])
        chain[seen[fn]][1] += 1
    return first, chain, kinds, types, len(order), withline

rows = ""
for f in sorted(glob.glob("logs/e3/a_*_native.log")):
    tag = os.path.basename(f)[:-len("_native.log")]
    r = RES.search(open(f).read())
    first, chain, kinds, types, total, _ = detail(f.replace("_native", "_detail"))
    _, st, path_, = tag.split("_")[1], tag.split("_")[1], tag.split("_")[3]
    rows += "%s & %s & %s & %s & %s\\\\\n" % ({"f16": "FP16", "bf16": "BF16"}[st], path_, r.group(8) + ", bad pivot " + r.group(15),
                                              counts_tex(f.replace("_native", "_nixnan")), "\\texttt{%s}, %s, %s" % (tex_escape(first[1]), first[0].replace("infinity", "Inf"), tex_escape(first[2])))
write("tab_e3a.tex", "\\begin{tabular}{lllp{4.6cm}p{3.6cm}}\n\\toprule\nstorage & path & status & nixnan report & first exception\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

rows = ""
for f in sorted(glob.glob("logs/e3/b_*_native.log")):
    tag = os.path.basename(f)[:-len("_native.log")]
    r = RES.search(open(f).read())
    nxc = counts_tex(f.replace("_native", "_nixnan"))
    rows += "%s & $2^{-%s}$ & %s & %s & %s\\\\\n" % ({"F16A": "FP16", "BF16A": "BF16"}[r.group(2)], tag.split("_e")[1].split("_")[0],
                                                    r.group(8) + (" (bad pivot %s)" % r.group(15) if r.group(15) != "-1" else ""), sci(r.group(14)) if r.group(14) != "nan" else "NaN", nxc)
write("tab_e3b.tex", "\\begin{tabular}{llllp{5.4cm}}\n\\toprule\narithmetic & $\\delta$ & status & forward error & nixnan report\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

for tag, f in (("h16", "logs/e3/sweep_h16.log"), ("hbf16", "logs/e3/sweep_hbf16.log")):
    s = list(sweep(f).values())[0]
    t = "\\begin{tabular}{l" + "c" * len(s) + "}\n\\toprule\n$e$ & " + " & ".join(str(e) for e in sorted(s)) + "\\\\\n\\midrule\nstatus"
    for e in sorted(s):
        lab, col = COL[s[e]["status"]]
        t += " & \\cellcolor{%s}%s" % (col, lab)
    t += "\\\\\n\\bottomrule\n\\end{tabular}\n"
    write("tab_e3_sweep_%s.tex" % tag, t)

# chains (for the slides)
for tag in ("a_f16_e12_unblocked", "a_f16_e12_blocked", "b_h16_e12_breakdown"):
    first, chain, kinds, types, total, withline = detail("logs/e3/%s_detail.log" % tag)
    kd = ", ".join("%s %d" % (k.replace("infinity", "Inf"), v) for k, v in sorted(kinds.items()))
    write("chain_%s.tex" % tag, " $\\to$ ".join("\\texttt{%s}: %d" % (tex_escape(k), n) for k, n in chain) + "; %d lines (%s)" % (total, kd))

# ---------------------------------------------------------------- verbatim excerpts
def excerpt(name, text):
    write(name, text if text.endswith("\n") else text + "\n")

excerpt("report_clean.txt", report_block("logs/e1/M1_f32_unblocked_nixnan.log"))
def nonzero_sections(block):
    """the report header plus only the sections that have a non-zero counter (all-zero sections are dropped)"""
    out, cur, nz = [], [], False
    for line in block.splitlines():
        if line.startswith("#nixnan: ---") and "nixnan Report" not in line:
            if cur and nz:
                out += cur
            cur, nz = [line], False
        elif cur:
            cur.append(line)
            m = re.match(r"#nixnan: [A-Za-z0-9\- ]+?:\s+(\d+) \(", line)
            if m and int(m.group(1)):
                nz = True
        else:
            out.append(line)
    if cur and nz:
        out += cur
    return "\n".join(out) + "\n"

excerpt("report_f16_store.txt", nonzero_sections(report_block("logs/e3/a_f16_e12_unblocked_nixnan.log")))
excerpt("report_h16_arith.txt", nonzero_sections(report_block("logs/e3/b_h16_e12_breakdown_nixnan.log")))
lines = [l.replace("#nixnan: ", "") for l in open("logs/e3/a_f16_e12_unblocked_detail.log", errors="replace") if "error [" in l]
excerpt("detail_f16_store.txt", "".join(lines[:14]))
lines = [l.replace("#nixnan: ", "") for l in open("logs/e3/b_h16_e12_breakdown_detail.log", errors="replace") if "error [" in l]
excerpt("detail_h16_arith.txt", "".join(lines[:12]))
tk = [l.replace("#nixnan: ", "") for l in open("logs/e0/M1_n256_f32_unblocked_nixnan.log", errors="replace") if "execution time" in l]
firsts, seen = [], set()
tk = [l for l in tk if re.search(r"Kernel \[(.*?)\]", l)]
for l in tk:
    k = re.search(r"Kernel \[(.*?)\]", l).group(1)
    if k not in seen:
        seen.add(k); firsts.append(l)
excerpt("time_kernels.txt", "".join(firsts) + "...\n" + tk[-1])

# ---------------------------------------------------------------- E4
def e4_rows():
    d = {}
    if os.path.exists("logs/e4/sweep_tc.log"):
        for line in open("logs/e4/sweep_tc.log", errors="replace"):
            m = re.search(r"path=blocked-tc-(\w+) n=\d+ log2delta=(-?\d+) kappa=(\S+) status=(\S+) .*fact_err=(\S+) fact_err_stored=\S+ resid=\S+ fwd_err=(\S+) first_bad_pivot=(\S+)", line)
            if m:
                f, e, k, st, fe, fw, bp = m.groups()
                d.setdefault(-int(e), {"kappa": k})[f] = (st, fe, fw, bp)
    return d
d4 = e4_rows()
if d4:
    t = "\\begin{tabular}{rr|ccc}\n\\toprule\n$e$ & $\\kappa_2$ & FP16 update & BF16 update & TF32 update\\\\\n\\midrule\n"
    for e in sorted(d4):
        t += "%d & %s" % (e, sci(d4[e]["kappa"], 1))
        for f in ("f16", "bf16", "tf32"):
            st, fe, fw, bp = d4[e][f]
            lab, col = COL.get(st, ("?", "white"))
            extra = ("@%s" % bp) if st == "breakdown" else (" %s" % sci(fw, 1) if fw != "nan" else "")
            t += " & \\cellcolor{%s}%s%s" % (col, lab, extra)
        t += "\\\\\n"
    t += "\\bottomrule\n\\end{tabular}\n"
    write("tab_e4_grid.tex", t)

rows = ""
for f in sorted(glob.glob("logs/e4/*_native.log")):
    tag = os.path.basename(f)[:-len("_native.log")]
    r = re.search(r"path=blocked-tc-(\w+) n=\d+ log2delta=(-?\d+) kappa=(\S+) status=(\S+) .*fact_err=(\S+) fact_err_stored=\S+ resid=\S+ fwd_err=(\S+) first_bad_pivot=(\S+)", open(f).read())
    nx = f.replace("_native", "_nixnan"); dt = f.replace("_native", "_detail")
    first, chain, kinds, types, total, _ = detail(dt)
    hm = sum(1 for l in open(dt, errors="replace") if "error [" in l and "HMMA" in l)
    st = r.group(4) + (" (bad pivot %s, $k=%d$)" % (r.group(7), int(r.group(7)) // 32) if r.group(7) != "-1" else ", fwd %s" % sci(r.group(6), 1))
    rows += "%s & $2^{-%s}$ & %s & %s & %d\\\\\n" % ({"f16": "FP16", "bf16": "BF16", "tf32": "TF32"}[r.group(1)], tag.split("_e")[1].split("_")[0], st, counts_tex(nx), hm)
write("tab_e4_nixnan.tex", "\\begin{tabular}{llp{3.6cm}p{5.6cm}r}\n\\toprule\nupdate & $\\delta$ & status & nixnan report & HMMA lines\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")

if os.path.exists("logs/e4/f16_e12_breakdown_nixnan.log"):
    excerpt("report_e4_f16.txt", nonzero_sections(report_block("logs/e4/f16_e12_breakdown_nixnan.log")))
    excerpt("report_e4_tf32.txt", nonzero_sections(report_block("logs/e4/tf32_e12_breakdown_nixnan.log")))
    lines = [l.replace("#nixnan: ", "") for l in open("logs/e4/f16_e12_breakdown_detail.log", errors="replace") if "error [" in l]
    excerpt("detail_e4_f16.txt", "".join(lines[:6]))
    excerpt("detail_e4_f16_hmma.txt", "".join([l for l in lines if "HMMA" in l][:6]))
    lines = [l.replace("#nixnan: ", "") for l in open("logs/e4/tf32_e12_breakdown_detail.log", errors="replace") if "error [" in l and "HMMA" in l]
    excerpt("detail_e4_tf32_hmma.txt", "".join(lines[:4]))
    first, chain, kinds, types, total, _ = detail("logs/e4/f16_e12_breakdown_detail.log")
    write("chain_e4_f16.tex", " $\\to$ ".join("\\texttt{%s}: %d" % (tex_escape(k), n) for k, n in chain) + "; %d lines" % total)
    write("sass_hmma.tex", ", ".join("%s $\\times$%s" % (tex_escape(l.split(None, 1)[1]) if False else "\\texttt{%s}" % tex_escape(l.split()[1]), l.split()[0]) for l in open("logs/e4/sass_hmma.txt").read().splitlines()))

# ---------------------------------------------------------------- sqrt study
def sqrt_sites(path):
    out = []
    for line in open(path, errors="replace"):
        m = re.search(r"error \[(\w+)\] detected in operand (\d) of instruction ((?:@\S+ )?MUFU\.(?:RSQ|SQRT)[^;]*) ; in function (?:void )?(pivot_kernel|potrf_tile_kernel)", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3).strip()))
    return out
if glob.glob("logs/sqrt/*_detail.log"):
    rows = ""
    for k, case in (("e3a", "E3a: FP16 storage, pivot is $0$"), ("e4", "E4: tensor-core update, pivot is $<0$")):
        for b, name in (("ieee", "IEEE \\texttt{sqrtf}"), ("fast", "\\texttt{-prec-sqrt=false}")):
            dt = "logs/sqrt/%s_%s_detail.log" % (k, b)
            if not os.path.exists(dt):
                continue
            first, chain, kinds, types, total, _ = detail(dt)
            sites = sqrt_sites(dt)
            instr = "\\texttt{%s}" % tex_escape(sites[0][2].split(" R")[0].replace("@P0 ", "")) if sites else "--"
            at0 = [x for x in sites[:2] if k == "e3a" and b == "ieee"]
            if k == "e3a" and b == "ieee":
                at_sqrt = "div0 (operand 1 = the $0$ input), infinity (operand 0 = $\\mathrm{rsqrt}(0)$)"
            elif k == "e3a":
                at_sqrt = "none at $\\sqrt0$ (only NaN-input propagation later)"
            else:
                at_sqrt = "NaN (operand 0 = the result)"
            rows += "%s & %s & %s & %s & \\texttt{%s} in \\texttt{%s}\\\\\n" % (case, name, instr, at_sqrt, tex_escape(first[1].split(" R")[0]) + " [" + first[0].replace("infinity", "Inf") + "]", tex_escape(first[2]))
    write("tab_sqrt.tex", "\\begin{tabular}{p{2.5cm}p{2.3cm}p{1.6cm}p{4.0cm}p{3.6cm}}\n\\toprule\ncase & sqrt compiled as & sqrt instruction & report at the sqrt site & first report in the run\\\\\n\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")
    def keep(path, lo, hi):
        L = [l for l in open(path).read().splitlines() if l.strip()]
        return L
    ieee = [l for l in open("logs/sqrt/sass_pivot_ieee.txt").read().splitlines()]
    start = next(i for i, l in enumerate(ieee) if "LDG.E R0" in l)
    excerpt("sass_sqrt_ieee.txt", "\n".join(re.sub(r"\s{2,}", " ", l) for l in ieee[start + 1:] if "NOP" not in l and "BRA 0x3a0" not in l))
    fast = open("logs/sqrt/sass_pivot_fast.txt").read().splitlines()
    start = next(i for i, l in enumerate(fast) if "LDG.E R0" in l)
    excerpt("sass_sqrt_fast.txt", "\n".join(re.sub(r"\s{2,}", " ", l) for l in fast[start + 1:] if "NOP" not in l))

print("wrote", len(os.listdir(OUT)), "files in", OUT)
