#!/usr/bin/env python3
"""E2, CPU cross-check: round A_delta (n=256) to each storage format, then factor with LAPACK (FP32 and FP64).

Usage: python3 e2_emulate.py [lo hi]      writes logs/e2/emulate.tsv, prints the table.
LAPACK reports 'not positive definite' (LinAlgError) where the GPU kernels see a pivot <= 0; the operation order differs, so a
boundary can move by one step. Rounding: f16 = numpy float16 (nearest-even); bf16 = nearest-even on the float32 bits;
tf32 = nearest, ties away from zero (cvt.rna), as in chol_study.cu.
"""
import os, sys, numpy as np, scipy.linalg as sl

def rnd(fmt, a):
    if fmt in ("f64",): return a
    f = a.astype(np.float32)
    if fmt == "f32": return f.astype(np.float64)
    if fmt == "f16": return f.astype(np.float16).astype(np.float64)
    u = f.view(np.uint32).astype(np.uint64)
    if fmt == "bf16": u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000)
    elif fmt == "tf32": u = ((u + 0x1000) & 0xFFFFE000)
    else: raise SystemExit("unknown " + fmt)
    return u.astype(np.uint32).view(np.float32).astype(np.float64)

def pd(A, dtype):
    try:
        sl.cholesky(A.astype(dtype), lower=True); return "pd"
    except np.linalg.LinAlgError:
        return "NOT-PD"

if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (2, 26)
    n = 256
    os.makedirs("logs/e2", exist_ok=True)
    rows = []
    print("%3s %10s | %-6s %-6s %-6s %-6s   (LAPACK float32 on the rounded matrix)" % ("e", "kappa", "f32", "tf32", "f16", "bf16"))
    for e in range(lo, hi + 1):
        d = 2.0 ** -e
        A = np.ones((n, n)) + d * np.eye(n)
        res = {f: pd(rnd(f, A), np.float32) for f in ("f32", "tf32", "f16", "bf16")}
        for f, r in res.items():
            rows.append((e, f, r))
        print("%3d %10.3e | %-6s %-6s %-6s %-6s" % (e, (n + d) / d, res["f32"], res["tf32"], res["f16"], res["bf16"]))
    with open("logs/e2/emulate.tsv", "w") as fh:
        fh.write("e\tstore\tlapack_float32\n")
        for r in rows:
            fh.write("%d\t%s\t%s\n" % r)
