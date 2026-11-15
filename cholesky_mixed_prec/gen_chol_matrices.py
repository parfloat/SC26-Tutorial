#!/usr/bin/env python3
"""Generate the SPD test matrices for the Cholesky study (EXPERIMENT_PLAN.md, Section 4).

  python3 gen_chol_matrices.py                    # M0 and M1 for n = 256 and n = 64  ->  matrices/
  python3 gen_chol_matrices.py --n 128            # other sizes
  python3 gen_chol_matrices.py --qlq 1e4 --seed 0 --out /tmp/Q.csv     # one random Q diag(lambda) Q^T matrix (E5)

Default matrices, same size and structure, different condition number:
  M0  A = 1 1^T + delta I,  delta = 2^-6    kappa_2 = (n+delta)/delta  ~ 1.6e4   (benign: FP16/BF16 storage keeps delta)
  M1  A = 1 1^T + delta I,  delta = 2^-12   kappa_2 ~ 1.05e6                      (ill-conditioned: FP16/TF32/BF16 storage rounds 1+delta to 1)
Both are SPD for every delta > 0: the eigenvalues are n+delta (once) and delta (n-1 times).

Files:  matrices/<name>_n<N>.csv  (full matrix, %.17g, row major)   matrices/matrices.tsv  (definition and kappa)
kappa is given analytically and, as a check, from the eigenvalues of the stored double matrix (numpy).
"""
import argparse, os, numpy as np

DEFS = {"M0": -6, "M1": -12}      # name -> log2(delta)

def a_delta(n, log2d):
    return np.ones((n, n)) + 2.0 ** log2d * np.eye(n)

def qlq(n, kappa, seed):
    """Q diag(sigma) Q^T with geometric spectrum sigma_j = kappa^(-j/(n-1)) (same recipe as ../mpir_gpu/Code2.cu)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sig = kappa ** (-np.arange(n) / (n - 1.0))
    A = (Q * sig) @ Q.T
    return (A + A.T) / 2

def kappa_numeric(A):
    w = np.linalg.eigvalsh(A)
    return w[-1] / w[0], w[0]

def save(path, A):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savetxt(path, A, delimiter=",", fmt="%.17g")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, action="append", help="size (repeatable; default 256 and 64)")
    ap.add_argument("--outdir", default="matrices")
    ap.add_argument("--qlq", type=float, metavar="KAPPA", help="write one random Q diag Q^T matrix instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output file for --qlq")
    a = ap.parse_args()
    if a.qlq:
        n = (a.n or [256])[0]
        A = qlq(n, a.qlq, a.seed)
        k, lo = kappa_numeric(A)
        save(a.out or os.path.join(a.outdir, "Q_n%d_k%g_s%d.csv" % (n, a.qlq, a.seed)), A)
        print("Q diag Q^T n=%d seed=%d: requested kappa %.3g, numeric %.3g, lambda_min %.3g (SPD: %s)" %
              (n, a.seed, a.qlq, k, lo, lo > 0))
        raise SystemExit
    rows = []
    for n in (a.n or [256, 64]):
        for name, e in DEFS.items():
            A = a_delta(n, e)
            k_num, lo = kappa_numeric(A)
            d = 2.0 ** e
            save(os.path.join(a.outdir, "%s_n%d.csv" % (name, n)), A)
            rows.append((name, n, "1 1^T + delta I", "2^%d" % e, "%.4e" % ((n + d) / d), "%.4e" % k_num, "%.3e" % lo))
            print("%s n=%d delta=2^%d  kappa analytic %.4e  numeric %.4e  lambda_min %.3e" %
                  (name, n, e, (n + d) / d, k_num, lo))
    with open(os.path.join(a.outdir, "matrices.tsv"), "w") as f:
        f.write("name\tn\tdefinition\tdelta\tkappa_analytic\tkappa_numeric\tlambda_min\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
