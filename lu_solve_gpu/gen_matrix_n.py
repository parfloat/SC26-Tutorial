#!/usr/bin/env python3
"""Generate n x n test matrices for lu_solve_gpu*: a well-conditioned one and a nearly singular one.

  python3 gen_matrix_n.py [n=128] [seed=1]   ->  matrix_<n>.csv, bad_matrix_<n>.csv

good: tridiag(-1, 3, -1)                              (cond < 5)
bad : B @ C with B (n x n-1), C (n-1 x n) each a tridiagonal(-1,2,-1)-like band whose entries are
      perturbed by ~1e-6 relative.  Rank <= n-1 in exact arithmetic, so the stored double matrix has
      sigma_min ~ 1e-16 * sigma_max, the same regime as bad_matrix.csv (cond ~ 2e17).
The condition number is computed in 80-digit arithmetic (mpmath) on the stored (rounded) matrix,
because double-precision SVD cannot resolve sigma_min below eps*sigma_max.
"""
import sys, numpy as np
n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rng = np.random.default_rng(seed)

good = 3.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

def band(rows, cols):
    M = np.zeros((rows, cols))
    for i in range(rows):
        for j in (i - 1, i, i + 1):
            if 0 <= j < cols:
                M[i, j] = (2.0 if i == j else -1.0) * (1.0 + 1e-6 * rng.standard_normal())
    return M
bad = band(n, n - 1) @ band(n - 1, n)

for name, A in (("matrix_%d.csv" % n, good), ("bad_matrix_%d.csv" % n, bad)):
    np.savetxt(name, A, delimiter=",", fmt="%.17g")
    print("wrote", name)

import mpmath as mp
mp.mp.dps = 80
def cond_est(A):
    """sigma_max by power iteration in double; sigma_min by inverse iteration of A^T A in 80 digits."""
    smax = np.linalg.norm(A, 2)
    M = mp.matrix(A.tolist())
    MT = M.T
    v = mp.matrix([1 + 0.01 * i for i in range(A.shape[0])])
    for _ in range(6):
        w = mp.lu_solve(M, mp.lu_solve(MT, v))        # (A^T A)^-1 v  = A^-1 A^-T v
        v = w / mp.norm(w)
    smin = 1 / mp.sqrt(mp.norm(mp.lu_solve(M, mp.lu_solve(MT, v))))
    return smax, smin
for name, A in (("good", good), ("bad", bad)):
    smax, smin = cond_est(A)
    print("%s: sigma_max=%.4g  sigma_min=%s  cond=%s" % (name, smax, mp.nstr(smin, 4), mp.nstr(smax / smin, 4)))
