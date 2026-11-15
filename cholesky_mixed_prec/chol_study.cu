// chol_study.cu -- Cholesky factorization A = L L^T on the GPU, for the mixed-precision / nixnan tutorial.
// See EXPERIMENT_PLAN.md.  This file implements the E0/E1 part: FP64 and FP32, two code paths:
//
//   unblocked  one column per step: pivot (sqrt), scale column, rank-1 update        3 kernels, 3n launches
//   blocked    NB x NB diagonal tile (POTRF), panel TRSM, trailing SYRK update       3 kernels, 3n/NB launches
//
// Build:  nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 -DREAL=double|float -o chol chol_study.cu
// Run:    ./chol --csv matrices/M1_n256.csv --path unblocked|blocked [--nb 32]      (matrices from gen_chol_matrices.py)
//         ./chol --n 256 --log2delta -12 --path ...                                     (built-in A_delta, no file)
//
// Matrix: read from --csv (n inferred), or the built-in A_delta = 1 1^T + delta I. Held in double, rounded to REAL for the solve.
// b = A * ones (in double, rounded to REAL), so the exact solution is x = ones.
// The kernels do NOT guard the pivot: a non-positive pivot still goes through sqrt and division. The index of the first
// bad pivot is only *recorded* (atomicMin) so the host can print it; it does not change the arithmetic.
// Metrics are computed on the host in double against the original double matrix:
//   fact_err = ||A - L L^T||_F / ||A||_F      resid = ||b - A x||_2 / ||b||_2      fwd_err = ||x - 1||_2 / ||1||_2

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef REAL
#define REAL double
#endif
typedef REAL real;
#define IDX(i, j, n) ((size_t)(i) * (n) + (j))
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// ------------------------------------------------------------------ unblocked path
__global__ void pivot_kernel(real *A, int n, int k, int *bad)
{
    real s = A[IDX(k, k, n)];
    if (!(s > (real)0)) atomicMin(bad, k);   // record only; the sqrt below still runs
    A[IDX(k, k, n)] = sqrt(s);
}

__global__ void scale_kernel(real *A, int n, int k)
{
    int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[IDX(i, k, n)] = A[IDX(i, k, n)] / A[IDX(k, k, n)];
}

__global__ void rank1_kernel(real *A, int n, int k)   // A[i][j] -= L[i][k] * L[j][k], k < j <= i
{
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j <= i) A[IDX(i, j, n)] -= A[IDX(i, k, n)] * A[IDX(j, k, n)];
}

// ------------------------------------------------------------------ blocked path
#define MAXNB 64
__global__ void potrf_tile_kernel(real *A, int n, int kb, int nb, int *bad)   // one block, nb threads
{
    __shared__ real s[MAXNB][MAXNB + 1];
    int t = threadIdx.x;
    for (int c = 0; c < nb; ++c) s[t][c] = A[IDX(kb + t, kb + c, n)];
    __syncthreads();
    for (int j = 0; j < nb; ++j) {
        if (t == j) {
            real d = s[j][j];
            if (!(d > (real)0)) atomicMin(bad, kb + j);
            s[j][j] = sqrt(d);
        }
        __syncthreads();
        if (t > j) s[t][j] = s[t][j] / s[j][j];
        __syncthreads();
        if (t > j) for (int l = j + 1; l <= t; ++l) s[t][l] -= s[t][j] * s[l][j];
        __syncthreads();
    }
    for (int c = 0; c <= t; ++c) A[IDX(kb + t, kb + c, n)] = s[t][c];
}

__global__ void trsm_panel_kernel(real *A, int n, int kb, int nb)   // rows below the tile: X L^T = A_panel
{
    int i = kb + nb + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < nb; ++j) {
        real v = A[IDX(i, kb + j, n)];
        for (int l = 0; l < j; ++l) v -= A[IDX(i, kb + l, n)] * A[IDX(kb + j, kb + l, n)];
        A[IDX(i, kb + j, n)] = v / A[IDX(kb + j, kb + j, n)];
    }
}

__global__ void syrk_kernel(real *A, int n, int kb, int nb)   // trailing: A[i][j] -= sum_l L[i][l] L[j][l], lower part
{
    int j = kb + nb + blockIdx.x * blockDim.x + threadIdx.x;
    int i = kb + nb + blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j <= i) {
        real acc = 0;
        for (int l = 0; l < nb; ++l) acc += A[IDX(i, kb + l, n)] * A[IDX(j, kb + l, n)];
        A[IDX(i, j, n)] -= acc;
    }
}

// ------------------------------------------------------------------ solves (single thread, as in lu_solve_gpu)
__global__ void forward_kernel(const real *L, const real *b, real *y, int n)   // L y = b
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < n; ++i) {
        real s = b[i];
        for (int j = 0; j < i; ++j) s -= L[IDX(i, j, n)] * y[j];
        y[i] = s / L[IDX(i, i, n)];
    }
}

__global__ void backward_kernel(const real *L, const real *y, real *x, int n)  // L^T x = y
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = n - 1; i >= 0; --i) {
        real s = y[i];
        for (int j = i + 1; j < n; ++j) s -= L[IDX(j, i, n)] * x[j];
        x[i] = s / L[IDX(i, i, n)];
    }
}

// ------------------------------------------------------------------ host
int main(int argc, char **argv)
{
    int n = 256, nb = 32, log2d = -12;
    const char *path = "unblocked", *csv = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nb") && i + 1 < argc) nb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log2delta") && i + 1 < argc) log2d = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--path") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { fprintf(stderr, "usage: %s [--csv FILE | --n N --log2delta E] [--path unblocked|blocked] [--nb NB]\n", argv[0]); return 1; }
    }
    bool blocked = !strcmp(path, "blocked");
    if (blocked && (nb > MAXNB || n % nb)) { fprintf(stderr, "blocked path needs nb <= %d and n %% nb == 0\n", MAXNB); return 1; }
    const double delta = ldexp(1.0, log2d);
    const char *cfg = sizeof(real) == 4 ? "F32" : "F64";

    // matrix (double), b = A*1 (double)
    std::vector<double> A, b;
    const char *mname = "A_delta";
    if (csv) {
        FILE *f = fopen(csv, "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", csv); return 1; }
        std::vector<char> line(1 << 20);
        int rows = 0;
        while (fgets(line.data(), (int)line.size(), f)) {
            char *p = line.data(), *e;
            int cols = 0;
            while (*p && *p != '\n') { A.push_back(strtod(p, &e)); if (e == p) break; p = e; if (*p == ',') ++p; ++cols; }
            if (rows == 0) n = cols;
            if (cols != n) { fprintf(stderr, "%s: row %d has %d entries, expected %d\n", csv, rows, cols, n); return 1; }
            ++rows;
        }
        fclose(f);
        if (rows != n) { fprintf(stderr, "%s: %d rows but %d columns\n", csv, rows, n); return 1; }
        mname = strrchr(csv, '/') ? strrchr(csv, '/') + 1 : csv;
        if (blocked && n % nb) { fprintf(stderr, "blocked path needs n %% nb == 0\n"); return 1; }
    } else {
        A.assign((size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) A[IDX(i, j, n)] = 1.0 + (i == j ? delta : 0.0);
    }
    b.assign(n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) b[i] += A[IDX(i, j, n)];

    std::vector<real> hA(A.begin(), A.end()), hb(b.begin(), b.end());
    size_t mb = (size_t)n * n * sizeof(real), vb = (size_t)n * sizeof(real);
    real *dA, *db, *dy, *dx; int *dbad;
    CK(cudaMalloc(&dA, mb)); CK(cudaMalloc(&db, vb)); CK(cudaMalloc(&dy, vb)); CK(cudaMalloc(&dx, vb));
    CK(cudaMalloc(&dbad, sizeof(int)));
    CK(cudaMemcpy(dA, hA.data(), mb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), vb, cudaMemcpyHostToDevice));
    int big = INT_MAX; CK(cudaMemcpy(dbad, &big, sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    if (!blocked) {
        for (int k = 0; k < n; ++k) {
            pivot_kernel<<<1, 1>>>(dA, n, k, dbad);
            if (k + 1 < n) {
                int m = n - k - 1;
                scale_kernel<<<(m + 127) / 128, 128>>>(dA, n, k);
                dim3 b2(16, 16), g2((m + 15) / 16, (m + 15) / 16);
                rank1_kernel<<<g2, b2>>>(dA, n, k);
            }
        }
    } else {
        for (int kb = 0; kb < n; kb += nb) {
            potrf_tile_kernel<<<1, nb>>>(dA, n, kb, nb, dbad);
            int m = n - kb - nb;
            if (m > 0) {
                trsm_panel_kernel<<<(m + 127) / 128, 128>>>(dA, n, kb, nb);
                dim3 b2(16, 16), g2((m + 15) / 16, (m + 15) / 16);
                syrk_kernel<<<g2, b2>>>(dA, n, kb, nb);
            }
        }
    }
    forward_kernel<<<1, 1>>>(dA, db, dy, n);
    backward_kernel<<<1, 1>>>(dA, dy, dx, n);
    cudaEventRecord(e1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);

    std::vector<real> L((size_t)n * n), x(n);
    int bad = 0;
    CK(cudaMemcpy(L.data(), dA, mb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x.data(), dx, vb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));

    // metrics in double
    bool finite = true;
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) finite &= std::isfinite((double)L[IDX(i, j, n)]);
    for (int i = 0; i < n; ++i) finite &= std::isfinite((double)x[i]);
    double fe = NAN, rs = NAN, fw = NAN;
    if (finite) {
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int k = 0; k <= (i < j ? i : j); ++k) s += (double)L[IDX(i, k, n)] * (double)L[IDX(j, k, n)];
                double d = A[IDX(i, j, n)] - s; num += d * d; den += A[IDX(i, j, n)] * A[IDX(i, j, n)];
            }
        fe = sqrt(num / den);
        double r2 = 0, b2 = 0, f2 = 0;
        for (int i = 0; i < n; ++i) {
            double ax = 0; for (int j = 0; j < n; ++j) ax += A[IDX(i, j, n)] * (double)x[j];
            r2 += (b[i] - ax) * (b[i] - ax); b2 += b[i] * b[i]; f2 += ((double)x[i] - 1.0) * ((double)x[i] - 1.0);
        }
        rs = sqrt(r2 / b2); fw = sqrt(f2 / n);
    }
    const double eps = sizeof(real) == 4 ? FLT_EPSILON : DBL_EPSILON;
    const char *status = bad != INT_MAX ? "breakdown" : !finite ? "nan" : fe > 1e3 * eps ? "inaccurate" : "ok";
    printf("RESULT matrix=%s config=%s path=%s n=%d status=%s fact_err=%.3e resid=%.3e fwd_err=%.3e "
           "first_bad_pivot=%d t_ms=%.3f\n", mname, cfg, path, n, status, fe, rs, fw,
           bad == INT_MAX ? -1 : bad, ms);
    return 0;
}
