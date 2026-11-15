// chol_half.cu -- E3b: unblocked Cholesky whose ARITHMETIC is FP16 or BF16 (every operation rounds to the narrow format).
// Same algorithm and kernel structure as the unblocked path of chol_study.cu (pivot / scale / rank-1 update / two substitutions),
// but written with the cuda_fp16 / cuda_bf16 intrinsics: the update is one fused __hfma (HFMA2), the scale is __hdiv, the pivot hsqrt.
// No pivot guard: the first pivot <= 0 is only recorded (atomicMin) and the arithmetic continues.
//
// Build:  nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 [-DUSE_BF16] -o chol_half chol_half.cu
// Run:    ./chol_half --log2delta -12 | --sweep LO HI  [--n 256]       (A_delta = 1 1^T + delta I, delta = 2^log2delta)
// Output: RESULT lines with the same fields as chol_study.cu (fact_err/resid/fwd_err are computed on the host in double against the
// exact A_delta; fact_err_stored against the matrix as rounded to the narrow format).

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef USE_BF16
typedef __nv_bfloat16 T;
#define TO_T(f) __float2bfloat16_rn(f)
#define FROM_T(h) __bfloat162float(h)
static const char *CFG = "BF16A";
static const double EPS_T = 0.0078125;       // 2^-7
#else
typedef __half T;
#define TO_T(f) __float2half_rn(f)
#define FROM_T(h) __half2float(h)
static const char *CFG = "F16A";
static const double EPS_T = 0.0009765625;    // 2^-10
#endif
#define IDX(i, j, n) ((size_t)(i) * (n) + (j))
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void pivot_kernel(T *A, int n, int k, int *bad)
{
    T s = A[IDX(k, k, n)];
    if (!__hgt(s, TO_T(0.0f))) atomicMin(bad, k);
    A[IDX(k, k, n)] = hsqrt(s);
}

__global__ void scale_kernel(T *A, int n, int k)
{
    int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[IDX(i, k, n)] = __hdiv(A[IDX(i, k, n)], A[IDX(k, k, n)]);
}

__global__ void rank1_kernel(T *A, int n, int k)       // A[i][j] -= L[i][k] * L[j][k]  (one fused half multiply-add)
{
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j <= i) A[IDX(i, j, n)] = __hfma(__hneg(A[IDX(i, k, n)]), A[IDX(j, k, n)], A[IDX(i, j, n)]);
}

__global__ void forward_kernel(const T *L, const T *b, T *y, int n)
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < n; ++i) {
        T s = b[i];
        for (int j = 0; j < i; ++j) s = __hfma(__hneg(L[IDX(i, j, n)]), y[j], s);
        y[i] = __hdiv(s, L[IDX(i, i, n)]);
    }
}

__global__ void backward_kernel(const T *L, const T *y, T *x, int n)
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = n - 1; i >= 0; --i) {
        T s = y[i];
        for (int j = i + 1; j < n; ++j) s = __hfma(__hneg(L[IDX(j, i, n)]), x[j], s);
        x[i] = __hdiv(s, L[IDX(i, i, n)]);
    }
}

static void run_case(int n, int e)
{
    const double delta = ldexp(1.0, -e);
    std::vector<double> A0((size_t)n * n), As((size_t)n * n), b(n, 0.0);
    std::vector<T> hA((size_t)n * n), hb(n);
    double chg = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A0[IDX(i, j, n)] = 1.0 + (i == j ? delta : 0.0);
            hA[IDX(i, j, n)] = TO_T((float)A0[IDX(i, j, n)]);
            As[IDX(i, j, n)] = (double)FROM_T(hA[IDX(i, j, n)]);
            chg = fmax(chg, fabs(As[IDX(i, j, n)] - A0[IDX(i, j, n)]));
        }
    for (int i = 0; i < n; ++i) { for (int j = 0; j < n; ++j) b[i] += A0[IDX(i, j, n)]; hb[i] = TO_T((float)b[i]); }
    T *dA, *db, *dy, *dx; int *dbad;
    size_t mb = (size_t)n * n * sizeof(T), vb = n * sizeof(T);
    CK(cudaMalloc(&dA, mb)); CK(cudaMalloc(&db, vb)); CK(cudaMalloc(&dy, vb)); CK(cudaMalloc(&dx, vb)); CK(cudaMalloc(&dbad, sizeof(int)));
    CK(cudaMemcpy(dA, hA.data(), mb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), vb, cudaMemcpyHostToDevice));
    int big = INT_MAX; CK(cudaMemcpy(dbad, &big, sizeof(int), cudaMemcpyHostToDevice));
    for (int k = 0; k < n; ++k) {
        pivot_kernel<<<1, 1>>>(dA, n, k, dbad);
        if (k + 1 < n) {
            int m = n - k - 1;
            scale_kernel<<<(m + 127) / 128, 128>>>(dA, n, k);
            dim3 b2(16, 16), g2((m + 15) / 16, (m + 15) / 16);
            rank1_kernel<<<g2, b2>>>(dA, n, k);
        }
    }
    forward_kernel<<<1, 1>>>(dA, db, dy, n);
    backward_kernel<<<1, 1>>>(dA, dy, dx, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<T> L((size_t)n * n), x(n);
    int bad = 0;
    CK(cudaMemcpy(L.data(), dA, mb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x.data(), dx, vb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(db); cudaFree(dy); cudaFree(dx); cudaFree(dbad);

    bool finite = true;
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) finite &= std::isfinite((double)FROM_T(L[IDX(i, j, n)]));
    for (int i = 0; i < n; ++i) finite &= std::isfinite((double)FROM_T(x[i]));
    double fe = NAN, fs = NAN, rs = NAN, fw = NAN;
    if (finite) {
        double num0 = 0, numS = 0, den0 = 0, denS = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int k = 0; k <= (i < j ? i : j); ++k) s += (double)FROM_T(L[IDX(i, k, n)]) * (double)FROM_T(L[IDX(j, k, n)]);
                double d0 = A0[IDX(i, j, n)] - s, dS = As[IDX(i, j, n)] - s;
                num0 += d0 * d0; den0 += A0[IDX(i, j, n)] * A0[IDX(i, j, n)];
                numS += dS * dS; denS += As[IDX(i, j, n)] * As[IDX(i, j, n)];
            }
        fe = sqrt(num0 / den0); fs = sqrt(numS / denS);
        double r2 = 0, b2 = 0, f2 = 0;
        for (int i = 0; i < n; ++i) {
            double ax = 0; for (int j = 0; j < n; ++j) ax += A0[IDX(i, j, n)] * (double)FROM_T(x[j]);
            r2 += (b[i] - ax) * (b[i] - ax); b2 += b[i] * b[i]; f2 += ((double)FROM_T(x[i]) - 1.0) * ((double)FROM_T(x[i]) - 1.0);
        }
        rs = sqrt(r2 / b2); fw = sqrt(f2 / n);
    }
    const char *status = bad != INT_MAX ? "breakdown" : !finite ? "nan" : (fe > 1e3 * EPS_T || fw > 0.1) ? "inaccurate" : "ok";
    const char *cause = !strcmp(status, "ok") ? "-" : chg > 0 ? "storage" : "arithmetic";
    printf("RESULT matrix=A_delta config=%s store=%s path=unblocked n=%d log2delta=%d kappa=%.3e status=%s cause=%s stored_changed=%d "
           "fact_err=%.3e fact_err_stored=%.3e resid=%.3e fwd_err=%.3e first_bad_pivot=%d t_ms=0\n",
           CFG, CFG[0] == 'B' ? "bf16" : "f16", n, -e, (n + delta) / delta, status, cause, chg > 0, fe, fs, rs, fw, bad == INT_MAX ? -1 : bad);
}

int main(int argc, char **argv)
{
    int n = 256, e = 12, lo = 0, hi = -1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log2delta") && i + 1 < argc) e = -atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep") && i + 2 < argc) { lo = atoi(argv[++i]); hi = atoi(argv[++i]); }
        else { fprintf(stderr, "usage: %s [--n N] [--log2delta E | --sweep LO HI]\n", argv[0]); return 1; }
    }
    if (hi >= lo) for (int k = lo; k <= hi; ++k) run_case(n, k);
    else run_case(n, e);
    return 0;
}
