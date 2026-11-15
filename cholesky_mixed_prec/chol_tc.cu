// chol_tc.cu -- E4: blocked Cholesky with a TENSOR-CORE trailing update (mixed precision).
//
//   A and L are held in FP32. Per block step k (NB = 32):
//     potrf_tile_kernel   diagonal tile, FP32 (as in chol_study.cu)
//     trsm_panel_kernel   panel L_Rk = A_Rk L_kk^-T, FP32
//     [convert_panel]     (FP16/BF16 only) round the panel L_Rk to the narrow format
//     tc_update_kernel    A_RR <- A_RR - L_Rk L_Rk^T on tensor cores (WMMA): narrow-format inputs, FP32 accumulate
//   then FP32 forward/backward substitution with the FP32 factor.
//   FMT = f16  : wmma 16x16x16, __half inputs            (HMMA.16816.F32)
//         bf16 : wmma 16x16x16, __nv_bfloat16 inputs     (HMMA.16816.F32.BF16)
//         tf32 : wmma 16x16x8,  tf32 inputs (float converted with __float_to_tf32, round to nearest, ties away)  (HMMA.1684.F32.TF32)
// The pivot is not guarded: a pivot <= 0 is only recorded (atomicMin of its index) and the arithmetic continues.
//
// Build:  nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -o chol_tc chol_tc.cu
// Run:    ./chol_tc --fmt f16|bf16|tf32 --log2delta -12 | --sweep LO HI [--fmts f16,bf16,tf32] [--n 256]
// Matrix: A_delta = 1 1^T + delta I, delta = 2^log2delta, b = A*1 (exact solution x = 1); metrics as in chol_study.cu.

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "csv_matrix.h"
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;

#define IDX(i, j, n) ((size_t)(i) * (n) + (j))
#define NB 32
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void potrf_tile_kernel(float *A, int n, int kb, int *bad)   // one block, NB threads
{
    __shared__ float s[NB][NB + 1];
    int t = threadIdx.x;
    for (int c = 0; c < NB; ++c) s[t][c] = A[IDX(kb + t, kb + c, n)];
    __syncthreads();
    for (int j = 0; j < NB; ++j) {
        if (t == j) {
            float d = s[j][j];
            if (!(d > 0.0f)) atomicMin(bad, kb + j);
            s[j][j] = sqrtf(d);
        }
        __syncthreads();
        if (t > j) s[t][j] = s[t][j] / s[j][j];
        __syncthreads();
        if (t > j) for (int l = j + 1; l <= t; ++l) s[t][l] -= s[t][j] * s[l][j];
        __syncthreads();
    }
    for (int c = 0; c <= t; ++c) A[IDX(kb + t, kb + c, n)] = s[t][c];
}

__global__ void trsm_panel_kernel(float *A, int n, int kb)
{
    int i = kb + NB + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < NB; ++j) {
        float v = A[IDX(i, kb + j, n)];
        for (int l = 0; l < j; ++l) v -= A[IDX(i, kb + l, n)] * A[IDX(kb + j, kb + l, n)];
        A[IDX(i, kb + j, n)] = v / A[IDX(kb + j, kb + j, n)];
    }
}

template <typename T> __device__ T to_T(float f);
template <> __device__ __half to_T<__half>(float f) { return __float2half_rn(f); }
template <> __device__ __nv_bfloat16 to_T<__nv_bfloat16>(float f) { return __float2bfloat16_rn(f); }

template <typename T>
__global__ void convert_panel_kernel(const float *A, T *Lh, int n, int kb)   // Lh[i][c] = round(A[i][kb+c]), rows i >= kb+NB
{
    int i = kb + NB + blockIdx.x;
    int c = threadIdx.x;
    if (i < n && c < NB) Lh[(size_t)i * NB + c] = to_T<T>(A[IDX(i, kb + c, n)]);
}

// trailing update, one warp per 16x16 tile (lower tiles only): A_tile -= L_i L_j^T
template <typename T>
__global__ void tc_update_kernel(float *A, const T *Lh, int n, int kb)
{
    int ti = blockIdx.y, tj = blockIdx.x;
    if (tj > ti) return;
    int i0 = kb + NB + ti * 16, j0 = kb + NB + tj * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int kk = 0; kk < NB; kk += 16) {
        wmma::load_matrix_sync(a, Lh + (size_t)i0 * NB + kk, NB);
        wmma::load_matrix_sync(b, Lh + (size_t)j0 * NB + kk, NB);   // B(k, j) = L[j0+j][kk+k]
        wmma::mma_sync(acc, a, b, acc);
    }
    __shared__ float s[16][16];
    wmma::store_matrix_sync(&s[0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = threadIdx.x; idx < 256; idx += 32) A[IDX(i0 + idx / 16, j0 + idx % 16, n)] -= s[idx / 16][idx % 16];
}

__global__ void tc_update_tf32_kernel(float *A, int n, int kb)   // inputs read straight from the FP32 panel
{
    int ti = blockIdx.y, tj = blockIdx.x;
    if (tj > ti) return;
    int i0 = kb + NB + ti * 16, j0 = kb + NB + tj * 16;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int kk = 0; kk < NB; kk += 8) {
        wmma::load_matrix_sync(a, A + IDX(i0, kb + kk, n), n);
        wmma::load_matrix_sync(b, A + IDX(j0, kb + kk, n), n);
        for (int t = 0; t < a.num_elements; ++t) a.x[t] = wmma::__float_to_tf32(a.x[t]);
        for (int t = 0; t < b.num_elements; ++t) b.x[t] = wmma::__float_to_tf32(b.x[t]);
        wmma::mma_sync(acc, a, b, acc);
    }
    __shared__ float s[16][16];
    wmma::store_matrix_sync(&s[0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int idx = threadIdx.x; idx < 256; idx += 32) A[IDX(i0 + idx / 16, j0 + idx % 16, n)] -= s[idx / 16][idx % 16];
}

__global__ void forward_kernel(const float *L, const float *b, float *y, int n)
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = 0; i < n; ++i) {
        float s = b[i];
        for (int j = 0; j < i; ++j) s -= L[IDX(i, j, n)] * y[j];
        y[i] = s / L[IDX(i, i, n)];
    }
}

__global__ void backward_kernel(const float *L, const float *y, float *x, int n)
{
    if (blockIdx.x || threadIdx.x) return;
    for (int i = n - 1; i >= 0; --i) {
        float s = y[i];
        for (int j = i + 1; j < n; ++j) s -= L[IDX(j, i, n)] * x[j];
        x[i] = s / L[IDX(i, i, n)];
    }
}

static std::vector<double> g_csv;            // --csv: matrix from a file instead of A_delta
static const char *g_name = "A_delta";

static void run_case(int n, int e, const char *fmt)
{
    const double delta = ldexp(1.0, -e);
    std::vector<double> A0((size_t)n * n), b(n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A0[IDX(i, j, n)] = g_csv.empty() ? 1.0 + (i == j ? delta : 0.0) : g_csv[IDX(i, j, n)];
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) b[i] += A0[IDX(i, j, n)];
    std::vector<float> hA(A0.begin(), A0.end()), hb(b.begin(), b.end());
    float *dA, *db, *dy, *dx; int *dbad; void *dLh = nullptr;
    size_t mb = (size_t)n * n * sizeof(float), vb = n * sizeof(float);
    CK(cudaMalloc(&dA, mb)); CK(cudaMalloc(&db, vb)); CK(cudaMalloc(&dy, vb)); CK(cudaMalloc(&dx, vb)); CK(cudaMalloc(&dbad, sizeof(int)));
    CK(cudaMalloc(&dLh, (size_t)n * NB * 2));
    CK(cudaMemcpy(dA, hA.data(), mb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), vb, cudaMemcpyHostToDevice));
    int big = INT_MAX; CK(cudaMemcpy(dbad, &big, sizeof(int), cudaMemcpyHostToDevice));
    const bool f16 = !strcmp(fmt, "f16"), bf16 = !strcmp(fmt, "bf16");
    for (int kb = 0; kb < n; kb += NB) {
        potrf_tile_kernel<<<1, NB>>>(dA, n, kb, dbad);
        int m = n - kb - NB;
        if (m > 0) {
            trsm_panel_kernel<<<(m + 127) / 128, 128>>>(dA, n, kb);
            dim3 g((m + 15) / 16, (m + 15) / 16);
            if (f16) {
                convert_panel_kernel<__half><<<m, NB>>>(dA, (__half *)dLh, n, kb);
                tc_update_kernel<__half><<<g, 32>>>(dA, (const __half *)dLh, n, kb);
            } else if (bf16) {
                convert_panel_kernel<__nv_bfloat16><<<m, NB>>>(dA, (__nv_bfloat16 *)dLh, n, kb);
                tc_update_kernel<__nv_bfloat16><<<g, 32>>>(dA, (const __nv_bfloat16 *)dLh, n, kb);
            } else {
                tc_update_tf32_kernel<<<g, 32>>>(dA, n, kb);
            }
        }
    }
    forward_kernel<<<1, 1>>>(dA, db, dy, n);
    backward_kernel<<<1, 1>>>(dA, dy, dx, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<float> L((size_t)n * n), x(n);
    int bad = 0;
    CK(cudaMemcpy(L.data(), dA, mb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(x.data(), dx, vb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(db); cudaFree(dy); cudaFree(dx); cudaFree(dbad); cudaFree(dLh);

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
                double d = A0[IDX(i, j, n)] - s; num += d * d; den += A0[IDX(i, j, n)] * A0[IDX(i, j, n)];
            }
        fe = sqrt(num / den);
        double r2 = 0, b2 = 0, f2 = 0;
        for (int i = 0; i < n; ++i) {
            double ax = 0; for (int j = 0; j < n; ++j) ax += A0[IDX(i, j, n)] * (double)x[j];
            r2 += (b[i] - ax) * (b[i] - ax); b2 += b[i] * b[i]; f2 += ((double)x[i] - 1.0) * ((double)x[i] - 1.0);
        }
        rs = sqrt(r2 / b2); fw = sqrt(f2 / n);
    }
    const char *status = bad != INT_MAX ? "breakdown" : !finite ? "nan" : (fe > 1e3 * FLT_EPSILON || fw > 0.1) ? "inaccurate" : "ok";
    printf("RESULT matrix=%s config=F32TC store=f32 path=blocked-tc-%s n=%d log2delta=%d kappa=%.3e status=%s cause=%s stored_changed=0 "
           "fact_err=%.3e fact_err_stored=%.3e resid=%.3e fwd_err=%.3e first_bad_pivot=%d t_ms=0\n",
           g_name, fmt, n, g_csv.empty() ? -e : 0, g_csv.empty() ? (n + delta) / delta : NAN, status, !strcmp(status, "ok") ? "-" : "arithmetic", fe, fe, rs, fw, bad == INT_MAX ? -1 : bad);
}

static std::vector<std::string> split(const char *s)
{
    std::vector<std::string> out; std::string cur;
    for (; *s; ++s) { if (*s == ',') { out.push_back(cur); cur.clear(); } else cur += *s; }
    out.push_back(cur);
    return out;
}

int main(int argc, char **argv)
{
    int n = 256, e = 12, lo = 0, hi = -1;
    const char *fmt = "f16", *fmts = nullptr, *csvp = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log2delta") && i + 1 < argc) e = -atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csvp = argv[++i];
        else if (!strcmp(argv[i], "--fmt") && i + 1 < argc) fmt = argv[++i];
        else if (!strcmp(argv[i], "--fmts") && i + 1 < argc) fmts = argv[++i];
        else if (!strcmp(argv[i], "--sweep") && i + 2 < argc) { lo = atoi(argv[++i]); hi = atoi(argv[++i]); }
        else { fprintf(stderr, "usage: %s [--n N] [--csv FILE] [--fmt f16|bf16|tf32] [--log2delta E | --sweep LO HI [--fmts a,b]]\n", argv[0]); return 1; }
    }
    if (csvp) {
        if (!read_csv_matrix(csvp, g_csv, n)) return 1;
        g_name = strrchr(csvp, '/') ? strrchr(csvp, '/') + 1 : csvp;
    }
    if (n % NB) { fprintf(stderr, "n must be a multiple of %d\n", NB); return 1; }
    if (hi >= lo) { for (int k = lo; k <= hi; ++k) for (auto &f : split(fmts ? fmts : fmt)) run_case(n, k, f.c_str()); }
    else run_case(n, e, fmt);
    return 0;
}
