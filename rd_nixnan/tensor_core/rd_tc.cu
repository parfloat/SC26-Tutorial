// rd_tc.cu
// -----------------------------------------------------------------------------
// rd_nixnan on TENSOR CORES: the same reaction-diffusion FTCS scheme, in 2D,
// with every time step done as two matrix multiplies on the tensor cores.
//
//     du/dt = D (u_xx + u_yy) + lambda u   on the unit square, u = 0 on the edge
//     u(x,y,0) = sin(pi x) sin(pi y)
//
// Store the N x N interior values as a matrix U.  The 1D second difference is
// the tridiagonal matrix T = tridiag(1, -2, 1) (the zero boundary is implicit:
// T simply has no entries outside the interior), so
//
//     u_xx  ~  T U / dx^2        (differences down the columns)
//     u_yy  ~  U T / dx^2        (differences along the rows)
//
// and one FTCS step  U' = U + r (T U + U T) + (lambda dt) U  is exactly
//
//     U' = P U + U P,      P = ((1 + lambda dt) / 2) I + r T,   r = D dt / dx^2
//
// Two GEMMs, accumulated into one 16x16 accumulator tile per warp with WMMA,
// which compiles to HMMA.16816 on sm_86.  Three tensor-core modes:
//
//   fp16acc16 : FP16 inputs, FP16 accumulator  (HMMA.16816.F16)
//               the accumulator itself overflows: inf is born INSIDE the HMMA.
//   fp16acc32 : FP16 inputs, FP32 accumulator  (HMMA.16816.F32)
//               the accumulator is fine; inf is born when the FP32 result is
//               rounded to FP16 for storage (F2F), and enters the next HMMA.
//   bf16acc32 : BF16 inputs, FP32 accumulator  (HMMA.16816.F32.BF16)
//               BF16 has FP32's range, so it lasts ~8x longer.
//
// The tensor-core twist: in the stencil code one inf touches 3 neighbours per
// step.  Here P is a DENSE operand that is mostly zeros, and the tensor core
// computes every 0 * inf product: 0 * inf = NaN, so a single inf in U poisons
// its whole row and column of U' in one step.
//
// Build:  nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 rd_tc.cu -o rd_tc
// Run  :  ./rd_tc [fp16acc16|fp16acc32|bf16acc32|all]
//         LD_PRELOAD=/path/nixnan.so ./rd_tc fp16acc16
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace nvcuda;

#ifndef N
#define N 64           // interior points per side (multiple of 16); dx = 1/(N+1)
#endif
#ifndef M
#define M 2000         // time steps
#endif
#define TILES (N / 16)

static const double D      = 0.01;
static const double LAMBDA = 50.0;
static const double DT     = 1e-3;
static const double DX     = 1.0 / (N + 1);
static const double R      = D * DT / (DX * DX);   // 0.04225 for N = 64 (stable: <= 0.25 in 2D)
static const double LDT    = LAMBDA * DT;          // 0.05

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){                \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__);\
    exit(1);} } while(0)

// =============================================================================
// One step U' = P U + U P.  One warp per 16x16 output tile (i, j):
//     acc  = sum_k P[i,k] U[k,j]      (TILES HMMA pairs)
//     acc += sum_k U[i,k] P[k,j]      (TILES HMMA pairs)
// =============================================================================

// FP16 in, FP16 accumulate: the overflow happens in the tensor core itself.
__global__ void rd_tc_fp16acc16(const __half* P, const __half* U, __half* Un)
{
    int ti = blockIdx.y, tj = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc;
    wmma::fill_fragment(acc, __float2half(0.f));

    for (int k = 0; k < TILES; ++k) {                       // P U
        wmma::load_matrix_sync(a, P + (ti * 16) * N + k * 16, N);
        wmma::load_matrix_sync(b, U + (k * 16) * N + tj * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    for (int k = 0; k < TILES; ++k) {                       // U P
        wmma::load_matrix_sync(a, U + (ti * 16) * N + k * 16, N);
        wmma::load_matrix_sync(b, P + (k * 16) * N + tj * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(Un + (ti * 16) * N + tj * 16, acc, N, wmma::mem_row_major);
}

// FP16 / BF16 in, FP32 accumulate, rounded back to the input format on store.
template <class T>
__global__ void rd_tc_acc32(const T* P, const T* U, T* Un)
{
    __shared__ float tile[16 * 16];
    int ti = blockIdx.y, tj = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);

    for (int k = 0; k < TILES; ++k) {                       // P U
        wmma::load_matrix_sync(a, P + (ti * 16) * N + k * 16, N);
        wmma::load_matrix_sync(b, U + (k * 16) * N + tj * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    for (int k = 0; k < TILES; ++k) {                       // U P
        wmma::load_matrix_sync(a, U + (ti * 16) * N + k * 16, N);
        wmma::load_matrix_sync(b, P + (k * 16) * N + tj * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = threadIdx.x; e < 256; e += 32) {           // FP32 -> FP16/BF16 (F2F)
        float v = tile[e];
        Un[(ti * 16 + e / 16) * N + tj * 16 + e % 16] = (T)v;
    }
}

// =============================================================================
// Host side
// =============================================================================

template <class T> static T to_T(double v);
template <> __half        to_T<__half>(double v)        { return __float2half((float)v); }
template <> __nv_bfloat16 to_T<__nv_bfloat16>(double v) { return __float2bfloat16((float)v); }
static double from_T(__half v)        { return (double)__half2float(v); }
static double from_T(__nv_bfloat16 v) { return (double)__bfloat162float(v); }

template <class T, class Kernel>
static void run(const char* tag, Kernel kern)
{
    const int NN = N * N;
    std::vector<T> hP(NN), hU(NN);
    double diag = (1.0 + LDT) / 2.0 - 2.0 * R, off = R;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            hP[i * N + j] = to_T<T>(i == j ? diag : (std::abs(i - j) == 1 ? off : 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            hU[i * N + j] = to_T<T>(std::sin(M_PI * (i + 1) * DX) * std::sin(M_PI * (j + 1) * DX));

    T *P, *U, *Un;
    CUDA_CHECK(cudaMalloc(&P,  NN * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&U,  NN * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&Un, NN * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(P, hP.data(), NN * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(U, hU.data(), NN * sizeof(T), cudaMemcpyHostToDevice));

    printf("\n===== %s : 2D reaction-diffusion on tensor cores (N=%dx%d, M=%d) =====\n",
           tag, N, N, M);
    printf("  P: diag %.5f (stored %.5f), off-diag %.5f (stored %.5f)\n",
           diag, from_T(to_T<T>(diag)), off, from_T(to_T<T>(off)));

    dim3 grid(TILES, TILES), block(32);
    int first_inf = -1, first_nan = -1;
    for (int n = 0; n < M; ++n) {
        kern<<<grid, block>>>(P, U, Un);
        CUDA_CHECK(cudaGetLastError());
        std::swap(U, Un);

        CUDA_CHECK(cudaMemcpy(hU.data(), U, NN * sizeof(T), cudaMemcpyDeviceToHost));
        // Count inf and NaN explicitly; a "v > max" reduction would skip NaN silently.
        double mx = 0; int ninf = 0, nnan = 0;
        for (int e = 0; e < NN; ++e) {
            double v = from_T(hU[e]);
            if (std::isnan(v)) ++nnan;
            else if (std::isinf(v)) ++ninf;
            else mx = std::max(mx, std::fabs(v));
        }
        bool news = (ninf && first_inf < 0) || (nnan && first_nan < 0);
        if (ninf && first_inf < 0) first_inf = n + 1;
        if (nnan && first_nan < 0) first_nan = n + 1;
        if ((n + 1) % 100 == 0 || news || (first_inf > 0 && n + 1 <= first_inf + 2)
            || n == M - 1)
            printf("  step %5d (t=%.3f)  max|finite|=%.4e  inf=%5d  nan=%5d\n",
                   n + 1, (n + 1) * DT, mx, ninf, nnan);
        if (nnan == NN) {
            printf("  every entry is NaN at step %d; stopping\n", n + 1);
            break;
        }
    }
    printf("  first inf at step %d, first NaN at step %d\n", first_inf, first_nan);
    cudaFree(P); cudaFree(U); cudaFree(Un);
}

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "all";
    bool all = !strcmp(mode, "all");
    printf("r = %.5f (2D diffusion number, stable if <= 0.25)\n", R);
    printf("lambda*dt = %.4f; mode (1,1) grows by %.6f per step\n", LDT,
           1.0 + LDT - 8.0 * R * std::pow(std::sin(M_PI * DX / 2.0), 2));

    if (all || !strcmp(mode, "fp16acc16")) run<__half>("fp16acc16", rd_tc_fp16acc16);
    if (all || !strcmp(mode, "fp16acc32")) run<__half>("fp16acc32", rd_tc_acc32<__half>);
    if (all || !strcmp(mode, "bf16acc32")) run<__nv_bfloat16>("bf16acc32", rd_tc_acc32<__nv_bfloat16>);
    return 0;
}
