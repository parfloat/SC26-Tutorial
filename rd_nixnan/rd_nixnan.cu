// rd_nixnan.cu
// -----------------------------------------------------------------------------
// Reaction-diffusion FTCS demo for the nixnan tutorial.
//
//     du/dt = D u_xx + lambda u,   x in [0,L],   u(0,t)=u(L,t)=0
//     u(x,0) = sin(pi x / L)          (fundamental mode, satisfies the BCs)
//
// Explicit forward-Euler + central difference:
//     u_i^{n+1} = u_i + r (u_{i+1} - 2 u_i + u_{i-1}) + (lambda dt) u_i
//     r = D dt / dx^2
//
// The reaction term makes the mode grow like exp((lambda - D pi^2) t), so the
// grid values march straight up the FP exponent axis until they hit each
// format's overflow edge.  We run the SAME step in four precisions, one kernel
// each, so nixnan attributes exceptions and the exponent histogram per format:
//
//   FP16 (5-bit exp, max 65504) : overflows ~step 227 (t~0.23); diffusion
//                                 increments go SUBNORMAL near the boundaries.
//   BF16 (8-bit exp, ~3.4e38)   : overflows ~step 1818 (t~1.82); 8-bit mantissa
//                                 makes the u_{i+1}-2u_i+u_{i-1} cancellation
//                                 pure noise (precision loss, not an exception).
//   FP32                        : overflows ~step 1818, clean mantissa.
//   FP64 (11-bit exp, ~1.8e308) : overflows much later; highest precision.
//
// Build:  nvcc -arch=sm_86 -lineinfo rd_nixnan.cu -o rd_nixnan
// Run  :  ./rd_nixnan                                   # baseline blow-up
//         LD_PRELOAD=/path/nixnan.so ./rd_nixnan        # exception summary
//         HISTOGRAM=1 LD_PRELOAD=/path/nixnan.so ./rd_nixnan
//         BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=/path/nixnan.so ./rd_nixnan
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef N
#define N 101          // grid points; dx = L/(N-1) = 0.01
#endif
#ifndef M
#define M 1600         // time steps; dt = 1e-3 -> T = 1.6
#endif

// --- physical / numerical constants (host + device) ---
static const double L      = 1.0;
static const double D      = 0.01;
static const double LAMBDA = 50.0;
static const double DX     = L / (N - 1);      // 0.01
static const double DT     = 1e-3;
static const double R      = D * DT / (DX * DX);   // 0.1   (diffusion number)
static const double LDT    = LAMBDA * DT;          // 0.05  (reaction per step)

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){                \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__);\
    exit(1);} } while(0)

// =============================================================================
// One explicit FTCS step, four precisions, four kernels.
// Everything stays IN-TYPE so nixnan sees genuine HADD/HMUL (f16), the bf16
// variants, FADD/FFMA (f32), and DADD/DFMA (f64) -- no silent promotion.
// =============================================================================

__global__ void rd_step_fp16(const __half* u, __half* un, __half r, __half lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = __float2half(0.f); return; }   // Dirichlet

    __half two  = __float2half(2.f);
    __half lap  = __hsub(__hadd(u[i + 1], u[i - 1]), __hmul(two, u[i])); // 2nd diff
    __half diff = __hmul(r, lap);
    __half reac = __hmul(lam, u[i]);
    un[i] = __hadd(__hadd(u[i], diff), reac);
}

__global__ void rd_step_bf16(const __nv_bfloat16* u, __nv_bfloat16* un,
                             __nv_bfloat16 r, __nv_bfloat16 lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = __float2bfloat16(0.f); return; }

    __nv_bfloat16 two  = __float2bfloat16(2.f);
    __nv_bfloat16 lap  = __hsub(__hadd(u[i + 1], u[i - 1]), __hmul(two, u[i]));
    __nv_bfloat16 diff = __hmul(r, lap);
    __nv_bfloat16 reac = __hmul(lam, u[i]);
    un[i] = __hadd(__hadd(u[i], diff), reac);
}

__global__ void rd_step_fp32(const float* u, float* un, float r, float lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = 0.f; return; }

    float lap = (u[i + 1] + u[i - 1]) - 2.f * u[i];
    un[i] = u[i] + r * lap + lam * u[i];   // FADD / FFMA in SASS
}

__global__ void rd_step_fp64(const double* u, double* un, double r, double lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i == 0 || i == N - 1) { un[i] = 0.0; return; }

    double lap = (u[i + 1] + u[i - 1]) - 2.0 * u[i];
    un[i] = u[i] + r * lap + lam * u[i];   // DADD / DFMA in SASS
}

// =============================================================================
// Host drivers.  One per format so each blow-up timeline stays clean.
// Host-side conversions (__*2float / __float2*) are host-callable in CUDA 12.
// =============================================================================

template <class T> static T   to_T(double);              // convert double -> T
template <class T> static double from_T(T);               // convert T -> double
template <> __half           to_T<__half>(double v){ return __float2half((float)v); }
template <> __nv_bfloat16    to_T<__nv_bfloat16>(double v){ return __float2bfloat16((float)v); }
template <> float            to_T<float>(double v){ return (float)v; }
template <> double           to_T<double>(double v){ return v; }
template <> double from_T<__half>(__half v){ return (double)__half2float(v); }
template <> double from_T<__nv_bfloat16>(__nv_bfloat16 v){ return (double)__bfloat162float(v); }
template <> double from_T<float>(float v){ return (double)v; }
template <> double from_T<double>(double v){ return v; }

template <class T, class Kernel>
static void run(const char* tag, Kernel kern)
{
    std::vector<T> h(N);
    for (int i = 0; i < N; ++i)
        h[i] = to_T<T>(std::sin(M_PI * (i * DX) / L));

    T *u, *un;
    CUDA_CHECK(cudaMalloc(&u,  N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&un, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(u, h.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    T r   = to_T<T>(R);
    T lam = to_T<T>(LDT);
    dim3 block(128), grid((N + 127) / 128);

    printf("\n===== %s : reaction-diffusion FTCS (N=%d, M=%d, dt=%.0e) =====\n",
           tag, N, M, DT);

    int first_bad = -1;
    for (int n = 0; n < M; ++n) {
        kern<<<grid, block>>>(u, un, r, lam);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u, un);

        if ((n + 1) % 100 == 0 || n == M - 1) {
            CUDA_CHECK(cudaMemcpy(h.data(), u, N * sizeof(T), cudaMemcpyDeviceToHost));
            // Stay in double here: from_T already returns double, and for T=double
            // that value can legitimately exceed float's ~3.4e38 max while still
            // being a perfectly finite double. Narrowing to float first would call
            // isfinite() on a value the cast itself turned into +Infinity.
            double mx = -1e30; bool bad = false;
            for (int i = 0; i < N; ++i) {
                double v = from_T<T>(h[i]);
                if (!std::isfinite(v)) bad = true;
                if (std::isfinite(v) && v > mx) mx = v;
            }
            if (bad && first_bad < 0) first_bad = n + 1;
            printf("  step %5d (t=%.3f)  max=%.4e%s\n",
                   n + 1, (n + 1) * DT, mx, bad ? "   <-- non-finite present" : "");
        }
    }
    if (first_bad > 0)
        printf("  first non-finite at step %d (t=%.3f)\n", first_bad, first_bad * DT);

    cudaFree(u); cudaFree(un);
}

#ifndef FP64_LONGRUN
#define FP64_LONGRUN 0
#endif

#if !FP64_LONGRUN

int main()
{
    printf("r = %.4f (diffusion number, stable if <= 0.5)\n", R);
    printf("lambda*dt = %.4f (reaction growth per step ~ %.4f x)\n", LDT, 1.0 + LDT);

    run<__half>       ("FP16", rd_step_fp16);
    run<__nv_bfloat16>("BF16", rd_step_bf16);
    run<float>        ("FP32", rd_step_fp32);
    run<double>       ("FP64", rd_step_fp64);

    printf("\nDone. Re-run under LD_PRELOAD=nixnan.so with HISTOGRAM=1 "
           "and BIN_SPEC_FILE=spec.json.\n");
    return 0;
}

#else  // FP64_LONGRUN: interleave all four to SUPPRESS_AFTER, then step FP64
       // alone until it, too, goes non-finite (or M_FP64_MAX as a safety cap).

#ifndef SUPPRESS_AFTER
#define SUPPRESS_AFTER 3800   // 2x the FP32/BF16 trigger step (1900) at M=2000
#endif
#ifndef M_FP64_MAX
#define M_FP64_MAX 20000      // safety cap; real FP64 overflow is expected ~14500-14600
#endif

int main()
{
    printf("r = %.4f (diffusion number, stable if <= 0.5)\n", R);
    printf("lambda*dt = %.4f (reaction growth per step ~ %.4f x)\n", LDT, 1.0 + LDT);
    printf("FP64_LONGRUN: FP16/BF16/FP32 suppressed after step %d "
           "(2x the FP32 trigger step); FP64 alone continues to step %d "
           "or first non-finite.\n\n", SUPPRESS_AFTER, M_FP64_MAX);

    std::vector<__half>        h16(N);
    std::vector<__nv_bfloat16> hb16(N);
    std::vector<float>         h32(N);
    std::vector<double>        h64(N);
    for (int i = 0; i < N; ++i) {
        double v0 = std::sin(M_PI * (i * DX) / L);
        h16[i]  = to_T<__half>(v0);
        hb16[i] = to_T<__nv_bfloat16>(v0);
        h32[i]  = to_T<float>(v0);
        h64[i]  = to_T<double>(v0);
    }

    __half        *u16, *un16;
    __nv_bfloat16 *ub16, *unb16;
    float         *u32, *un32;
    double        *u64, *un64;
    CUDA_CHECK(cudaMalloc(&u16,  N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&un16, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&ub16,  N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&unb16, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&u32,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&un32, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u64,  N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&un64, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(u16,  h16.data(),  N * sizeof(__half),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ub16, hb16.data(), N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(u32,  h32.data(),  N * sizeof(float),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(u64,  h64.data(),  N * sizeof(double),        cudaMemcpyHostToDevice));

    __half        r16  = to_T<__half>(R),        lam16  = to_T<__half>(LDT);
    __nv_bfloat16 rb16 = to_T<__nv_bfloat16>(R),  lamb16 = to_T<__nv_bfloat16>(LDT);
    float         r32  = to_T<float>(R),          lam32  = to_T<float>(LDT);
    double        r64  = to_T<double>(R),         lam64  = to_T<double>(LDT);

    dim3 block(128), grid((N + 127) / 128);

    int first_bad16 = -1, first_badb16 = -1, first_bad32 = -1, first_bad64 = -1;
    int n = 0;
    for (; n < SUPPRESS_AFTER; ++n) {
        rd_step_fp16<<<grid, block>>>(u16, un16, r16, lam16);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u16, un16);
        rd_step_bf16<<<grid, block>>>(ub16, unb16, rb16, lamb16);
        CUDA_CHECK(cudaGetLastError());
        std::swap(ub16, unb16);
        rd_step_fp32<<<grid, block>>>(u32, un32, r32, lam32);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u32, un32);
        rd_step_fp64<<<grid, block>>>(u64, un64, r64, lam64);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u64, un64);

        if ((n + 1) % 100 == 0 || n == SUPPRESS_AFTER - 1) {
            CUDA_CHECK(cudaMemcpy(h16.data(),  u16,  N * sizeof(__half),        cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb16.data(), ub16, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h32.data(),  u32,  N * sizeof(float),         cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h64.data(),  u64,  N * sizeof(double),        cudaMemcpyDeviceToHost));

            double mx16 = -1e30, mxb16 = -1e30, mx32 = -1e30, mx64 = -1e30;
            bool bad16 = false, badb16 = false, bad32 = false, bad64 = false;
            for (int i = 0; i < N; ++i) {
                double v16  = from_T<__half>(h16[i]);
                double vb16 = from_T<__nv_bfloat16>(hb16[i]);
                double v32  = from_T<float>(h32[i]);
                double v64  = from_T<double>(h64[i]);
                if (!std::isfinite(v16))  bad16  = true; else if (v16  > mx16)  mx16  = v16;
                if (!std::isfinite(vb16)) badb16 = true; else if (vb16 > mxb16) mxb16 = vb16;
                if (!std::isfinite(v32))  bad32  = true; else if (v32  > mx32)  mx32  = v32;
                if (!std::isfinite(v64))  bad64  = true; else if (v64  > mx64)  mx64  = v64;
            }
            if (bad16  && first_bad16  < 0) first_bad16  = n + 1;
            if (badb16 && first_badb16 < 0) first_badb16 = n + 1;
            if (bad32  && first_bad32  < 0) first_bad32  = n + 1;
            if (bad64  && first_bad64  < 0) first_bad64  = n + 1;
            printf("  step %6d (t=%.3f)  FP16=%.3e%s BF16=%.3e%s FP32=%.3e%s FP64=%.3e%s\n",
                   n + 1, (n + 1) * DT,
                   mx16,  bad16  ? "*" : " ", mxb16, badb16 ? "*" : " ",
                   mx32,  bad32  ? "*" : " ", mx64,  bad64  ? "*" : " ");
        }
    }

    printf("\nInterleaved phase done (steps 1..%d):\n", SUPPRESS_AFTER);
    printf("  FP16 first non-finite at step %d\n", first_bad16);
    printf("  BF16 first non-finite at step %d\n", first_badb16);
    printf("  FP32 first non-finite at step %d\n", first_bad32);
    printf("Suppressing FP16/BF16/FP32 from here on (already non-finite well "
           "before this point); FP64 continues alone.\n\n");
    cudaFree(u16); cudaFree(un16);
    cudaFree(ub16); cudaFree(unb16);
    cudaFree(u32); cudaFree(un32);

    for (; n < M_FP64_MAX; ++n) {
        rd_step_fp64<<<grid, block>>>(u64, un64, r64, lam64);
        CUDA_CHECK(cudaGetLastError());
        std::swap(u64, un64);

        if ((n + 1) % 100 == 0) {
            CUDA_CHECK(cudaMemcpy(h64.data(), u64, N * sizeof(double), cudaMemcpyDeviceToHost));
            double mx64 = -1e30; bool bad64 = false;
            for (int i = 0; i < N; ++i) {
                double v64 = from_T<double>(h64[i]);
                if (!std::isfinite(v64)) bad64 = true;
                else if (v64 > mx64) mx64 = v64;
            }
            printf("  step %6d (t=%.3f)  FP64 max=%.4e%s\n",
                   n + 1, (n + 1) * DT, mx64, bad64 ? "   <-- non-finite present" : "");
            if (bad64) { first_bad64 = n + 1; break; }
        }
    }
    if (first_bad64 < 0)
        printf("\nFP64 never went non-finite through step %d (hit M_FP64_MAX cap).\n", n);
    else
        printf("\nFP64 first non-finite at step %d (t=%.3f).\n",
               first_bad64, first_bad64 * DT);

    cudaFree(u64); cudaFree(un64);
    printf("\nDone. Re-run under LD_PRELOAD=nixnan.so to see FP64 exceptions "
           "classified for real this time.\n");
    return 0;
}

#endif // FP64_LONGRUN
