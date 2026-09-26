// leimao-simt-variant-fp16-128x128-overflow.cu -- FP16 overflow experiment,
// genuine SIMT (ordinary CUDA-core) side.
//
// Companion to leimao-mm-128x128-overflow.cu. That file showed a tensor
// core with a *narrow* (__half) accumulator overflows just as readily as
// ordinary arithmetic once inputs are large enough -- it is specifically
// the *float* accumulator (launch_wmma_mm<__half, float>, what
// leimao-mm-128x128.cu actually uses) that avoids overflow, not something
// inherent to tensor-core hardware.
//
// This file asks the SIMT-side version of the same question: does an
// *ordinary* CUDA-core FP16 multiply-accumulate -- no WMMA/tensor-core
// instructions at all, real __half arithmetic the whole way (__hmul/__hadd via
// operator*/operator+, no __half2float promotion) -- also overflow at this
// input magnitude, and can the accumulation step itself (not just an
// individual product) turn Inf into NaN via Inf + (-Inf)?
//
// Unlike leimao-simt-variant-fp16-128x128.cu (which accumulates in float,
// deliberately matching HMMA's precision contract for a fair perf
// comparison), the kernel here accumulates in __half the whole way -- the
// "no wide accumulator at all" case.
//
// Build: nvcc -arch=sm_86 -g -lineinfo leimao-simt-variant-fp16-128x128-overflow.cu -o leimao-simt-variant-fp16-128x128-overflow

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE = 32;

// Same shared-memory tiling as leimao-simt-variant-fp16-128x128.cu, but the
// accumulator `acc` is __half, and the inner-loop multiply-add is real
// half-precision arithmetic (operator*/operator+ on __half resolve to
// __hmul/__hadd on device, no promotion to float anywhere in this kernel).
__global__ void simt_gemm_row_major_a_row_major_b_fp16_pure_half(
    __half const* A, __half const* B, __half* C, uint32_t m, uint32_t n,
    uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    uint32_t const row{blockIdx.y * TILE + threadIdx.y};
    uint32_t const col{blockIdx.x * TILE + threadIdx.x};

    __half acc{__float2half(0.0f)};
    uint32_t const num_tiles{(k + TILE - 1) / TILE};
    for (uint32_t t{0}; t < num_tiles; ++t)
    {
        uint32_t const a_col{t * TILE + threadIdx.x};
        uint32_t const b_row{t * TILE + threadIdx.y};

        As[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? A[row * lda + a_col] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? B[b_row * ldb + col] : __float2half(0.0f);

        __syncthreads();

#pragma unroll
        for (uint32_t kk{0}; kk < TILE; ++kk)
        {
            // Pure __half multiply-accumulate: the product itself is
            // rounded to half (unlike HMMA's internally-extended-precision
            // product), and the running sum is stored in half the whole
            // time -- no wide accumulator anywhere in this kernel.
            acc = acc + As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        // C[row, col], column-major: C[col * ldc + row]
        C[col * ldc + row] = acc;
    }
}

void launch_simt_mm(__half const* A, __half const* B, __half* C, uint32_t m,
                     uint32_t n, uint32_t k, cudaStream_t stream)
{
    uint32_t const lda{k};
    uint32_t const ldb{n};
    uint32_t const ldc{m};

    dim3 const blockDim(TILE, TILE);
    dim3 const gridDim((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    simt_gemm_row_major_a_row_major_b_fp16_pure_half<<<gridDim, blockDim, 0,
                                                        stream>>>(
        A, B, C, m, n, k, lda, ldb, ldc);
    CHECK_LAST_CUDA_ERROR();
}

// Raised from leimao-simt-variant-fp16-128x128.cu's (-256, 256) -- see
// leimao-mm-128x128-overflow.cu's header for why. Already sign-dithered
// (the distribution is symmetric about 0).
void fill_random_float_values(float* arr, size_t n,
                              std::default_random_engine& e, float range)
{
    std::uniform_real_distribution<float> uniform_dist(-range, range);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void float2half(__half* half_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
    }
}

struct NonFiniteCounts
{
    size_t nan{0};
    size_t pos_inf{0};
    size_t neg_inf{0};
    size_t finite{0};
};

NonFiniteCounts count_non_finite_half(__half const* arr, size_t n)
{
    NonFiniteCounts c;
    for (size_t i{0}; i < n; ++i)
    {
        float const v{__half2float(arr[i])};
        if (std::isnan(v)) ++c.nan;
        else if (std::isinf(v)) (v > 0 ? ++c.pos_inf : ++c.neg_inf);
        else ++c.finite;
    }
    return c;
}

std::ostream& operator<<(std::ostream& os, NonFiniteCounts const& c)
{
    os << "finite=" << c.finite << " NaN=" << c.nan
       << " +Inf=" << c.pos_inf << " -Inf=" << c.neg_inf;
    return os;
}

int main()
{
    uint32_t const matrix_size_m{128};
    uint32_t const matrix_size_n{128};
    uint32_t const matrix_size_k{128};
    float const range{2000.0f};

    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;
    std::cout << "Input range: uniform(-" << range << ", " << range << ")"
              << std::endl;
    std::cout << "FP16 SIMT overflow experiment (pure __half accumulate, "
                 "TILE=" << TILE << ")" << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_c(matrix_size_m * matrix_size_n);

    float* h_a_float{matrix_a_float.data()};
    float* h_b_float{matrix_b_float.data()};
    __half* h_a_half{matrix_a_half.data()};
    __half* h_b_half{matrix_b_half.data()};
    __half* h_c{matrix_c.data()};

    fill_random_float_values(h_a_float, matrix_a_float.size(), random_engine,
                             range);
    fill_random_float_values(h_b_float, matrix_b_float.size(), random_engine,
                             range);
    float2half(h_a_half, h_a_float, matrix_a_float.size());
    float2half(h_b_half, h_b_float, matrix_b_float.size());

    __half *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_a, matrix_size_m * matrix_size_k * sizeof(__half)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_b, matrix_size_k * matrix_size_n * sizeof(__half)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_c, matrix_size_m * matrix_size_n * sizeof(__half)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    launch_simt_mm(d_a, d_b, d_c, matrix_size_m, matrix_size_n,
                   matrix_size_k, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, matrix_c.size() * sizeof(__half),
                                cudaMemcpyDeviceToHost));

    std::cout << "C = A * B (row-major x row-major), pure-half accumulate: "
              << count_non_finite_half(h_c, matrix_c.size()) << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
