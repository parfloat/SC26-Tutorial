// leimao-tc-variant-bf16.cu -- BF16 tensor-core (WMMA/HMMA-family) GEMM,
// the variant tc_matmult/report.tex, Section "BF16: Supported by the
// Header, Not Exercised by This File", described but leimao-mm.cu never
// instantiates: __nv_bfloat16 does not appear anywhere in that file.
//
// Not from Lei Mao's post -- written here as a direct, same-shape port of
// leimao-mm.cu's HMMA kernel (Section "FP16" of the report), with the
// element type swapped to __nv_bfloat16 and the accumulator forced to
// float (the only accumulator type the CUDA 12.4 mma.hpp header defines
// for BF16 -- confirmed there, no __nv_bfloat16 accumulator overload
// exists at all, unlike FP16).
//
// Restricted to row-major A, row-major B only (report.tex, Section
// "Scope") -- the case this whole study cares about, matching
// leimao-simt-variant-fp32.cu/leimao-simt-variant-fp16.cu's scope, not
// leimao-mm.cu's general four-transpose-combination interface.
//
// Build: nvcc leimao-tc-variant-bf16.cu -o leimao-tc-variant-bf16 --gpu-architecture=compute_86

#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

// Tensor-core (BMMA/HMMA-family) GEMM. A is row-major M x K, B is row-major
// K x N, C is column-major M x N (same convention as leimao-mm.cu and the
// SIMT variants, so the CPU reference and correctness check are directly
// comparable). One warp computes one 16x16 output tile per K-step, via
// nvcuda::wmma::mma_sync -- a single tensor-core instruction per tile, not
// 4096 scalar multiply-adds.
__global__ void wmma_gemm_row_major_a_row_major_b_bf16(
    __nv_bfloat16 const* A, __nv_bfloat16 const* B, float* C, uint32_t m,
    uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc,
    float alpha, float beta)
{
    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, nvcuda::wmma::row_major>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, nvcuda::wmma::row_major>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>
        c_frag{};

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    uint32_t const m_start{warpM * WMMA_M};
    uint32_t const n_start{warpN * WMMA_N};

    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        if (m_start < m && ki < k && n_start < n)
        {
            // A[m_start, ki], row-major: A + m_start * lda + ki
            __nv_bfloat16 const* a_mptr{A + m_start * lda + ki};
            // B[ki, n_start], row-major: B + ki * ldb + n_start
            __nv_bfloat16 const* b_mptr{B + ki * ldb + n_start};

            nvcuda::wmma::load_matrix_sync(a_frag, a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, b_mptr, ldb);

            // One BF16 tensor-core instruction computes this whole
            // 16x16x16 tile's contribution.
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (m_start < m && n_start < n)
    {
        // C[m_start, n_start], column-major: C + n_start * ldc + m_start
        float* c_mptr{C + n_start * ldc + m_start};
        nvcuda::wmma::load_matrix_sync(c_frag, c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);
        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        nvcuda::wmma::store_matrix_sync(c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}

void launch_tc_mm(__nv_bfloat16 const* A, __nv_bfloat16 const* B, float* C,
                  uint32_t m, uint32_t n, uint32_t k, cudaStream_t stream)
{
    uint32_t const lda{k};
    uint32_t const ldb{n};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WARP_SIZE{32};

    // Same block/grid sizing scheme as leimao-mm.cu's launch_wmma_mm:
    // 4x4 warps per block (128x4 threads), each warp a 16x16 output tile,
    // each block a 64x64 output tile.
    int const num_warps_x{4};
    int const num_warps_y{4};
    dim3 blockDim(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 gridDim((m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x),
                (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y));

    wmma_gemm_row_major_a_row_major_b_bf16<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, m, n, k, lda, ldb, ldc, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

// CPU reference (in float, from the same float source data before
// __float2bfloat16 rounding) -- same convention as
// leimao-simt-variant-fp16.cu's cpu_reference.
void cpu_reference(float const* A, float const* B, float* C, uint32_t m,
                   uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb,
                   uint32_t ldc)
{
    for (uint32_t ni{0}; ni < n; ++ni)
    {
        for (uint32_t mi{0}; mi < m; ++mi)
        {
            float accum{0.0f};
            for (uint32_t ki{0}; ki < k; ++ki)
            {
                accum += A[mi * lda + ki] * B[ki * ldb + ni];
            }
            C[ni * ldc + mi] = accum;
        }
    }
}

void fill_random_float_values(float* arr, size_t n,
                              std::default_random_engine& e)
{
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(e);
    }
}

void float2bfloat16(__nv_bfloat16* bf16_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        bf16_arr[i] = __float2bfloat16(float_arr[i]);
    }
}

float get_avg_abs_diff_ratio(float const* arr_1, float const* arr_2, size_t n)
{
    float sum_abs_diff_ratio{0};
    for (size_t i{0}; i < n; ++i)
    {
        sum_abs_diff_ratio += std::abs(arr_1[i] - arr_2[i]) /
                              std::abs(arr_1[i] + arr_2[i]);
    }
    return sum_abs_diff_ratio / n;
}

int main()
{
    constexpr int num_repeats{10};
    constexpr int num_warmups{10};

    uint32_t const matrix_size_m{1024};
    uint32_t const matrix_size_n{1024};
    uint32_t const matrix_size_k{1024};
    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;
    std::cout << "BF16 tensor core (WMMA, float accumulate)" << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__nv_bfloat16> matrix_a_bf16(matrix_size_m * matrix_size_k);
    std::vector<__nv_bfloat16> matrix_b_bf16(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_reference(matrix_size_m * matrix_size_n);

    float* h_a_float{matrix_a_float.data()};
    float* h_b_float{matrix_b_float.data()};
    __nv_bfloat16* h_a_bf16{matrix_a_bf16.data()};
    __nv_bfloat16* h_b_bf16{matrix_b_bf16.data()};
    float* h_c{matrix_c.data()};
    float* h_c_reference{matrix_c_reference.data()};

    fill_random_float_values(h_a_float, matrix_a_float.size(), random_engine);
    fill_random_float_values(h_b_float, matrix_b_float.size(), random_engine);
    fill_random_float_values(h_c, matrix_c.size(), random_engine);
    fill_random_float_values(h_c_reference, matrix_c_reference.size(),
                             random_engine);
    float2bfloat16(h_a_bf16, h_a_float, matrix_a_float.size());
    float2bfloat16(h_b_bf16, h_b_float, matrix_b_float.size());

    __nv_bfloat16 *d_a, *d_b;
    float* d_c;
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_a, matrix_size_m * matrix_size_k * sizeof(__nv_bfloat16)));
    CHECK_CUDA_ERROR(cudaMalloc(
        &d_b, matrix_size_k * matrix_size_n * sizeof(__nv_bfloat16)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_c, matrix_size_m * matrix_size_n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a_bf16,
                                matrix_a_float.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b_bf16,
                                matrix_b_float.size() * sizeof(__nv_bfloat16),
                                cudaMemcpyHostToDevice));

    // Correctness: reference computed from the float source data, compared
    // against the BF16-input tensor-core result.
    cpu_reference(h_a_float, h_b_float, h_c_reference, matrix_size_m,
                 matrix_size_n, matrix_size_k, matrix_size_k, matrix_size_n,
                 matrix_size_m);
    launch_tc_mm(d_a, d_b, d_c, matrix_size_m, matrix_size_n, matrix_size_k,
                stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c,
                                matrix_c.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));

    float const avg_abs_diff_ratio{
        get_avg_abs_diff_ratio(h_c, h_c_reference, matrix_c.size())};
    if (avg_abs_diff_ratio > 0.01)
    {
        std::cout << "Got high average absolute diff ratio: "
                  << avg_abs_diff_ratio << std::endl;
    }

    // Performance measurement.
    std::function<void(cudaStream_t)> const function_bf16{
        std::bind(launch_tc_mm, d_a, d_b, d_c, matrix_size_m, matrix_size_n,
                  matrix_size_k, std::placeholders::_1)};
    float const latency_bf16{
        measure_performance(function_bf16, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "BF16 Tensor Core Latency: " << latency_bf16 << " ms"
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
