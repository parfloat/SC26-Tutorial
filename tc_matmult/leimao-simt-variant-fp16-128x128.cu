// leimao-simt-variant-fp16.cu -- SIMT (ordinary CUDA-core) counterpart to
// leimao-mm.cu's tensor-core WMMA kernel, FP16.
//
// Not from Lei Mao's post -- written here as the SIMT benchmark counterpart
// described in tc_matmult/report.tex, Section "Is Any of This SIMT-Core
// Matrix Multiplication?": leimao-mm.cu contains zero SIMT-core GPU matrix
// multiplication (its only non-tensor-core matmul is a CPU-side reference
// function). This file is that missing SIMT kernel, made real and
// benchmarkable, restricted to the case this whole study cares about: A
// row-major, B row-major (see report.tex, Section "Scope").
//
// A and B are stored as __half (matching leimao-mm.cu's HMMA path); the
// accumulate happens in float (via __half2float), and C is float -- the
// same accumulation precision leimao-mm.cu's launch_wmma_mm<__half, float>
// instantiation uses. That keeps the SIMT-vs-tensor-core comparison isolated
// to "which hardware path does the multiply-accumulate", not conflated with
// a difference in accumulation precision.
//
// Build: nvcc leimao-simt-variant-fp16-128x128.cu -o leimao-simt-variant-fp16-128x128 --gpu-architecture=compute_86
//
// 128x128x128 variant of leimao-simt-variant-fp16.cu (matrix_size_m/n/k
// changed from 1024 to 128, everything else identical) -- see
// leimao-mm-128x128.cu for why (nixnan observation at BIN_SPEC_FILE's
// finest "count":1 granularity needs a much smaller problem to finish in
// a reasonable time).

#include <cassert>
#include <chrono>
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

// Shared-memory tiled SIMT GEMM. A is row-major M x K, B is row-major K x N,
// C is column-major M x N (matching leimao-mm.cu's WMMA kernel output
// convention). One ordinary CUDA thread computes one output element; no
// wmma/tensor-core instructions anywhere in this kernel -- accumulation is
// done with plain float arithmetic after __half2float conversion.
constexpr int TILE = 32;

__global__ void simt_gemm_row_major_a_row_major_b_fp16(
    __half const* A, __half const* B, float* C, uint32_t m, uint32_t n,
    uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc, float alpha,
    float beta)
{
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    uint32_t const row{blockIdx.y * TILE + threadIdx.y}; // index into M
    uint32_t const col{blockIdx.x * TILE + threadIdx.x}; // index into N

    float acc{0.0f};
    uint32_t const num_tiles{(k + TILE - 1) / TILE};
    for (uint32_t t{0}; t < num_tiles; ++t)
    {
        uint32_t const a_col{t * TILE + threadIdx.x}; // K-index into A
        uint32_t const b_row{t * TILE + threadIdx.y}; // K-index into B

        // A[row, a_col], row-major: A[row * lda + a_col]
        As[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? A[row * lda + a_col] : __float2half(0.0f);
        // B[b_row, col], row-major: B[b_row * ldb + col]
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? B[b_row * ldb + col] : __float2half(0.0f);

        __syncthreads();

#pragma unroll
        for (uint32_t kk{0}; kk < TILE; ++kk)
        {
            // Ordinary scalar FMA on the SIMT (CUDA) cores, accumulated in
            // float -- no tensor-core instruction is issued anywhere here.
            acc += __half2float(As[threadIdx.y][kk]) *
                   __half2float(Bs[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < m && col < n)
    {
        // C[row, col], column-major: C[col * ldc + row]
        float const c_prev{C[col * ldc + row]};
        C[col * ldc + row] = alpha * acc + beta * c_prev;
    }
}

void launch_simt_mm(__half const* A, __half const* B, float* C, uint32_t m,
                     uint32_t n, uint32_t k, cudaStream_t stream)
{
    // Row-major A, row-major B only -- this file's whole purpose is a
    // benchmark counterpart for exactly that case (report.tex, Section
    // "Scope"), not a general-purpose GEMM library.
    uint32_t const lda{k};
    uint32_t const ldb{n};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    dim3 const blockDim(TILE, TILE);
    dim3 const gridDim((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    simt_gemm_row_major_a_row_major_b_fp16<<<gridDim, blockDim, 0, stream>>>(
        A, B, C, m, n, k, lda, ldb, ldc, alpha, beta);
    CHECK_LAST_CUDA_ERROR();
}

// CPU reference (in float, from the same float source data before
// __float2half rounding) -- matches leimao-mm.cu's
// mm_a_col_major_b_col_major in its is_A_transpose=true, is_B_transpose=true
// branch, and its get_avg_abs_diff_ratio-based correctness check.
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

void float2half(__half* half_arr, float const* float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
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

    uint32_t const matrix_size_m{128};
    uint32_t const matrix_size_n{128};
    uint32_t const matrix_size_k{128};
    std::cout << "Matrix Sizes" << std::endl;
    std::cout << "M: " << matrix_size_m << std::endl;
    std::cout << "N: " << matrix_size_n << std::endl;
    std::cout << "K: " << matrix_size_k << std::endl;
    std::cout << "FP16 SIMT (shared-memory tiled, TILE=" << TILE
              << ", float accumulate)" << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_reference(matrix_size_m * matrix_size_n);

    float* h_a_float{matrix_a_float.data()};
    float* h_b_float{matrix_b_float.data()};
    __half* h_a_half{matrix_a_half.data()};
    __half* h_b_half{matrix_b_half.data()};
    float* h_c{matrix_c.data()};
    float* h_c_reference{matrix_c_reference.data()};

    fill_random_float_values(h_a_float, matrix_a_float.size(), random_engine);
    fill_random_float_values(h_b_float, matrix_b_float.size(), random_engine);
    fill_random_float_values(h_c, matrix_c.size(), random_engine);
    fill_random_float_values(h_c_reference, matrix_c_reference.size(),
                             random_engine);
    float2half(h_a_half, h_a_float, matrix_a_float.size());
    float2half(h_b_half, h_b_float, matrix_b_float.size());

    __half *d_a, *d_b;
    float* d_c;
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_a, matrix_size_m * matrix_size_k * sizeof(__half)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_b, matrix_size_k * matrix_size_n * sizeof(__half)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_c, matrix_size_m * matrix_size_n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    // Correctness: reference computed from the float source data (same
    // convention as leimao-mm.cu), compared against the FP16-input SIMT result.
    cpu_reference(h_a_float, h_b_float, h_c_reference, matrix_size_m,
                 matrix_size_n, matrix_size_k, matrix_size_k, matrix_size_n,
                 matrix_size_m);
    launch_simt_mm(d_a, d_b, d_c, matrix_size_m, matrix_size_n,
                   matrix_size_k, stream);
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
    std::function<void(cudaStream_t)> const function_simt{
        std::bind(launch_simt_mm, d_a, d_b, d_c, matrix_size_m,
                  matrix_size_n, matrix_size_k, std::placeholders::_1)};
    float const latency_simt{
        measure_performance(function_simt, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "SIMT FP16 Latency: " << latency_simt << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
