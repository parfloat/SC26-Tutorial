// leimao-mm-128x128-overflow.cu -- FP16 overflow experiment, tensor-core
// (WMMA) side.
//
// Question this file answers: tc_matmult's earlier nixnan runs
// (leimao-mm-128x128.cu, uniform(-256,256) inputs) produced zero FP
// exceptions of any kind. That is because HMMA's multiply-accumulate is
// computed at full internal precision into a float accumulator
// (launch_wmma_mm<__half, float>) -- the accumulator's ~3.4e38 range makes
// overflow essentially unreachable at that input magnitude. Does a tensor
// core still avoid overflow if forced to use a genuinely narrow (__half)
// accumulator instead of float, once inputs are large enough that
// individual products already exceed FP16's ~65504 max magnitude?
//
// Changes from leimao-mm-128x128.cu:
//   - Random input magnitude raised from uniform(-256,256) to
//     uniform(-2000,2000). Values are already sign-dithered (the
//     distribution is symmetric about 0); at this magnitude, a_i*b_i can
//     reach ~4,000,000 -- ~61x past FP16's max finite value -- so
//     individual products can already overflow FP16 range on their own.
//   - Every transpose combination now runs launch_wmma_mm<__half, float>
//     (control, wide accumulator, same as leimao-mm-128x128.cu) *and*
//     launch_wmma_mm<__half, __half> (narrow accumulator) back to back,
//     and both outputs are scanned for NaN/Inf independent of nixnan, as a
//     sanity check that overflow genuinely occurred before ever involving
//     the instrumentation.
//   - No performance measurement here (not the point of this file); one
//     untimed launch per (transpose combo, accumulator type).
//
// Build: nvcc -arch=sm_86 -g -lineinfo leimao-mm-128x128-overflow.cu -o leimao-mm-128x128-overflow

#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

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

// Identical WMMA kernel and launcher to leimao-mm-128x128.cu -- unchanged,
// generic over the accumulator type T2.
template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename WMMA_FRAG_LAYOUT_A, typename WMMA_FRAG_LAYOUT_B>
__global__ void wmma_gemm_a_col_major_b_col_major(
    T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, bool is_A_transpose,
    bool is_B_transpose, float alpha, float beta)
{
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_A>
        a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1,
                           WMMA_FRAG_LAYOUT_B>
        b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           T2>
        c_frag{};

    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki
                                                           : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M
                                                           : ki};
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N
                                                           : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki
                                                           : warpN * WMMA_N};

        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const* matrix_mma_a_mptr{A + matrix_mma_a_row_idx +
                                        matrix_mma_a_col_idx * lda};
            T1 const* matrix_mma_b_mptr{B + matrix_mma_b_row_idx +
                                        matrix_mma_b_col_idx * ldb};
            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);

            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
    {
        T2* matrix_mma_c_mptr{C + matrix_mma_c_row_idx +
                              matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc,
                                       nvcuda::wmma::mem_col_major);
        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            // Widened to float for the scale/accumulate arithmetic itself
            // (ambiguous operator* between float and __half otherwise);
            // narrowed back to T2 on assignment, same as the original
            // (float-only) leimao-mm.cu when T2=float. With alpha=1,
            // beta=0 this is just acc_frag.x[i] cast to T2 -- for T2=half,
            // that narrowing is exactly where overflow can produce Inf/NaN.
            c_frag.x[i] = static_cast<T2>(
                alpha * static_cast<float>(acc_frag.x[i]) +
                beta * static_cast<float>(c_frag.x[i]));
        }
        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc,
                                        nvcuda::wmma::mem_col_major);
    }
}

template <typename T1, typename T2>
void launch_wmma_mm(T1 const* A, T1 const* B, T2* C, uint32_t m, uint32_t n,
                    uint32_t k, bool is_A_transpose, bool is_B_transpose,
                    cudaStream_t stream)
{
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};
    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    if ((!is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    else if ((is_A_transpose) && (!is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::col_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    else if ((!is_A_transpose) && (is_B_transpose))
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::col_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    else
    {
        wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K,
                                          nvcuda::wmma::row_major,
                                          nvcuda::wmma::row_major>
            <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc,
                                               is_A_transpose, is_B_transpose,
                                               alpha, beta);
    }
    CHECK_LAST_CUDA_ERROR();
}

// Raised from leimao-mm-128x128.cu's (-256, 256) -- see file header. Already
// sign-dithered: std::uniform_real_distribution(-R, R) is symmetric about 0.
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

// Independent of nixnan: count how many output elements are NaN, +Inf, or
// -Inf, so we can confirm overflow actually happened before ever looking at
// an instrumented run.
struct NonFiniteCounts
{
    size_t nan{0};
    size_t pos_inf{0};
    size_t neg_inf{0};
    size_t finite{0};
};

NonFiniteCounts count_non_finite_float(float const* arr, size_t n)
{
    NonFiniteCounts c;
    for (size_t i{0}; i < n; ++i)
    {
        if (std::isnan(arr[i])) ++c.nan;
        else if (std::isinf(arr[i])) (arr[i] > 0 ? ++c.pos_inf : ++c.neg_inf);
        else ++c.finite;
    }
    return c;
}

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

void print_test_header(bool is_A_transpose, bool is_B_transpose)
{
    if ((!is_A_transpose) && (!is_B_transpose))
        std::cout << "C = A * B" << std::endl;
    else if ((is_A_transpose) && (!is_B_transpose))
        std::cout << "C = A^T * B" << std::endl;
    else if ((!is_A_transpose) && (is_B_transpose))
        std::cout << "C = A * B^T" << std::endl;
    else
        std::cout << "C = A^T * B^T" << std::endl;
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
    std::cout << "FP16 HMMA overflow experiment (float vs. half accumulator)"
              << std::endl;

    std::default_random_engine random_engine(0);

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c_float(matrix_size_m * matrix_size_n);
    std::vector<__half> matrix_c_half(matrix_size_m * matrix_size_n);

    float* h_matrix_a_float{matrix_a_float.data()};
    float* h_matrix_b_float{matrix_b_float.data()};
    __half* h_matrix_a_half{matrix_a_half.data()};
    __half* h_matrix_b_half{matrix_b_half.data()};
    float* h_matrix_c_float{matrix_c_float.data()};
    __half* h_matrix_c_half{matrix_c_half.data()};

    fill_random_float_values(h_matrix_a_float, matrix_a_float.size(),
                             random_engine, range);
    fill_random_float_values(h_matrix_b_float, matrix_b_float.size(),
                             random_engine, range);
    float2half(h_matrix_a_half, h_matrix_a_float, matrix_a_float.size());
    float2half(h_matrix_b_half, h_matrix_b_float, matrix_b_float.size());

    half *d_matrix_a_half, *d_matrix_b_half;
    float* d_matrix_c_float;
    half* d_matrix_c_half;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half,
                                matrix_size_m * matrix_size_k * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half,
                                matrix_size_k * matrix_size_n * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float,
                                matrix_size_m * matrix_size_n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_half,
                                matrix_size_m * matrix_size_n * sizeof(half)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_a_half, h_matrix_a_half,
                                matrix_a_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix_b_half, h_matrix_b_half,
                                matrix_b_float.size() * sizeof(__half),
                                cudaMemcpyHostToDevice));

    for (bool is_A_transpose : {true, false})
    {
        for (bool is_B_transpose : {true, false})
        {
            print_test_header(is_A_transpose, is_B_transpose);

            // beta=0 in launch_wmma_mm's epilogue means C's *prior* value
            // should not matter -- but 0 * NaN/Inf is itself NaN in IEEE
            // arithmetic, and this loop reuses the same C buffers across
            // transpose combos, so a prior combo's overflowed output would
            // otherwise leak into the next combo's "prior C" read. Zero
            // both C buffers before every launch so each combo's result is
            // attributable only to *that* combo's own accumulation.
            CHECK_CUDA_ERROR(cudaMemset(d_matrix_c_float, 0,
                                        matrix_c_float.size() * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_matrix_c_half, 0,
                                        matrix_c_half.size() * sizeof(half)));

            // Control: float accumulator (same as leimao-mm-128x128.cu).
            launch_wmma_mm(d_matrix_a_half, d_matrix_b_half, d_matrix_c_float,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_float, d_matrix_c_float,
                                        matrix_c_float.size() * sizeof(float),
                                        cudaMemcpyDeviceToHost));
            std::cout << "  float accumulator: "
                      << count_non_finite_float(h_matrix_c_float,
                                                matrix_c_float.size())
                      << std::endl;

            // Experiment: half accumulator.
            launch_wmma_mm(d_matrix_a_half, d_matrix_b_half, d_matrix_c_half,
                           matrix_size_m, matrix_size_n, matrix_size_k,
                           is_A_transpose, is_B_transpose, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            CHECK_CUDA_ERROR(cudaMemcpy(h_matrix_c_half, d_matrix_c_half,
                                        matrix_c_half.size() * sizeof(__half),
                                        cudaMemcpyDeviceToHost));
            std::cout << "  half accumulator:  "
                      << count_non_finite_half(h_matrix_c_half,
                                               matrix_c_half.size())
                      << std::endl;
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_half));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
