// Blocked CUDA LU solver whose trailing update runs on TENSOR CORES (WMMA).
//
// Derived from lu_solve_gpu.cu (same inputs, same partial pivoting, same
// forward/backward substitution in FP64). The difference is the structure of
// the factorization: instead of a rank-1 update per pivot column, we use the
// standard right-looking *blocked* LU with block size NB = 16:
//
//   for each block column kb (16 columns):
//     1. panel factorization of columns [kb, kb+16) in FP64 (pivot search, row
//        swap, multipliers, elimination restricted to the panel columns)
//     2. U12 = L11^{-1} A12            (FP64 triangular solve, 16 rows)
//     3. A22 -= L21 * U12              <-- tensor cores (WMMA m16n16k16)
//
// Step 3 is a matrix-matrix product. L21 and U12 are rounded from double to
// the tensor-core input type (FP16 or BF16), multiplied with
// wmma::mma_sync (which compiles to HMMA.16816.* on sm_80/86), and the
// product tile is subtracted from A22 in FP64. The matrix itself, the panel,
// the triangular solves and the substitutions all stay in FP64; only the
// product L21*U12 is computed in reduced precision. This is the usual
// mixed-precision LU (as in HPL-AI), without the iterative refinement.
//
// Modes (second command-line argument):
//   fp16acc32  FP16 inputs, FP32 accumulator   -> HMMA.16816.F32
//   fp16acc16  FP16 inputs, FP16 accumulator   -> HMMA.16816.F16
//   bf16acc32  BF16 inputs, FP32 accumulator   -> HMMA.16816.F32.BF16
//
// Matrix argument: a CSV file, or "rand:N[:scale[:seed]]" for an N x N
// pseudo-random, diagonally weighted matrix with entries multiplied by
// `scale`. Large scales push the double->FP16 rounding of L21/U12 beyond
// 65504, which is what nixnan is expected to flag around the HMMAs.
//
// Build: make lu_solve_gpu_tc     (needs sm_80+; ARCH=sm_86 by default)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace nvcuda;

#define IDX(i, j, m) ((i) * (m) + (j))
#define NB 16 // block size == WMMA tile edge

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "   \
                 << cudaGetErrorString(err__) << endl;                        \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------
// Host helpers
// -----------------------------------------------------------

static vector<vector<double>> load_matrix_from_csv(const string &filename)
{
    vector<vector<double>> matrix;
    ifstream file(filename);
    string line;
    if (!file.is_open())
    {
        cerr << "Error: Could not open file '" << filename << "'" << endl;
        return matrix;
    }
    while (getline(file, line))
    {
        vector<double> row;
        stringstream ss(line);
        string value;
        while (getline(ss, value, ','))
            row.push_back(stod(value));
        matrix.push_back(row);
    }
    return matrix;
}

// "rand:N[:scale[:seed]]" -> N x N matrix, entries in (-1,1)*scale, diagonal
// boosted by N*scale/4 so it is nonsingular but still needs pivoting.
static vector<vector<double>> random_matrix(const string &spec)
{
    int n = 0;
    double scale = 1.0;
    unsigned seed = 12345;
    sscanf(spec.c_str(), "rand:%d:%lf:%u", &n, &scale, &seed);
    vector<vector<double>> A(n, vector<double>(n));
    unsigned s = seed;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            s = s * 1664525u + 1013904223u;
            double u = ((s >> 8) & 0xFFFF) / 32768.0 - 1.0;
            A[i][j] = u * scale;
            if (i == j && (i % 3 == 0))
                A[i][j] += (n * scale) / 4.0;
        }
    return A;
}

static void print_matrix(const vector<vector<double>> &matrix, size_t max_n = 24)
{
    if (matrix.size() > max_n)
    {
        cout << "(matrix " << matrix.size() << "x" << matrix[0].size()
             << " not printed)" << endl << endl;
        return;
    }
    for (const auto &row : matrix)
    {
        for (double v : row)
            cout << v << "\t";
        cout << endl;
    }
    cout << endl;
}

static vector<double> multiply_matrix_vector(const vector<vector<double>> &A, const vector<double> &x)
{
    int rows_A = A.size();
    int cols_A = rows_A ? A[0].size() : 0;
    vector<double> result(rows_A, 0.0);
    for (int i = 0; i < rows_A; ++i)
        for (int j = 0; j < cols_A; ++j)
            result[i] += A[i][j] * x[j];
    return result;
}

static vector<double> subtract_vectors(const vector<double> &x, const vector<double> &y)
{
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        result[i] = x[i] - y[i];
    return result;
}

static double l2_norm(const vector<double> &v)
{
    double sum_of_squares = 0.0;
    for (double val : v)
        sum_of_squares += val * val;
    return sqrt(sum_of_squares);
}

// -----------------------------------------------------------
// FP64 kernels (panel, triangular solve, substitutions)
// -----------------------------------------------------------

__global__ void init_identity_kernel(double *M, int m)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m && j < m)
        M[IDX(i, j, m)] = (i == j) ? 1.0 : 0.0;
}

__global__ void find_pivot_kernel(const double *U, int m, int k, int *pivot_row)
{
    extern __shared__ unsigned char smem[];
    double *smax = reinterpret_cast<double *>(smem);
    int *sidx = reinterpret_cast<int *>(smax + blockDim.x);

    int tid = threadIdx.x;
    int row = k + tid;

    double val = -1.0;
    int idx = k;
    if (row < m)
    {
        val = fabs(U[IDX(row, k, m)]);
        idx = row;
    }
    smax[tid] = val;
    sidx[tid] = idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && smax[tid + stride] > smax[tid])
        {
            smax[tid] = smax[tid + stride];
            sidx[tid] = sidx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        *pivot_row = sidx[0];
}

__global__ void swap_rows_kernel(double *U, double *L, double *P, int m, int k,
                                 const int *pivot_row_ptr)
{
    int pr = *pivot_row_ptr;
    if (pr == k)
        return;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m)
        return;

    double tmp = U[IDX(k, j, m)];
    U[IDX(k, j, m)] = U[IDX(pr, j, m)];
    U[IDX(pr, j, m)] = tmp;

    tmp = P[IDX(k, j, m)];
    P[IDX(k, j, m)] = P[IDX(pr, j, m)];
    P[IDX(pr, j, m)] = tmp;

    if (j < k)
    {
        tmp = L[IDX(k, j, m)];
        L[IDX(k, j, m)] = L[IDX(pr, j, m)];
        L[IDX(pr, j, m)] = tmp;
    }
}

__global__ void compute_multipliers_kernel(const double *U, double *L, int m, int k)
{
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m)
        L[IDX(j, k, m)] = U[IDX(j, k, m)] / U[IDX(k, k, m)];
}

// Panel elimination: update only columns [k, lend) (the current panel).
__global__ void eliminate_panel_kernel(double *U, const double *L, int m, int k, int lend)
{
    int j = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int l = k + blockIdx.x * blockDim.x + threadIdx.x;
    if (j < m && l < lend)
    {
        double val = U[IDX(j, l, m)] - L[IDX(j, k, m)] * U[IDX(k, l, m)];
        if (fabs(val) < (10.0 * DBL_EPSILON))
            val = 0.0;
        U[IDX(j, l, m)] = val;
    }
}

// U12 = L11^{-1} A12, in place: rows [kb, kb+NB), columns [kb+NB, m).
// One thread per column; sequential over the NB rows.
__global__ void u12_trsm_kernel(double *U, const double *L, int m, int kb)
{
    int c = kb + NB + blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= m)
        return;
    for (int r = 1; r < NB; ++r)
    {
        double sum = 0.0;
        for (int s = 0; s < r; ++s)
            sum += L[IDX(kb + r, kb + s, m)] * U[IDX(kb + s, c, m)];
        U[IDX(kb + r, c, m)] -= sum;
    }
}

__global__ void apply_permutation_kernel(const double *P, const double *b, double *bp, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        double sum = 0.0;
        for (int j = 0; j < m; ++j)
            sum += P[IDX(i, j, m)] * b[j];
        bp[i] = sum;
    }
}

__global__ void forward_substitution_kernel(const double *L, const double *bp, double *y, int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < i; ++j)
            sum += L[IDX(i, j, m)] * y[j];
        y[i] = bp[i] - sum;
    }
}

__global__ void backward_substitution_kernel(const double *U, const double *y, double *x, int m)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int i = m - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for (int j = i + 1; j < m; ++j)
            sum += U[IDX(i, j, m)] * x[j];
        x[i] = (y[i] - sum) / U[IDX(i, i, m)];
    }
}

// -----------------------------------------------------------
// Tensor-core trailing update  A22 -= L21 * U12
// -----------------------------------------------------------

// Round double -> tensor-core input type (overflow becomes +-Inf).
template <typename T> __device__ T to_tc(double v);
template <> __device__ __half to_tc<__half>(double v) { return __double2half(v); }
template <> __device__ __nv_bfloat16 to_tc<__nv_bfloat16>(double v) { return __double2bfloat16(v); }

// Lh: nrows_pad x NB (row-major, ld = NB): rows of L21, columns kb..kb+NB.
// Uh: NB x ncols_pad  (row-major, ld = ncols_pad): columns of U12.
// Padding is filled with exact zeros.
template <typename T>
__global__ void pack_tc_operands(const double *L, const double *U, T *Lh, T *Uh,
                                 int m, int kb, int nrows_pad, int ncols_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nL = nrows_pad * NB;
    int nU = NB * ncols_pad;
    if (idx < nL)
    {
        int r = idx / NB, c = idx % NB;
        int gr = kb + NB + r;
        double v = (gr < m) ? L[IDX(gr, kb + c, m)] : 0.0;
        Lh[idx] = to_tc<T>(v);
    }
    else if (idx < nL + nU)
    {
        int u = idx - nL;
        int r = u / ncols_pad, c = u % ncols_pad;
        int gc = kb + NB + c;
        double v = (gc < m) ? U[IDX(kb + r, gc, m)] : 0.0;
        Uh[u] = to_tc<T>(v);
    }
}

// One warp per 16x16 output tile. One m16n16k16 mma_sync (= two HMMA.16816).
// AccT = float  -> HMMA.16816.F32 / .F32.BF16 ; AccT = __half -> HMMA.16816.F16
template <typename T, typename AccT>
__global__ void tc_gemm_tile(const T *Lh, const T *Uh, AccT *Pt, int ncols_pad)
{
    int ti = blockIdx.y, tj = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, AccT> c;
    wmma::fill_fragment(c, (AccT)0.0f);
    wmma::load_matrix_sync(a, Lh + ti * 16 * NB, NB);
    wmma::load_matrix_sync(b, Uh + tj * 16, ncols_pad);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(Pt + (size_t)ti * 16 * ncols_pad + tj * 16, c, ncols_pad,
                            wmma::mem_row_major);
}

__device__ inline double acc_to_double(float v) { return (double)v; }
__device__ inline double acc_to_double(__half v) { return (double)__half2float(v); }

// A22[j][l] -= P[j][l], same epsilon snap as the FP64 code.
template <typename AccT>
__global__ void apply_update_kernel(double *U, const AccT *Pt, int m, int kb, int ncols_pad)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int gr = kb + NB + r, gc = kb + NB + c;
    if (gr < m && gc < m)
    {
        double val = U[IDX(gr, gc, m)] - acc_to_double(Pt[(size_t)r * ncols_pad + c]);
        if (fabs(val) < (10.0 * DBL_EPSILON))
            val = 0.0;
        U[IDX(gr, gc, m)] = val;
    }
}

enum Mode { FP16ACC32, FP16ACC16, BF16ACC32 };

static const char *mode_name(Mode md)
{
    return md == FP16ACC32 ? "fp16acc32" : md == FP16ACC16 ? "fp16acc16" : "bf16acc32";
}

template <typename T, typename AccT>
static void trailing_update(double *d_U, const double *d_L, int m, int kb)
{
    int rem = m - kb - NB; // rows/cols left in A22
    int nt = (rem + 15) / 16;
    int pad = nt * 16;

    T *Lh, *Uh;
    AccT *Pt;
    CUDA_CHECK(cudaMalloc(&Lh, (size_t)pad * NB * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&Uh, (size_t)NB * pad * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&Pt, (size_t)pad * pad * sizeof(AccT)));

    int total = pad * NB + NB * pad;
    pack_tc_operands<T><<<(total + 255) / 256, 256>>>(d_L, d_U, Lh, Uh, m, kb, pad, pad);
    CUDA_CHECK(cudaGetLastError());

    tc_gemm_tile<T, AccT><<<dim3(nt, nt), 32>>>(Lh, Uh, Pt, pad);
    CUDA_CHECK(cudaGetLastError());

    dim3 ub(16, 16);
    dim3 ug((rem + 15) / 16, (rem + 15) / 16);
    apply_update_kernel<AccT><<<ug, ub>>>(d_U, Pt, m, kb, pad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(Lh);
    cudaFree(Uh);
    cudaFree(Pt);
}

// -----------------------------------------------------------
// Host driver
// -----------------------------------------------------------

static vector<double> flatten(const vector<vector<double>> &M)
{
    int m = M.size();
    vector<double> flat(m * m);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            flat[IDX(i, j, m)] = M[i][j];
    return flat;
}

static int next_pow2(int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

static vector<double> solve_system_with_LU_gpu_tc(const vector<vector<double>> &A,
                                                  const vector<double> &b, Mode mode)
{
    int m = A.size();
    size_t mat_bytes = static_cast<size_t>(m) * m * sizeof(double);
    size_t vec_bytes = static_cast<size_t>(m) * sizeof(double);

    vector<double> h_A = flatten(A);

    double *d_U, *d_L, *d_P, *d_b, *d_bp, *d_y, *d_x;
    int *d_pivot_row;
    CUDA_CHECK(cudaMalloc(&d_U, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_L, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_P, mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_bp, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_pivot_row, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_U, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vec_bytes, cudaMemcpyHostToDevice));

    dim3 idBlock(16, 16);
    dim3 idGrid((m + idBlock.x - 1) / idBlock.x, (m + idBlock.y - 1) / idBlock.y);
    init_identity_kernel<<<idGrid, idBlock>>>(d_L, m);
    init_identity_kernel<<<idGrid, idBlock>>>(d_P, m);
    CUDA_CHECK(cudaGetLastError());

    for (int kb = 0; kb < m - 1; kb += NB)
    {
        int kend = min(kb + NB, m);

        // 1. FP64 panel factorization of columns [kb, kend)
        for (int k = kb; k < kend && k < m - 1; ++k)
        {
            int pivotThreads = next_pow2(m - k);
            size_t shmem = pivotThreads * (sizeof(double) + sizeof(int));
            find_pivot_kernel<<<1, pivotThreads, shmem>>>(d_U, m, k, d_pivot_row);

            int swapBlocks = (m + 255) / 256;
            swap_rows_kernel<<<swapBlocks, 256>>>(d_U, d_L, d_P, m, k, d_pivot_row);

            int mulBlocks = (m - k - 1 + 255) / 256;
            if (mulBlocks > 0)
                compute_multipliers_kernel<<<mulBlocks, 256>>>(d_U, d_L, m, k);

            dim3 eb(16, 16);
            dim3 eg((kend - k + 15) / 16, (m - k - 1 + 15) / 16);
            if (eg.x > 0 && eg.y > 0)
                eliminate_panel_kernel<<<eg, eb>>>(d_U, d_L, m, k, kend);
            CUDA_CHECK(cudaGetLastError());
        }

        if (kend - kb == NB && kb + NB < m)
        {
            // 2. U12 = L11^{-1} A12 (FP64)
            int cols = m - kb - NB;
            u12_trsm_kernel<<<(cols + 127) / 128, 128>>>(d_U, d_L, m, kb);
            CUDA_CHECK(cudaGetLastError());

            // 3. A22 -= L21 * U12 on tensor cores
            switch (mode)
            {
            case FP16ACC32: trailing_update<__half, float>(d_U, d_L, m, kb); break;
            case FP16ACC16: trailing_update<__half, __half>(d_U, d_L, m, kb); break;
            case BF16ACC32: trailing_update<__nv_bfloat16, float>(d_U, d_L, m, kb); break;
            }
        }
    }

    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    apply_permutation_kernel<<<blocks, threads>>>(d_P, d_b, d_bp, m);
    forward_substitution_kernel<<<1, 1>>>(d_L, d_bp, d_y, m);
    backward_substitution_kernel<<<1, 1>>>(d_U, d_y, d_x, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> x(m);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, vec_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_U);
    cudaFree(d_L);
    cudaFree(d_P);
    cudaFree(d_b);
    cudaFree(d_bp);
    cudaFree(d_y);
    cudaFree(d_x);
    cudaFree(d_pivot_row);
    return x;
}

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        cerr << "Usage: " << argv[0]
             << " <matrix.csv | rand:N[:scale[:seed]]> [fp16acc32|fp16acc16|bf16acc32]" << endl;
        return 1;
    }
    string spec = argv[1];
    string mname = argc == 3 ? argv[2] : "fp16acc32";
    Mode mode;
    if (mname == "fp16acc32") mode = FP16ACC32;
    else if (mname == "fp16acc16") mode = FP16ACC16;
    else if (mname == "bf16acc32") mode = BF16ACC32;
    else { cerr << "unknown mode " << mname << endl; return 1; }

    cout << "Loading matrix A (" << spec << "), tensor-core mode " << mode_name(mode) << ":" << endl;
    vector<vector<double>> A =
        spec.rfind("rand:", 0) == 0 ? random_matrix(spec) : load_matrix_from_csv(spec);
    if (A.empty())
        return 1;
    print_matrix(A);
    cout << "-----------------------------------------------------------" << endl;

    vector<double> b(A.size(), 1.0);
    auto x = solve_system_with_LU_gpu_tc(A, b, mode);

    cout << "Solution x:" << endl;
    for (size_t i = 0; i < x.size() && i < 24; ++i)
        cout << "x[" << i << "]: " << x[i] << endl;
    if (x.size() > 24)
        cout << "... (" << x.size() << " entries)" << endl;
    cout << "-----------------------------------------------------------" << endl;

    auto a = multiply_matrix_vector(A, x);
    auto residual = subtract_vectors(a, b);
    auto norm = l2_norm(residual);
    cout << "Residual norm ||Ax - b||: " << norm << endl;
    return 0;
}
