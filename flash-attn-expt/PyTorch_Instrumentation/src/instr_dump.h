// INSTRUMENTATION (PyTorch_Instrumentation): layout of the dump buffers.
// Everything is float. Indices refer to the hdim-32 FP16 kernels TinyGPT uses
// (flash_fwd_kernel tile 128 x 128, head dim 32).
#pragma once

namespace instr_flash { namespace dbg {

// --- flash_fwd_kernel (non-split): one block of HEAD floats per head, batch 0, m_block 0,
//     first K/V block processed (n_block = n_block_max - 1). Row-major (row, col).
//     Q, K, V hold raw FP16 bit patterns (0..65535); the other tiles hold values.
constexpr int M = 128, N = 128, D = 32;
constexpr int Q      = 0;                // sQ  smem tile   M x D  (rows >= seqlen_q: never loaded)
constexpr int K      = Q + M * D;        // sK  smem tile   N x D  (rows >= seqlen_k: never loaded)
constexpr int V      = K + N * D;        // sV  smem tile   N x D  (rows >= seqlen_k: cleared to 0)
constexpr int S_RAW  = V + N * D;        // acc_s = Q K^T   M x N  before masking
constexpr int S_MASK = S_RAW + M * N;    // acc_s           M x N  after apply_mask
constexpr int P      = S_MASK + M * N;   // rP (FP16)       M x N  after softmax_rescale_o
constexpr int O      = P + M * N;        // acc_o           M x D  after normalize_softmax_lse
constexpr int ROW    = O + M * D;        // per row: row_max, row_sum, lse, guard (1 = sum==0, 2 = sum NaN)
constexpr int ROWF   = 4;
constexpr int HEAD   = ROW + M * ROWF;

// --- flash_fwd_splitkv_kernel: record index ((split * b + bidb) * h + bidh) * R + row,
//     R = seqlen_q rounded up to the split kernel's kBlockM (64).
// fields: written, split, head, row, valid (row < seqlen_q), row_max, row_sum, lse,
//         guard (0 none, 1 sum==0, 2 sum!=sum, 3 split had no key block: early exit writes -inf)
constexpr int SPLITF = 12;

// --- flash_fwd_splitkv_combine_kernel: record index = combine row (b * h * seqlen_q flattened,
//     including the padding rows of the last 16-row combine block).
// fields: written, row, valid, lse_split0, lse_split1, lse_max (after -inf -> 0 clamp), lse_sum,
//         sum_is_zero, sum_is_nan, guard_fired, logf(lse_sum)+lse_max (what the guard replaces),
//         lse_logsum (what is stored), non-finite O values in the combined output row
constexpr int COMBF = 16;

}}  // namespace instr_flash::dbg
