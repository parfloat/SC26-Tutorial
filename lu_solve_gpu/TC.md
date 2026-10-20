# lu_solve_gpu_tc: blocked LU with a tensor-core trailing update

`lu_solve_gpu_tc.cu` keeps the FP64 pivoting, panel, triangular solves and substitutions of
`lu_solve_gpu.cu`, but uses a blocked right-looking LU (NB=16). The trailing update
`A22 -= L21*U12` is a WMMA m16n16k16 product (`mma_sync` = 2 x HMMA.16816):
L21, U12 are rounded double->FP16/BF16, multiplied on the tensor cores, and the product is subtracted
in FP64. No iterative refinement.

    make lu_solve_gpu_tc
    ./lu_solve_gpu_tc <matrix.csv | rand:N[:scale[:seed]]> [fp16acc32|fp16acc16|bf16acc32]

SASS (sm_86): HMMA.16816.F32 / .F16 / .F32.BF16, 2 each, one kernel `tc_gemm_tile` (template).

## Results (RTX 3090)
| input | fp16acc32 | fp16acc16 | bf16acc32 | FP64 original |
|---|---|---|---|---|
| matrix.csv residual | 1.1e-4 | 2.0e-4 | 4.4e-4 | 9.7e-16 |
| bad_matrix.csv | NaN | NaN | NaN | NaN |
| rand:64:1e5 | NaN | NaN | 0.17 | n/a |

Residuals grow with n (rand:256:1: 0.75 fp16, 5.4 bf16) because there is no refinement; I did not
investigate the conditioning of the random matrices.

## nixnan (`env -u LD_LIBRARY_PATH`, LINE_INFO=1 PRINT_ILL_INSTR=1; logs in logs/tc/)
- matrix.csv (3 modes) and rand:64:1e5 bf16acc32: no exceptions.
- rand:64:1e5 fp16acc32/fp16acc16: first report is `[infinity]` in operand 2 (B = U12, type f16) of
  HMMA.16816.F32 / .F16; then NaN in operand 0 (0 x inf, since the padding is exact zeros), NaN on the
  STG that stores the tile, and later FP64 NaN/Inf in the DFMA/DADD update. The Inf is created by
  `F2F.F16.F64` (double->half in `pack_tc_operands`), which nixnan has no table entry for, so it is
  seen only as an input of the HMMA.
- LINE_INFO did not resolve any source lines for these kernels (0 matches for lu_solve_gpu_tc.cu).
