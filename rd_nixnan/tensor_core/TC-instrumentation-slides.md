# Tensor-core instrumentation: annotated slide list (7 slides)

Deck: `latex/tc-instrumentation-steps.tex/.pdf`. Raw data: `latex/data/verbose-{fp16acc16,fp16acc32,bf16acc32}.log`
(each from `TOOL_VERBOSE=1 LINE_INFO=1 PRINT_ILL_INSTR=1`, `env -u LD_LIBRARY_PATH`, `LD_PRELOAD=~/repos/nixnan/nixnan.so ./rd_tc <mode>`).

1. **What the GPU runs.** HMMA.16816.F16 / .F32 / .F32.BF16, operand order D,A,B,C; 16 HMMAs per kernel; first of each chain has C=RZ. Say: NixNan sees SASS, not `mma_sync`.
2. **The table.** `instructions.json` {type,count} per operand; count = values, f16/bf16 packed 2 per register, so F16 -> 2/4/2/2 regs, F32 -> 4/4/2/4. No entry (F2FP.PACK) = not instrumented.
3. **Verbose log.** AFTER call has 1+regs(D) args (3 for F16, 5 for F32); BEFORE has 3 reginfo words + regs(A,B,C) (11 / 13). Identical for all 16 HMMAs.
4. **Compact encoding.** One 32-bit reginfo per operand (count:6,type:4,h0,h1,div0,operand:2,num_regs:4,f64high). Words listed per variant, computed from the real struct (not dumped from the live tool).
5. **Registers picked up.** Consecutive R_n.. per operand; example third HMMA of each kernel. D overlaps B (and C for F32) -> that is why there are two call sites (BEFORE reads inputs, AFTER reads result).
6. **Device side.** `nixnan_check_regs`: va_arg walk, half2/bf162/float classify, OR + warp shuffle, lowest lane does atomicAdd record; channel push only on first occurrence.
7. **Result.** fp16acc16: inf in operand 0 of HMMA.16816.F16 (step 230), NaN after (231). acc32 modes: inf only seen as input of next HMMA (born at F2FP, no table entry). Tile-internal products are invisible.

## Verified vs inferred
- Verified (live run): instrumented instruction text, "N registers" counts, regs per operand, 16 HMMAs/kernel, report lines.
- Computed (struct compiled with g++): reginfo words.
- Inferred from code: RZ sent as index 255 reading 0; `.reuse` has no effect on the read.
