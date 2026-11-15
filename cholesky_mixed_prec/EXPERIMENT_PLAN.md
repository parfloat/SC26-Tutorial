# Experimental plan: Cholesky on a badly conditioned matrix, mixed precision, under nixnan

Status: **E0 and E1 are implemented and run (Section 9); E2–E7 are still plan only.** The numbers marked "checked" in Sections 3 and 6 come from a quick CPU check (NumPy/SciPy, LAPACK) made while writing the plan.
"checked" come from a quick CPU check (NumPy/SciPy, LAPACK) made while writing the plan.

Source: `cg-and-cholesky-nixnan.pdf` in this directory (6 pages). It is a ChatGPT transcript, not a reference: it
explains CG, asks whether CG suits tensor cores, then explains Cholesky and ends with a proposal for a nixnan
tutorial example. This plan keeps that proposal's idea and tests each claim instead of assuming it.

## 1. What the tutorial should show

> A **straight Cholesky** of a badly conditioned SPD matrix, with the precision lowered without thinking about the
> condition number, produces floating-point exceptions that nixnan can locate. With the precision lowered carefully, the same
> matrix factors cleanly.

The PDF's framing, which this plan adopts: the failure is **precision-induced loss of positive definiteness**, not
"a big condition number" by itself. Rounding destroys the small positive Schur-complement pivot that Cholesky needs
(`L_jj = sqrt(S_jj)`). A negative `S_jj` gives `sqrt` of a negative (NaN); a zero pivot gives a division by zero, then NaN.

A second lesson, also from the PDF and matching the LU clamp story (`../lu_solve_gpu/CLAMP_STORY.md`): bad
conditioning does **not** guarantee an exception. Often the factor is finite and inaccurate and nothing is raised.
The tutorial should show both outcomes, so the plan includes a randomised sweep (E5).

## 2. Run-time budget (a design constraint)

- Each experiment run: **under 1 minute natively, under 2 minutes with nixnan.**
- Evidence for what that allows (from this repo, not new measurements):
  - nixnan instruments each kernel once, at its first launch: about 0.28–0.33 s per distinct kernel
    (`../lu_solve_gpu/LU_Slides.pdf`, TIME_KERNELS slide). A solver with about 10 distinct kernels costs a few seconds.
  - Library-heavy binaries are far worse: the cuSOLVER-based `pytorch_slogdet` run was killed at 120 s during
    instrumentation, and `mpir_gpu` is run with `iter-autonixnan -t 90` (`../run_all_examples.sh`).
- Consequences:
  1. Use **custom, single-file CUDA kernels** (POTRF tile, TRSM, trailing update, substitution) for every run that goes
     under nixnan. cuBLAS/cuSOLVER only for native reference runs, if at all.
  2. Keep the **number of distinct kernels small** (target: 10 or fewer) and the matrix small: **n = 256 primary, n = 64
     for debugging.** n = 1024 only if the budget allows after E0.
  3. **E0 measures the budget first.** If a configuration exceeds it, shrink n or the kernel count before going on, and use
     `FUNCTION_WHITELIST` / `ksel-autonixnan` to instrument only the kernels that matter.

## 3. The PDF's claims, and how each is handled

| Claim in the PDF | Status | Plan |
|---|---|---|
| `A_δ = 11ᵀ + δI` is SPD for every δ > 0, with κ₂ = (n+δ)/δ ≈ n/δ | **Checked** (n=256, δ=2⁻¹²: κ = 1.049e6 both ways) | Use as the deterministic matrix |
| In FP16, TF32-input and BF16, `1+δ` rounds to 1 when δ is below 2⁻¹¹ (FP16, TF32) or 2⁻⁸ (BF16), so the stored matrix becomes the rank-one `11ᵀ` | **Checked** for FP16 (all 65,536 entries equal 1) and BF16 (`bf16(1+2⁻¹²) = 1`) | Rounding emulation in E2; real hardware in E3/E4 |
| FP32 "remains safe" because 1+δ rounds only below 2⁻²⁴ | **Misleading.** 2⁻²⁴ is where *storage* rounds. FP32 *factorization* fails much earlier from cancellation in the Schur complement: LAPACK in FP32 factors δ=2⁻¹⁴ (κ≈4e6) and reports "not positive definite" at δ=2⁻¹⁶ (κ≈2e7); 2⁻¹⁵ not tried | E2 finds the FP32 threshold on the GPU; the demo δ must sit **between** the FP32 and FP16 thresholds (δ=2⁻¹² does: FP32 factors it with ‖A−LLᵀ‖/‖A‖ = 4.3e-7, checked) |
| An unblocked Cholesky of the rounded matrix has a zero second pivot, then 0/0 | **Not yet run.** LAPACK reports "not PD" instead of producing a NaN, so the NaN needs our own kernel that does not check | E3 |
| nixnan "can identify the invalid sqrt, divide-by-zero and resulting NaN" | **Partly known.** nixnan counts NaN/Inf/div-by-zero at SASS level and has `fp32/sqrt_single`, `div_single` tests in `../../nixnan/test/`. Whether the sqrt-of-negative path is reported in the form this kernel compiles to is **unverified** | E3, with `PRINT_ILL_INSTR` |
| nixnan "can watch HMMA instructions" for FP16/BF16/TF32 inputs with FP32 accumulate | **Known** from nixnan's `test/mma/` (HMMA.16816.F32, .F32.BF16, HMMA.1684.F32.TF32 covered). A blocked Cholesky actually compiling to HMMA is **unverified** | E4 (check SASS with `cuobjdump -sass`) |
| FP16/BF16 conversions (`F2F`, `F2FP`) feed the exceptions | **Known gap:** nixnan does not instrument those opcodes (`../../nixnan/test/README.md`, coverage). The exception will show up on the FP32 operation *after* the conversion, not on the conversion | Say so in the slides; E3 confirms |
| cuSOLVER can do the FP16/BF16 factorization | **Doubtful.** I believe `cusolverDnXpotrf` supports FP32/FP64 only; not checked | Do not rely on it; custom kernels (also meets the time budget) |

## 4. Matrices

1. **`A_δ = 11ᵀ + δI`**, n = 256, δ ∈ {2⁻⁸ … 2⁻²⁴}. Deterministic, κ ≈ n/δ. Primary matrix. Fixed right-hand side `b = A·x_true`
   with `x_true = 1`, so errors are computable.
2. **`A = QΛQᵀ`** with geometric spectrum, κ ∈ {1e2, 1e3, 1e4, 1e5, 1e6, 1e7}, seeds 0–4. Same recipe as
   `../mpir_gpu/Code2.cu` `make_problem` (QR of a Gaussian, `σ_j = κ^(−j/(n−1))`), so results are comparable. Used in E5.

Both are generated by one small script (`gen_chol_matrices.py`), written to CSV like `../lu_solve_gpu/gen_matrix_n.py`, with
condition numbers computed in 80-digit arithmetic (double-precision SVD cannot resolve κ above ~1e16; here the
values are far lower, but the same helper can be reused).

## 5. Precision configurations

| name | stored `A`/`L` | arithmetic of POTRF / TRSM | trailing update (SYRK/GEMM) |
|---|---|---|---|
| `F64` | FP64 | FP64 | FP64 (reference) |
| `F32` | FP32 | FP32 | FP32 |
| `naive-F16`, `naive-BF16` | FP16 / BF16 | FP32 arithmetic on rounded input (E3), and a true half-arithmetic variant (E3b) | same as the factorization |
| `mixed-F16`, `mixed-BF16`, `mixed-TF32` | FP32 | FP32 | tensor cores: FP16/BF16/TF32 inputs, FP32 accumulate (E4) |

`mixed-*` is the "better mixed precision" row of the PDF: low precision only where the work is (the O(n³) trailing update),
FP32 for the diagonal tile, the panel TRSM and the pivots.

## 6. Experiments

Common outputs: one `RESULT` line per run in the style of `../mpir_gpu/Code2.cu`
(`config, n, delta|kappa, seed, status, factorization error ‖A−LLᵀ‖/‖A‖, solve residual ‖b−Ax‖/‖b‖, first non-positive pivot index,
time`), where `status ∈ {ok, inaccurate, breakdown, nan}`. `breakdown` means the code itself saw a pivot ≤ 0; `nan` means
non-finite values appeared **without** the code noticing.

### E0 — Budget and tooling smoke test
- **Question:** can the planned kernel set run inside the time budget, natively and under nixnan?
- **Do:** write the custom unblocked and blocked Cholesky kernels for FP32 at n=64 and 256; run each natively and with plain
  `LD_PRELOAD=nixnan.so`; record wall times and the distinct-kernel count (`TIME_KERNELS=1` shows first-launch cost).
- **Pass:** native < 1 min, nixnan < 2 min for n = 256. **If not:** reduce n or kernel count (Section 2) before E1.
- Also check that `../mpir_gpu` builds for this GPU (sm_86); its README says it was written without being compiled.

### E1 — Reference: the matrix is fine at full precision
- **Setup:** `A_δ`, n=256, δ=2⁻¹²; `F64` and `F32`.
- **Expect (checked on CPU):** both factor; FP32 error about 4e-7; no exceptions under nixnan.
- **Why:** establishes "the matrix is valid and the algorithm is right", so every later failure is due to precision.

### E2 — Where does each precision break? (δ and κ thresholds)
- **Question:** for each storage/arithmetic format, at what δ (κ ≈ n/δ) does the factorization stop working, and **why** (storage rounding or cancellation)?
- **Do:** first a fast host-side emulation (round `A_δ` to the format, then factor in FP32/FP64) over δ = 2⁻⁸ … 2⁻²⁴, to choose
  points; then confirm the chosen points on the GPU. Classify each point `ok`/`inaccurate`/`breakdown`/`nan`.
- **Predictions to test:** FP16 and TF32-input storage fail below δ≈2⁻¹¹; BF16 below ≈2⁻⁸; FP32 factorization fails at some δ between
  2⁻¹⁴ (works) and 2⁻¹⁶ (fails). The result decides the **demo δ**, which must be factorable in FP32 and not in FP16/BF16 (δ=2⁻¹² is the candidate).

### E3 — Naive reduced precision: the vivid exception
- **Setup:** `A_δ`, δ=2⁻¹², store in FP16 and in BF16, factor with the custom **unblocked** kernel that does not check pivots.
- **Expect:** stored matrix is exactly `11ᵀ`; the second pivot is 0; the division by it gives Inf/NaN (0/0) and the NaNs spread through `L`, `y`, `x`.
- **nixnan:** run with defaults, then with `PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1` and `LINE_INFO=1`. Record: which kernel and source line reports first,
  which instruction (`MUFU.RCP`/`RSQ`, `FMUL`…), counts per precision section, and the propagation chain (as on the LU "Following the first exception" slide).
- **E3b:** repeat with arithmetic in true FP16/BF16 (`__hfma`/`half2` or `__nv_bfloat16`), to see exceptions in the FP16/BF16 sections of the report.
  FP32-arithmetic runs report in the FP32 section, since conversions are not instrumented.
- **Variant:** δ slightly **above** the format's threshold, where the matrix stays SPD but the pivot is tiny: expect finite-but-wrong output and no exception (the "silent" case).

### E4 — Blocked mixed precision with tensor cores
- **Setup:** blocked right-looking Cholesky, block size 16–32; FP32 POTRF on the diagonal tile and FP32 TRSM on the panel; trailing update
  `A_RR ← A_RR − L_Rk L_Rkᵀ` through WMMA with FP16 / BF16 / TF32 inputs and FP32 accumulate. Reuse the WMMA tile from
  `../lu_solve_gpu/lu_solve_gpu_tc.cu` (`TC.md`) and the tests in `../../nixnan/test/mma/`.
- **Check first:** SASS contains `HMMA.16816.F32` (`.F32.BF16`, `HMMA.1684.F32.TF32`), and the matrix entries stay in range (FP16 maximum 65504).
- **Expect:** works at moderate κ; at κ·u_lowprec ≳ 1 the update error exceeds the tiny Schur pivot, the pivot goes ≤ 0, `sqrt` gives NaN.
- **nixnan:** tensor-core instructions visible (NaN in HMMA inputs/outputs if they occur); relate the first exception to the block step `k`.
- **Sweep:** δ (or κ) from E2, to find the mixed-precision threshold and compare with the naive one in E3.

### E5 — Randomised matrices: "sometimes silent, sometimes exceptional"
- **Setup:** `QΛQᵀ`, κ ∈ {1e2 … 1e7}, 5 seeds, configurations `F32`, `mixed-F16`, `mixed-BF16`, `mixed-TF32`, `naive-F16`.
- **Output:** a table (config × κ) of the four statuses, with the nixnan exception counts beside the error norms.
- **Purpose:** show that nixnan reports only the `nan` and `breakdown` cases and says nothing about `inaccurate`, so the factorization error / residual must be checked too (same lesson as the LU clamp story).
- **Budget:** 5 configs × 6 κ × 5 seeds = 150 native runs, each well under a second at n=256; nixnan on a **selected** subset only (about 10 runs).

### E6 — nixnan views on the failing case
For one failing run from E3 and one from E4:
- default report; `PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1` first-exception trace;
- `HISTOGRAM=1` with a `BIN_SPEC_FILE` (as in `../lu_solve_gpu/bin_spec_*.json`) to watch the exponent range of the Schur-complement values approach zero: the "danger zone" view;
- `INSTR_MEM=1` to see NaNs stored to `L`;
- `SAMPLING` and `FUNCTION_WHITELIST` to cut instrumentation time if E0 showed a problem.
Record time for each setting.

### E7 — Detect instead of crash (optional, short)
- **Do:** add an explicit pivot test to the kernel (`if (S_jj <= τ·‖A‖)` → report the index and stop), with τ relative to the matrix norm, and compare with nixnan's report on the same input.
- **Purpose:** the "defensive" version of the story and the link to the LU lesson (a relative check beats an absolute clamp). Also point to the Cholesky-IR route in `../mpir_gpu` (a low-precision factor plus FP64 residual refinement) as the way to get accuracy back.

## 7. Deliverables and layout

Planned files in this directory (none exist yet):

| file | content |
|---|---|
| `chol_study.cu` | single-file solver: `-DCFG=` selects the configuration; unblocked and blocked paths; prints `RESULT` lines |
| `gen_chol_matrices.py` | `A_δ` and `QΛQᵀ` generators with condition numbers |
| `run_chol_story.sh` | `e0` … `e7`, `all`, `summary`; native and nixnan runs; same style as `../lu_solve_gpu/run_clamp_story.sh` |
| `story_summary.py` | summary table from `logs/` |
| `logs/<experiment>/…` | native and nixnan logs |
| `CHOLESKY_STORY.md`, slides | the narrative and replacement slides, after the experiments |

## 8. Risks and open questions

1. **FP32 threshold may leave no gap.** If FP32 fails at nearly the same δ as FP16 storage, the "naive reduced vs. careful" contrast is weak. E2 decides; the fallback is the `QΛQᵀ` family.
2. **Exceptions may not appear in the expected form.** `sqrt` of a negative and `0/0` compile to SASS sequences (`MUFU.*` plus fix-ups) that may report as NaN, Inf and division-by-zero in several places; E3 must map them, as was done for `1.0/0.0` in `hello_nixnan`.
3. **Only sm_86 (RTX 3090) is available here.** The proposal and `mpir_gpu` were written for A100 (sm_80). Results are specific to this machine until run elsewhere.
4. **TF32 and the exact rounding of tensor-core inputs** (truncation vs round-to-nearest in `wmma::__float_to_tf32`) change the δ threshold slightly; measure rather than predict.
5. **The matrix `A_δ` is special** (rank-one plus a shift); results on it say how the failure looks, not how often it occurs. E5 is there for that.
6. **The PDF is an unverified chat transcript.** Its tutorial proposal contains the correct core idea and one misleading table row (Section 3). Nothing in it should be quoted in the tutorial without the check in this plan.

## 9. Results so far: E0 and E1 on M0 and M1 (RTX 3090, sm_86, CUDA 12.4)

Matrices (`gen_chol_matrices.py`, files in `matrices/`, definitions and κ in `matrices/matrices.tsv`), both `A = 1 1ᵀ + δI`:

| name | δ | κ₂ (n=256) | κ₂ (n=64) |
|---|---|---|---|
| M0 | 2⁻⁶ | 1.64e4 | 4.10e3 |
| M1 | 2⁻¹² | 1.05e6 | 2.62e5 |

κ₂ is given analytically and checked against the eigenvalues of the stored matrix (they agree to 4 digits). M1 is the matrix of the first E0/E1 run;
its results reproduce that run exactly. M0 is the benign one: FP16/BF16 storage keeps δ=2⁻⁶. The generator also writes random
`Q diag(σ) Qᵀ` matrices on request (`--qlq KAPPA`, for E5); tested once, not used yet.

Code: `chol_study.cu` (FP64/FP32, `unblocked` and `blocked` paths, 5 kernels each; reads `--csv`), `run_chol_story.sh` (`e0`, `e1`, `all`, `summary`),
`chol_summary.py`. Logs: `logs/e0/`, `logs/e1/`. Reproduce: `./run_chol_story.sh` (about 2 minutes, most of it the one `mpir_gpu` timing run).

### E0 — budget: **passes** for both matrices

| run (FP32) | native | nixnan | distinct kernels |
|---|---|---|---|
| M0 n=64 unblocked / blocked | 0.4 / 0.3 s | 2.2 / 2.2 s | 5 |
| M0 n=256 unblocked / blocked | 0.3 / 0.3 s | 2.9 / 3.0 s | 5 |
| M1 n=64 unblocked / blocked | 0.3 / 0.3 s | 2.1 / 2.2 s | 5 |
| M1 n=256 unblocked / blocked | 0.3 / 0.3 s | 2.8 / 3.0 s | 5 |

- Far inside the limits (60 s native, 120 s nixnan). The nixnan time is dominated by the one-time instrumentation of 5 kernels (about 0.3 s each).
  **n=1024 therefore looks affordable for the custom kernels; not run.**
- For comparison, the existing cuSOLVER/cuBLAS `../mpir_gpu/ir_gpu` (n=256, κ=1e3, F32): native 0.5 s (1.1 s in the first run), **nixnan 87.9 s** (92.5 s in the first run).
  Its nixnan run prints a complete report and then ends with `terminate called without an active exception` and an abort; I did not investigate that.
  I did not rebuild `ir_gpu`; the existing binary runs on sm_86.
- E0 timed FP32 only; FP64 runs are covered by E1 (also seconds).

### E1 — reference: **passes** (n=256)

Four columns: for each matrix, the factorization error ‖A−LLᵀ‖/‖A‖ and the forward error ‖x−1‖/‖1‖.

| config, path | M0: ‖A−LLᵀ‖/‖A‖ | M0: ‖x−1‖/‖1‖ | M1: ‖A−LLᵀ‖/‖A‖ | M1: ‖x−1‖/‖1‖ |
|---|---|---|---|---|
| F64 unblocked | 1.0e-16 | 1.1e-14 | 1.1e-16 | 3.1e-13 |
| F64 blocked | 9.5e-17 | 4.0e-13 | 1.1e-16 | 3.3e-13 |
| F32 unblocked | 1.3e-9 | 8.8e-6 | 1.3e-9 | 7.5e-6 |
| F32 blocked | 1.0e-7 | 5.6e-4 | 5.2e-8 | 1.8e-2 |

- All 8 runs: status `ok`, no bad pivot, **no nixnan exception** in any precision section or in memory. Both matrices are valid, and the kernels are right.
- The forward error is the more sensitive metric and depends on the path: for F32 it differs by 2–3 orders of magnitude between `unblocked` and `blocked` (M1: 7.5e-6 vs 1.8e-2) while
  the factorization errors are all small. For M1 the forward error grows with κ as expected for the blocked path (5.6e-4 → 1.8e-2 for κ 1.6e4 → 1.0e6); the unblocked path does not show that growth here (8.8e-6 → 7.5e-6).
  I have not investigated why; likely a property of this particular matrix and summation order. E2 should report the forward error alongside the status.
- The status `ok`/`inaccurate` in `chol_study.cu` uses only the factorization error (`inaccurate` above 1000·ε).

### Deviations from the plan
- E0's "blocked" variant uses SIMT kernels (no tensor cores yet); the WMMA update belongs to E4.
- The matrix names are M0 (benign) and M1 (ill-conditioned); E2 onward will add the δ sweep and the randomised matrices.
