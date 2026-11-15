# Experimental plan: Cholesky on a badly conditioned matrix, mixed precision, under nixnan

Status: **E0, E1 (Section 9), E2 (Section 10), E3 (Section 11) and E4 (Section 12) and E5 (Section 13) and E6 (Section 14) and E7 (Section 15) are implemented and run; the slides and report cover E0-E7.** The numbers marked "checked" in Sections 3 and 6 come from a quick CPU check (NumPy/SciPy, LAPACK) made while writing the plan.
"checked" come from a quick CPU check (NumPy/SciPy, LAPACK) made while writing the plan.

Source: `cg-and-cholesky-nixnan.pdf` in this directory (6 pages). It is a ChatGPT transcript, not a reference: it
explains CG, asks whether CG suits tensor cores, then explains Cholesky and ends with a proposal for a nixnan
tutorial example. This plan keeps that proposal's idea and tests each claim instead of assuming it.

Documents for E0--E4 and the sqrt study: slides `Chol_Slides.pdf` (source `Chol_Slides.tex`), full report `chol_report.pdf` (source `chol_report.tex`, includes the NixNan details); both are built from the logs through `make_chol_docs.py` (fragments in `docs_gen/`).

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
| FP32 "remains safe" because 1+δ rounds only below 2⁻²⁴ | **Partly right, see Section 10.** *First reading (CPU, kept for the record):* misleading, 2⁻²⁴ is where *storage* rounds. FP32 *factorization* fails much earlier from cancellation in the Schur complement: LAPACK in FP32 factors δ=2⁻¹⁴ (κ≈4e6) and reports "not positive definite" at δ=2⁻¹⁶ (κ≈2e7); 2⁻¹⁵ not tried | E2 found the thresholds on the GPU: breakdown does occur at 2⁻²⁴ for our FP32 kernels, but LAPACK and the kernels disagree between 2⁻¹⁶ and 2⁻²², and accuracy is lost long before. The demo δ must sit **between** the FP32 and FP16 thresholds; δ=2⁻¹² does (Section 10) |
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
  the factorization errors are all small. Going from M0 to M1 the forward error of the blocked path grows with κ (5.6e-4 → 1.8e-2 for κ 1.6e4 → 1.0e6); the unblocked path does not show that growth here (8.8e-6 → 7.5e-6).
  I have not investigated why; likely a property of this particular matrix and summation order. E2 should report the forward error alongside the status.
- The status `ok`/`inaccurate` in `chol_study.cu` uses only the factorization error (`inaccurate` above 1000·ε).

### Deviations from the plan
- E0's "blocked" variant uses SIMT kernels (no tensor cores yet); the WMMA update belongs to E4.
- The matrix names are M0 (benign) and M1 (ill-conditioned); E2 onward will add the δ sweep and the randomised matrices.

## 10. Results: E2, where each format breaks (RTX 3090, sm_86; `A_δ`, n=256)

Code: `chol_study.cu --sweep LO HI --stores ... --paths ...` (one process, all δ), `e2_emulate.py` (CPU cross-check), `run_chol_story.sh e2`
(about 1 minute), tables from `chol_summary.py`. Logs: `logs/e2/` (`sweep_f32.log`, `sweep_f64.log`, `emulate.tsv`, `confirm_*`). δ = 2⁻ᵉ, κ₂ = (n+δ)/δ.
"Store" = the format `A` is rounded to before factoring. **All arithmetic is FP32 (FP64 for the FP64 rows); half-precision arithmetic is E3b.**
Status: `ok`; `inacc` = factorization error above 1000·ε or forward error above 0.1 (my threshold, arbitrary); `BRK` = the kernel saw a pivot ≤ 0.

### Where the failures start

| arithmetic / store | first not-ok e (κ₂) | status, cause | last ok |
|---|---|---|---|
| FP32 / **bf16** | 8 (6.6e4) | breakdown, **storage** (1+2⁻⁸ rounds to 1: tie, round-to-even) | 7 |
| FP32 / **f16** | 11 (5.2e5) | breakdown, storage (1+2⁻¹¹ is a tie, rounds to 1) | 10 |
| FP32 / **tf32** | 12 (1.0e6) | breakdown, storage | 11 |
| FP32 / f32, unblocked | 23 (2.1e9) inaccurate; **breakdown at 24** (storage: 1+2⁻²⁴ rounds to 1) | | 22 |
| FP32 / f32, blocked | 14 (4.2e6) inaccurate (not monotone: e=16 is ok again); breakdown at 24 | arithmetic | 13 |
| FP64 / f64 | 52 (unblocked) or 43 (blocked) inaccurate; **breakdown at 53** (1+2⁻⁵³ rounds to 1 in the matrix itself) | | 51 / 42 |

- **Predictions confirmed:** FP16 and BF16 storage break where 1+δ is a tie that rounds to 1 (2⁻¹¹, 2⁻⁸); the failure is the stored matrix becoming
  exactly `11ᵀ`: **the first bad pivot is index 1 (the second pivot is 0) in every breakdown**, as in the PDF.
- **TF32 differs from FP16 by one step:** with ties away from zero (`cvt.rna`, as `wmma::__float_to_tf32`) 1+2⁻¹¹ rounds *up* to 1+2⁻¹⁰, so TF32 still works at e=11 and fails at 12.
  With round-to-nearest-even it would fail at 11 like FP16. The rounding mode of the tensor-core inputs therefore moves the threshold, as the plan's risk 4 expected.
- **The demo point works:** at δ=2⁻¹² (M1, κ₂ = 1.05e6) FP32 is `ok` on both paths while TF32, FP16 and BF16 storage all `BRK`. At δ=2⁻¹⁰ (κ₂ = 2.6e5) FP16 and TF32 are still `ok` and BF16 breaks.
- **The FP32 thresholds are implementation-dependent, and `A_δ` is special.** LAPACK (`spotrf` via SciPy) reports "not positive definite" for FP32 at e=16–22; the GPU
  kernels do not (unblocked is ok up to e=22). The kernels' CPU/GPU agreement for tf32, f16 and bf16 is exact at every e (those fail at storage); for f32 they disagree
  at e=16–22 as just said. `A_δ`'s structure (every Schur complement keeps the form `s·11ᵀ + δ'I`) makes the errors benign for some operation orders, so a
  threshold measured on it says how the failure looks, not where it will occur in general (plan risk 5). E5's random matrices are the real test.
- **Accuracy is lost long before breakdown.** FP32 forward error exceeds 0.1 from e=14 (blocked) or 23 (unblocked), with no pivot ≤ 0 and no exception.
  Cause label `arithmetic` for the FP64 breakdown at e=53 is loose: there the matrix itself, built in double, has already lost δ.

### nixnan on chosen points (`logs/e2/confirm_*`, FP32 arithmetic, n=256)

| store, e | path | status | nixnan |
|---|---|---|---|
| f32, 12 | both | ok | no exceptions |
| f32, 14 (blocked) / 23 (unblocked) | | inaccurate | **no exceptions** (the silent case) |
| tf32 / f16 / bf16 at their first failing e, and at 12 | unblocked | breakdown, bad pivot 1 | FP32 NaN 231 (675,688 repeats), Inf 13, Div0 4, memory NaN 1 |
| same | blocked | breakdown, bad pivot 1 | FP32 NaN 566 (689,040 repeats), Inf 17, Div0 5, memory NaN 1 |

- The counts are **identical across tf32, f16 and bf16** (and across δ): all three stores produce the same rounded matrix `11ᵀ`, so the same operations run on the same values.
  Per path, the exception profile is a property of the algorithm, not of the format.
- The exceptions are all in the FP32 section: the conversions to the narrow formats happen on the host here, and nixnan does not instrument conversion opcodes anyway.
- E3 will trace these exceptions to kernel and source line.

### Deviations and caveats
- The status threshold (forward error 0.1) is mine; moving it moves the `inacc` boundary, not the `BRK` one.
- The matrix is rounded on the host (the formats are emulated for storage); it is not yet produced by device conversions or tensor-core inputs (E4).
- FP64 was swept to e=56 only for the f64/f64 combination.

## 11. Results: E3, naive reduced precision (RTX 3090, sm_86; `A_δ`, n=256, unblocked and blocked)

Code: `chol_half.cu` (new: unblocked Cholesky whose arithmetic is **true FP16 or BF16**, built with and without `-DUSE_BF16`), the FP32-arithmetic
kernels of `chol_study.cu` with `--store f16|bf16`, `run_chol_story.sh e3` (about 1 minute), `chol_summary.py` (trace analysis). Logs: `logs/e3/`
(`*_native.log`, `*_nixnan.log` = default report, `*_detail.log` = `PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1`, `sweep_h16.log`, `sweep_hbf16.log`).

### E3a: FP16/BF16 *storage*, FP32 arithmetic, δ=2⁻¹² (M1)

| | unblocked | blocked |
|---|---|---|
| status | breakdown, **first bad pivot = 1** (second pivot is 0), cause: storage | same |
| nixnan report (FP16 and BF16 storage identical) | FP32 NaN 231 (675,688 repeats), Inf 13, Div0 4, memory NaN 1 | FP32 NaN 566 (689,040), Inf 17, Div0 5, memory NaN 1 |
| first exception | `MUFU.RSQ` div0 in `pivot_kernel` | `MUFU.RSQ` div0 in `potrf_tile_kernel` |
| chain (kernel: report lines) | pivot: 6 → scale: 29 → rank1: 5 → forward: 153 → backward: 56 | potrf_tile: 159 → trsm_panel: 153 → syrk: 68 → forward: 153 → backward: 56 |

- **The story in the PDF is confirmed, with one subtlety.** The stored matrix is `11ᵀ`; the second pivot is exactly 0; the division by it (`MUFU.RCP` of 0, then `FFMA`/`FMUL` fix-ups)
  gives Inf and then NaN, which `rank1`, the panel/update kernels and both substitutions propagate to the end.
- **The first report is a discarded `rsqrt(0)`, not the defect (corrected explanation; the first version of this bullet only said "benign expansion").** IEEE `sqrtf` compiles to a sequence that executes `MUFU.RSQ` for every input *before* range-testing it (SASS of `pivot_kernel`: `MUFU.RSQ R7, R0` at 0x00a0, then `ISETP.GT.U32 … 0x727fffff`); in-range inputs take the Newton-refinement fast path,
  others call a slow path that returns the special value (±0 → ±0, negative → NaN). For `x = 0` the hardware really computes `rsqrt(+0) = +Inf` (an IEEE division by zero), nixnan reports it (`div0` on operand 1, the zero input; `infinity` on operand 0), and the slow path returns 0, so the Inf is never used.
  At source level `sqrt(0) = 0` is exact: a **false positive of kind div0/Inf, not a NaN** (nothing at this site produces a NaN). Verified by building with `-prec-sqrt=false` (`MUFU.SQRT`, one instruction) and re-running under nixnan (`./run_chol_story.sh sqrt`, `logs/sqrt/`):
  the `pivot_kernel` div0/Inf lines at `sqrt(0)` disappear (6 → 2 lines, the 2 left are NaN inputs from earlier kernels) and the first report becomes `MUFU.RCP` div0 in `scale_kernel`, the **genuine** division by the zero pivot. In E4, `sqrt` of a *negative* pivot reports NaN at the sqrt instruction in both builds (genuine).
  The FP16/BF16 build of E3b (`hsqrt` → `MUFU.SQRT`) never had the false positive. The remaining pivot-kernel lines in the IEEE build (`NaN` on `MUFU.RSQ` operands 1 and 0, `@P0 FADD.FTZ …, 1`) are NaN inputs propagated from earlier kernels.
- **Source lines:** only the forward and backward substitution kernels carry `file:line` in the report (153 and 56 lines); the reports in `pivot`, `scale`, `rank1` (and the tile kernels) have none.
  The cubin does contain a line table (`nvdisasm -g`), and I did not find why nixnan cannot use it for those kernels. Attribution therefore goes by **kernel and SASS instruction**.
- The exceptions are all FP32: the host does the FP16/BF16 rounding, and nixnan does not instrument conversion opcodes anyway.

### E3b: true FP16 / BF16 arithmetic (every operation rounds to the narrow format), unblocked

SASS of the FP16 build: 39 `HFMA2`, 3 `MUFU.RCP`, 1 `MUFU.SQRT`, 3 `FFMA`, 3 `FMUL`, 7 `F2FP.PACK_AB`, `HSETP2`, `HADD2`. So the update is a genuine FP16 fused multiply-add, but
**`__hdiv` and `hsqrt` run in FP32** (reciprocal in `MUFU.RCP`, converted back by `F2FP`).

| format | sweep of δ=2⁻ᵉ (ok / inacc / BRK) | at δ=2⁻¹² |
|---|---|---|
| FP16 arithmetic | ok for e ≤ 9, **inacc at 10**, BRK from 11 | breakdown (storage), bad pivot 1; nixnan: **FP16 NaN 156 (636,986 repeats)**, FP32 NaN 17, Inf 6, Div0 3 |
| BF16 arithmetic | inacc already at e=2…7, BRK from 8 | breakdown (storage), bad pivot 1; nixnan: **BF16 NaN 156 (636,986)**, FP32 NaN 73, Inf 12, Div0 3 |

- First report here is in `scale_kernel` (`MUFU.RCP` of the zero pivot), then `rank1` (HFMA2 in the FP16/BF16 section), then `pivot`, forward, backward. The FP16/BF16 NaNs come from the
  `HFMA2` updates and substitutions; the division itself is FP32. **One source-level operation is reported in two precision sections.**
- **The silent (finite-but-wrong) points:** δ=2⁻¹⁰ for FP16 arithmetic and δ=2⁻⁷ for BF16 arithmetic.
  The stored matrix is exact (δ representable), no pivot is ≤ 0, the factorization error is small (3.8e-6 and 3.1e-5), **nixnan reports no exceptions**, and the forward error is **15.97** in both, i.e. x is garbage.
  The same value 15.97 (= √255) also appeared in the FP32 (e=23) and FP64 (e=52) sweeps of E2; I do not know why it recurs.
  This is the "small backward error, huge forward error" signature of κ·u ≫ 1 (κ = 2.6e5, u = 2⁻¹⁰ for FP16).
- **The `inacc` status of BF16 for every e ≤ 7 is partly my threshold:** BF16 arithmetic gives forward errors around 0.6–1 even at κ ≈ 1e3 (u = 2⁻⁸), above the 0.1 cut-off; the sweep says "BF16 arithmetic is never accurate on this matrix", not where it starts to fail.

### What E3 gives the tutorial
- A deterministic, fast (3 s) failing run: `./bin_chol/chol_f32 --log2delta -12 --store f16 --path unblocked` under `LD_PRELOAD=nixnan.so`, and a quiet one (`--log2delta -12 --store f32`).
- A silent run for contrast, where only the forward error (or a pivot/residual check) shows the problem.
- Two caveats to say aloud: the first nixnan line may be the `sqrt(0)` expansion, and FP16 `/` and `sqrt` are FP32 operations.

### Deviations and caveats
- E3b covers the unblocked path only (the blocked path would need tile kernels in half arithmetic).
- Not done: the "variant: δ slightly above the threshold" for FP32-arithmetic storage: E2 shows FP16 at δ=2⁻¹⁰ is `ok` there (stored exactly, FP32 arithmetic), so the silent behaviour only appears with narrow arithmetic (E3b).

## 12. Results: E4, tensor-core trailing update (RTX 3090, sm_86; `A_δ`, n=256, NB=32)

Code: `chol_tc.cu` (new), `run_chol_story.sh e4` (about 1 minute), `chol_summary.py` (E4 report). Logs: `logs/e4/` (`sweep_tc.log`, `sass_hmma.txt`, and per point `*_native.log`, `*_nixnan.log`, `*_detail.log`).
Design: blocked right-looking Cholesky; `potrf_tile` and `trsm_panel` in **FP32**; the trailing update `A_RR ← A_RR − L_Rk L_Rkᵀ` is **WMMA on tensor cores with FP32 accumulate**, one warp per 16×16 tile:
FP16 (`__half`) and BF16 inputs through a rounded copy of the panel (`convert_panel_kernel`), TF32 inputs read from the FP32 panel and converted with `__float_to_tf32` (ties away from zero);
FP32 forward/backward substitution on the FP32 factor. The pivot is still unguarded.

### Checks that were open in the plan
- **SASS contains the expected tensor-core instructions** (`cuobjdump -sass`): `HMMA.16816.F32` ×4 (FP16), `HMMA.16816.F32.BF16` ×4, `HMMA.1684.F32.TF32` ×16.
- **Range:** all panel entries have magnitude ≤ 1 here, far below the FP16 maximum 65504; no overflow in the narrow-format conversion is possible on this matrix.
- **Cost:** a nixnan run takes about 3.5 s (7 distinct kernels, FP16 path), inside the budget.

### Sweep (δ = 2⁻ᵉ, `logs/e4/sweep_tc.log`)

| e (κ₂) | FP16 update | BF16 update | TF32 update |
|---|---|---|---|
| 2…7 (1e3…3e4) | inacc (forward error 0.05–0.78; 0.0642 at e=3 is `ok`) | inacc (0.17–1.44) | as FP16 |
| 8…10 (6.6e4…2.6e5) | inacc / `ok` at e=10 (fwd 0.083) | **breakdown, first bad pivot 33** | inacc / `ok` at e=10 (0.088) |
| 11…21 (5.2e5…5.4e8) | **breakdown, first bad pivot 33** | breakdown (33) | breakdown (33) |
| 22, 23 | finite garbage (forward error 18.5 and 16.0), no pivot ≤ 0 | same | same |
| 24…26 | breakdown, bad pivot 1 (FP32 storage of `A_δ` itself loses δ) | same | same |

- **The failure moves to the second block.** Every breakdown has **first bad pivot 33 (block step k=1)**, not 1 as with naive storage (E3): the pivots of block 0 are fine; the update with the
  rounded panel is what damages the Schur complement, and the damage shows at the next diagonal tile. Mixed precision postpones the failure to where the rounded update is first used.
- **The thresholds match the naive ones except TF32:** FP16 e=11 (E2: 11), BF16 e=8 (E2: 8), TF32 e=11 (E2: 12).
  Explanation, consistent with the data: the dominant panel entry is `L_{i0} = 1/√(1+δ) ≈ 1 − δ/2`, and I checked on the CPU that it rounds to **exactly 1.0** at e=8 (BF16) and e=11 (FP16, TF32),
  and not before. Because `L_{i0}` is not an exact tie, TF32's ties-away rule no longer matters, which is why TF32 now breaks with FP16 (it survived one step longer as *storage* of `1+δ` in E2).
  I did not prove that this rounding is the only cause of the breakdown.
- **Without refinement the mixed-precision factorization is inaccurate long before it breaks:** forward error 0.05–1 already at κ₂ = 1e3, and 0.66 on M0 (e=6, κ₂ = 1.6e4) for all three formats.
  The factor is only useful as a preconditioner (the iterative-refinement route in `../mpir_gpu`). My 0.1 threshold therefore labels most low-e points `inacc`; the `ok` at e=3 and e=10 are borderline (forward error 0.064 and 0.083).
- **Non-monotone again:** at e=22 and 23 the panel entry rounds to exactly 1 and the run finishes with forward error 16–18.5 and no pivot ≤ 0 (the same 15.97 value as in E2/E3 at e=23).

### nixnan (`logs/e4/*_nixnan.log`, `*_detail.log`)

| point | status | nixnan report |
|---|---|---|
| FP16 or BF16 update, δ=2⁻¹² | breakdown, bad pivot 33 (k=1) | **FP16 (resp. BF16) NaN 8 (1,616 repeats)**, FP32 NaN 514 (316,906), memory NaN 4 (1,620) |
| TF32 update, δ=2⁻¹² | breakdown, bad pivot 33 | FP32 NaN 570 (328,218), memory NaN 4 (1,620); **no FP16/BF16 section** |
| M0 (δ=2⁻⁶), any format | inaccurate, forward error 0.61–0.66 | **no exceptions** |
| last non-breakdown δ (FP16, TF32: 2⁻¹⁰; BF16: 2⁻⁷) | ok / inacc | **no exceptions** |

- **The first exception is the real defect this time:** `MUFU.RSQ` in `potrf_tile_kernel`, reported as **NaN** (square root of a negative Schur-complement diagonal), not the benign `sqrt(0)` expansion of E3.
  This is the scenario in the PDF: a negative pivot gives a NaN.
- **Chain:** `potrf_tile` (151 lines) → `trsm_panel` (147) → `tc_update` (30) → `forward` (147) → `backward` (51). The tensor-core kernel only sees NaN *after* the factorization has already failed.
- **NixNan does instrument the HMMA instructions:** 14 report lines with `HMMA.16816.F32 …`: **NaN in operands 1 and 2 (the A and B inputs), decoded as f16 (BF16 for the BF16 build)**, which is the FP16 (BF16) section above.
  For TF32 there are 62 HMMA lines, typed **f32** (TF32 inputs are 32-bit registers), so they land in the FP32 section, with no tensor-core-specific section.
  Reading the tensor-core report therefore needs the instruction text, not only the section.
- **Silence again where the matrix is still SPD** (M0 and the last-ok points): nixnan is blind to the 5–66 % forward error.
- The parsers had to learn the kernel name format `void tc_update_kernel<__half>`, which nixnan prints without its template arguments (`in function void tc_update_kernel`).

### Caveats
- `A_δ` is special (one dominant panel column): the clean link between the rounding of `L_{i0}` and the breakdown is a property of this matrix; E5 is the real test.
- Only FP32 accumulate was run; FP16 accumulate (`HMMA.16816.F16`, as in `lu_solve_gpu_tc`) would break much earlier and is not covered.
- NB=32 only; the number of mma steps per tile (2 for FP16/BF16, 4 for TF32) was not varied.
- The slide deck and report (`Chol_Slides.pdf`, `chol_report.pdf`) now cover E0–E4 and the sqrt study (`run_chol_story.sh sqrt`, `logs/sqrt/`).

## 13. Results: E5, random matrices (RTX 3090, sm_86; `Q diag(sigma) Q^T`, n=256, 5 seeds)

Reproduce: `./run_chol_story.sh e5` (about 2 min in total: 150 native runs, then 11 nixnan points, each run twice). Logs: `logs/e5/` (`sweep.log`, per-point `*_nixnan.log` and `*_detail.log`, `matrices.txt` with the numeric kappa of each matrix). The 30 matrices (1.3 MB each) are regenerated from the seeds into `matrices_q/`, which is git-ignored.

Set-up. `sigma_j = kappa^(-j/(n-1))`, Q from the QR of a seeded Gaussian (`gen_chol_matrices.py --qlq`), kappa in {1e2 ... 1e7}, seeds 0-4. `b = A*1`, so the exact solution is 1. The E5/E7 binaries are built with `-DGUARD` (`bin_chol/g_chol_*`): the same kernels plus a running minimum of the relative pivot (`min_rel_pivot` in the RESULT line) and the `--guard` option; the default build used for E0-E4, E6 and the sqrt study is unchanged (its SASS was compared with the earlier build: identical). Because the guard build adds FP instructions, NixNan counts for E5 differ slightly from a plain build (FP32 NaN 505 became 513 for mixed-F16 kappa=1e6). Configurations: `F32` (blocked FP32, `chol_study`), `mixed-F16/BF16/TF32` (`chol_tc`, FP32 panel, tensor-core update, FP32 accumulate), `naive-F16` (`chol_half`, true FP16 arithmetic, unblocked). `chol_tc` and `chol_half` gained `--csv`.

```
E5: random Q diag(sigma) Q^T, n=256, 5 seeds per cell. Cell = status counts; second line = median forward error (finite runs)
config     kappa 1e2             kappa 1e3             kappa 1e4             kappa 1e5             kappa 1e6             kappa 1e7             
F32        5 ok                  5 ok                  5 ok                  5 ok                  5 ok                  5 ok                  
           2.1e-06               1.2e-05               9.6e-05               6.0e-04               6.5e-03               4.4e-02               
mixed-F16  5 ok                  5 inacc               5 inacc               5 inacc               5 BRK                 5 BRK                 
           1.0e-03               8.8e-03               6.9e-02               9.1e-01               -                     -                     
mixed-BF16 5 inacc               5 inacc               5 inacc               5 BRK                 5 BRK                 5 BRK                 
           8.9e-03               6.4e-02               1.7e+00               -                     -                     -                     
mixed-TF32 5 ok                  5 inacc               5 inacc               5 inacc               5 BRK                 5 BRK                 
           1.1e-03               8.8e-03               7.1e-02               8.9e-01               -                     -                     
naive-F16  5 ok                  5 inacc               3 BRK 2 inacc         5 BRK                 5 BRK                 5 BRK                 
           2.3e-02               1.3e-01               3.2e+01               -                     -                     -                     
E5 nixnan on chosen points (counters unique(repeats); native status in brackets)
  F32         kappa 1e2  seed 0 [ok]: no exceptions
  mixed-F16   kappa 1e2  seed 0 [ok]: FP16 Sub 8(634)
  mixed-F16   kappa 1e3  seed 0 [inaccurate]: FP16 Sub 8(681)
  mixed-F16   kappa 1e6  seed 0 [breakdown]: FP16 NaN 10(14); FP16 Sub 12(952); FP32 NaN 513(121662); FP32 Memory NaN 4(20)
  mixed-BF16  kappa 1e2  seed 0 [inaccurate]: no exceptions
  mixed-BF16  kappa 1e5  seed 0 [breakdown]: BF16 NaN 8(96); FP32 NaN 522(134884); FP32 Memory NaN 4(100)
  mixed-TF32  kappa 1e3  seed 0 [inaccurate]: no exceptions
  mixed-TF32  kappa 1e6  seed 0 [breakdown]: FP32 NaN 569(121774); FP32 Memory NaN 4(20)
  naive-F16   kappa 1e2  seed 0 [ok]: FP16 Sub 41(25666)
  naive-F16   kappa 1e3  seed 0 [inaccurate]: FP16 Sub 57(40314)
  naive-F16   kappa 1e4  seed 0 [breakdown]: FP16 NaN 34(97653); FP16 Sub 82(63676); FP32 NaN 27(542)
```

What the table says (statuses use the E2 rule: `inaccurate` = factorization error above 1000 eps or forward error above 0.1; the 0.1 is arbitrary):
- **Same ordering as on `A_delta`, now on random matrices.** FP32 never fails in this range. Mixed precision with FP16/TF32 inputs is inaccurate from kappa=1e3 and breaks down at 1e6; BF16 is inaccurate already at 1e2 and breaks down at 1e5; naive FP16 arithmetic breaks down from 1e4 (3 of 5 seeds; the other 2 are inaccurate: seed-dependent at the threshold).
- **Every breakdown here is caught by the code's own pivot check** (status `breakdown`; a pivot <= 0 is recorded). The status `nan` (non-finite output nobody noticed) did not occur. The "silent" failures are the `inaccurate` ones, not NaN ones.
- **The inaccurate cells are the large majority of the failing region, and nixnan sees little of them.** `mixed-BF16` kappa=1e2 and `mixed-TF32` kappa=1e3 are inaccurate (forward error 9e-3 and 9e-3) with **no exception at all**. `mixed-F16` at kappa=1e2 (ok) and 1e3 (inaccurate) both report 8 FP16 subnormals on the HMMA inputs (operands 1 and 2): the same count in an ok and an inaccurate run, so it does not separate them. `naive-F16` reports many subnormals (41 unique, 25666 repeats) already at kappa=1e2 where the status is ok.
- **NaN reports appear only where the factorization has already failed** (breakdown rows: FP32 NaN 513-569 unique, 4 NaN stores to memory in the tensor-core runs; FP16 and BF16 NaNs only where the format is used in the arithmetic).
- Lesson as in the LU story: nixnan reports exceptions, not accuracy. The factorization error or residual (here `fact_err`, `fwd_err`) has to be checked beside it.

Caveats.
- `F32` at kappa=1e7 has median forward error 4.4e-2 and is classed ok only because the threshold is 0.1.
- One matrix family, one size, FP32 accumulate only, NB=32; five seeds give counts, not confidence intervals.
- The subnormal reports on the HMMA inputs come from the FP16 panel entries (values below 6.1e-5); I did not trace them to individual panel entries.
- The `mixed-*` rows use the panel conversion exactly as in E4; I did not separate the effect of the panel rounding from the accumulate/update rounding.

## 14. Results: E6, the histogram "danger-zone" view and the cost of nixnan's options (RTX 3090, sm_86; `A_delta`, n=256)

Reproduce: `./run_chol_story.sh e6` (about 50 s). Logs and the two generated bin specs: `logs/e6/`.

Set-up. Three scenarios, each run healthy (delta = 2^-6) and failing (delta = 2^-12) under `HISTOGRAM=1 BIN_SPEC_FILE=...`: s1 FP32 arithmetic on FP16-rounded A (unblocked, E3a), s2 the tensor-core update (E4), s3 true FP16 arithmetic (E3b). Bins: FP32 `zero` (exponent field 0), [-126,-30], [-29,-14], -13, -12, [-11,-8], [-7,-1], 0, [1,10]; FP16 the analogous set. -12 is the binade of the exact pivots (delta = 2^-12). The zero bin has `(record_inst)` so each hit names its SASS instruction. `count` = 64: a bin reports at every 64th occurrence, so a cell below is (reports x 64), an approximate number of operand values.

```
E6: nixnan histogram, danger-zone view. Cell = approx. occurrences (threshold reports x count, count=64) of values with that exponent
 s1: FP32 arithmetic on FP16-rounded A (chol_f32 --store f16, unblocked)
   delta=2^-6
     backward_kernel [-11..-8]:28672  [-126..-30]:384  [-29..-14]:576  [-7..-1]:82624  [0]:17408  [1..10]:5888  [zero]:256
     forward_kernel [-11..-8]:28672  [-126..-30]:384  [-29..-14]:576  [-7..-1]:94592  [0]:4736  [1..10]:6528  [zero]:192
     pivot_kernel   [-126..-30]:384  [-29..-14]:64  [-7..-1]:1984  [1..10]:1216
     rank1_kernel   [-11..-8]:6316608  [-12]:1380480  [-13]:1649984  [-29..-14]:690816  [-7..-1]:1113920  [0]:32576
     scale_kernel   [-11..-8]:129024  [-12]:13248  [-126..-30]:68736  [-13]:20352  [-29..-14]:78016  [-7..-1]:115264  [0]:1216  [1..10]:226688  [zero]:32576
   delta=2^-12
     backward_kernel [0]:256  [zero]:256
     forward_kernel [0]:256  [1..10]:512  [zero]:768
     rank1_kernel   [0]:97856  [zero]:65024
     scale_kernel   [0]:4032  [zero]:34880
 s2: tensor-core update (chol_tc --fmt f16)
   delta=2^-6
     backward_kernel [-11..-8]:28672  [-126..-30]:192  [-29..-14]:768  [-7..-1]:95232  [0]:4672  [1..10]:5952  [zero]:192
     forward_kernel [-11..-8]:28672  [-126..-30]:384  [-29..-14]:512  [-7..-1]:94592  [0]:4736  [1..10]:6528  [zero]:192
     potrf_tile_kernel [-11..-8]:108096  [-12]:10880  [-126..-30]:8768  [-13]:21760  [-29..-14]:44608  [-7..-1]:33984  [0]:640  [1..10]:28800  [zero]:3904
     tc_update_kernel [-11..-8]:46400  [-12]:113536  [-13]:37312  [-29..-14]:139712  [-7..-1]:109056  [0]:26816  [zero]:78784
     trsm_panel_kernel [-11..-8]:1128064  [-12]:187328  [-126..-30]:61248  [-13]:273280  [-29..-14]:209920  [-7..-1]:283776  [0]:8000  [1..10]:199104  [zero]:28608
   delta=2^-12
     backward_kernel [-11..-8]:7104  [-7..-1]:832  [1..10]:192  [zero]:192
     forward_kernel [-11..-8]:18944  [-126..-30]:64  [-7..-1]:11776  [0]:512  [1..10]:960  [zero]:192
     potrf_tile_kernel [-11..-8]:11840  [-12]:1024  [-126..-30]:1088  [-13]:1472  [-29..-14]:10432  [-7..-1]:4992  [0]:640  [1..10]:3584  [zero]:3904
     tc_update_kernel [-13]:46400  [-29..-14]:384  [0]:134336  [zero]:78784
     trsm_panel_kernel [-11..-8]:234560  [-126..-30]:15552  [-13]:20416  [-29..-14]:228800  [-7..-1]:64128  [0]:8000  [1..10]:49920  [zero]:28608
 s3: true FP16 arithmetic (chol_h16)
   delta=2^-6
     backward_kernel [-10..-8]:20544  [-14..-11]:8064  [-7..-1]:135808  [0]:20608  [1..10]:8384  [zero]:67456
     forward_kernel [-10..-8]:20544  [-14..-11]:8064  [-7..-1]:153600  [0]:5696  [1..10]:5184  [zero]:67776
     rank1_kernel   [-10..-8]:5272768  [-14..-11]:7496384  [-7..-1]:1178944  [0]:32576  [zero]:8388416
   delta=2^-12
     backward_kernel [0]:192  [zero]:67456
     forward_kernel [0]:192  [1..10]:448  [zero]:68800
     rank1_kernel   [0]:97856  [zero]:8486144
 cost of the options (s1, delta=2^-12, wall s): native=0.3; report=2.9; detail=2.9; histogram=3.4; hist_sampling4=3.3; whitelist_rank1=0.8; hist_whitelist=0.8; instr_mem=2.9
   report           threshold reports 0      exception lines 249
   histogram        threshold reports 3185   exception lines 249
   hist_sampling4   threshold reports 2277   exception lines 241
   whitelist_rank1  threshold reports 0      exception lines 5
   hist_whitelist   threshold reports 2545   exception lines 5
   instr_mem        threshold reports 0      exception lines 249
```

What it shows.
- **Healthy runs have a population in the low binades; failing runs have none.** In s1, `rank1_kernel` at delta=2^-6 holds millions of values in [-13,-8] and 690k in [-29,-14] (the Schur complement shrinking toward the small pivots). At delta=2^-12 the only populated bins are exponent 0 (the 1s of the rank-one matrix) and the zero bin: the values do not approach zero, they are exactly 0. The same in s3 (rank1: 12.7M values in [-14,-8] healthy, none failing) and in s2's `tc_update_kernel`.
- **The zero bin names the instruction.** s1 delta=2^-12: 1016 of the about 1.6k zero-bin reports are `rank1_kernel FFMA R3, R8, -R11, R3` (the Schur update `a_ij - l_i l_j` landing on exactly 0), 513 are `scale_kernel FFMA R2, R0, R3, RZ`. Caution: the second is the multiply by the reciprocal with an RZ addend; RZ is a zero operand, so it is the histogram counting an input, not a defect. The zero bin also counts exponent-field 0 subnormals and is populated in the healthy runs too (s3 healthy: 8.4M), so by itself it is not an alarm; the signature is the empty low binades next to a full zero bin.
- **The histogram is an operand census.** It counts the exponent of every FP source and destination operand of every instrumented instruction, per kernel, so the numbers are not "values of the Schur complement". The pivot kernel in the failing runs reports nothing because its few values never reach 64 occurrences in one bin.
- **It does not replace the exception report.** `exception lines` is the same (249) with and without the histogram.

Cost of the options (s1, delta=2^-12, wall seconds; native 0.3):

| option | s | what changes |
|---|---|---|
| default report | 2.9 | startup dominates: about 0.3 s per distinct kernel x 5, plus launch overhead of 255 x 3 instrumented launches |
| + `PRINT_ILL_INSTR`, `ENABLE_FUN_DETAIL` | 2.9 | 768 "running kernel" lines are the extra output |
| + `HISTOGRAM` (count 64) | 3.4 | 3185 threshold reports |
| + `SAMPLING=4` | 3.3 | reports 3185 -> 2277; no visible time saving here |
| `FUNCTION_WHITELIST` (rank1 only) | 0.8 | exception lines 249 -> 5; the file must hold the full demangled name `rank1_kernel(float*, int, int)`; the bare name `rank1_kernel` instrumented nothing and reported nothing |
| `INSTR_MEM=1` | 2.9 | identical to the default report (see below) |

Findings and warnings.
- `INSTR_MEM=1` changed nothing (same exception lines, same time), as `nixnan/test/README.md` suspected: the NaN-store reports are already in the default report. I did not read the code path further.
- Writing `LD_PRELOAD=nixnan.so timeout 100 ./prog` loads the tool into `timeout`, not the program: nothing is instrumented and nothing says so. Use `timeout 100 env LD_PRELOAD=... ./prog`. (I lost several runs to this while setting E6 up.)
- A whitelist entry that does not match a kernel silently produces an empty report, which looks like a clean run.
- The threshold test is `prev % count == 0 && prev != 0`: with `count=1` the first occurrence is not reported, and one occurrence per kernel is never seen; I did not check whether that is intended.
- `count=1` on this program prints about 10^5 lines per run and exceeded my 120 s limit once; count=64 keeps it under 4 s.

Caveats. Operand-level census; one matrix; s2's FP16 HMMA operands are not binned (only FP32 bins were set for it); the per-kernel breakdown of s2 depends on the NB=32 blocking; I did not run `E6` for BF16/TF32.

## 15. Results: E7, an explicit pivot guard against NixNan (RTX 3090, sm_86; random `Q diag(sigma) Q^T`, n=256, and `A_delta`)

Reproduce: `./run_chol_story.sh e7` (about 4 min; needs `matrices_q/` from `e5`). Logs: `logs/e7/`.

The guard. At every pivot the kernel computes `rel = pivot / (original diagonal entry)` and keeps the minimum (`min_rel_pivot`). With `--guard TAU` a pivot with `!(rel > TAU)` (NaN included) is recorded and the kernel returns **before the sqrt**, so no Inf or NaN is created; the host reads the flag after each block step (after each pivot in the unblocked path), prints `GUARD pivot i: relative pivot r <= tau t ...` and stops. Code: `-DGUARD` builds of `chol_study.cu`, `chol_tc.cu`, `chol_half.cu`. The check costs a host sync per step; wall time was 0.3 s with and without it (FP32 unblocked, n=256, best of 3), too coarse to resolve a difference.

Choosing TAU. `rel` is a property of the matrix (about 1/kappa, and the same for the F32 and the mixed runs of one matrix), so a useful TAU has to scale with the unit roundoff `u` of the narrowest format that feeds the arithmetic: TAU = c*u, with u = 2^-24 (FP32), 2^-11 (FP16, TF32), 2^-8 (BF16). Measured on the mixed runs: forward error is about 0.5 u / min_rel (mixed FP16, kappa 300-700) and about 10 u / min_rel (naive FP16 arithmetic), so c fixes the forward error the user is willing to accept: roughly 0.5/c (mixed) or 10/c (naive). This is an empirical fit to my matrices, not a bound.

```
E7a: guard threshold tau = c u, evaluated on the 150 E5 runs from min_rel_pivot (a run is flagged if min_rel_pivot <= c u)
  Cell = non-ok runs flagged / non-ok runs, and ok runs flagged / ok runs (false alarms)
  config     c=1                   c=10                  c=30                  c=100                 c=300                 
  F32        0/0, 0/30             0/0, 0/30             0/0, 0/30             0/0, 0/30             0/0, 2/30             
  mixed-F16  10/25, 0/5            15/25, 0/5            20/25, 0/5            25/25, 0/5            25/25, 0/5            
  mixed-BF16 16/30, 0/0            23/30, 0/0            25/30, 0/0            30/30, 0/0            30/30, 0/0            
  mixed-TF32 10/25, 0/5            15/25, 0/5            20/25, 0/5            25/25, 0/5            25/25, 0/5            
  naive-F16  18/25, 0/5            19/25, 0/5            20/25, 0/5            25/25, 0/5            25/25, 0/5            
E7b: live runs, tau = 100 u. Native status without / with the guard; nixnan counters without / with the guard
  Adelta_f16_e12         plain: breakdown fwd=nan bad_pivot=33 | guard: status=guarded, pivot 1 rel 4.881e-04 <= 4.883e-02 | nixnan plain: FP16 NaN 8(1616); FP32 NaN 522(319118); FP32 Memory NaN 4(1620) | nixnan guard: no exceptions
  F32_k1e7_s0            plain: ok fwd=4.398e-02 bad_pivot=-1 | guard: status=ok (no trip) | nixnan plain: no exceptions | nixnan guard: no exceptions
  mixed-BF16_k1e2_s0     plain: inaccurate fwd=7.674e-03 bad_pivot=-1 | guard: status=guarded, pivot 133 rel 3.812e-01 <= 3.906e-01 | nixnan plain: no exceptions | nixnan guard: no exceptions
  mixed-F16_k1e2_s0      plain: ok fwd=1.009e-03 bad_pivot=-1 | guard: status=ok (no trip) | nixnan plain: FP16 Sub 8(634) | nixnan guard: FP16 Sub 8(634)
  mixed-F16_k1e4_s0      plain: inaccurate fwd=5.921e-02 bad_pivot=-1 | guard: status=guarded, pivot 147 rel 4.120e-02 <= 4.883e-02 | nixnan plain: FP16 Sub 8(838) | nixnan guard: FP16 Sub 8(671)
  mixed-F16_k1e6_s0      plain: breakdown fwd=nan bad_pivot=193 | guard: status=guarded, pivot 83 rel 4.157e-02 <= 4.883e-02 | nixnan plain: FP16 NaN 10(14); FP16 Sub 12(952); FP32 NaN 513(121662); FP32 Memory NaN 4(20) | nixnan guard: FP16 Sub 8(293)
  mixed-TF32_k1e3_s0     plain: inaccurate fwd=8.792e-03 bad_pivot=-1 | guard: status=guarded, pivot 247 rel 4.747e-02 <= 4.883e-02 | nixnan plain: no exceptions | nixnan guard: no exceptions
  naive-F16_k1e3_s0      plain: inaccurate fwd=1.284e-01 bad_pivot=-1 | guard: status=guarded, pivot 251 rel 4.561e-02 <= 4.883e-02 | nixnan plain: FP16 Sub 57(40314) | nixnan guard: FP16 Sub 4(40184)
  naive-F16_k1e4_s0      plain: breakdown fwd=nan bad_pivot=253 | guard: status=guarded, pivot 147 rel 4.220e-02 <= 4.883e-02 | nixnan plain: FP16 NaN 34(97653); FP16 Sub 82(63676); FP32 NaN 27(542) | nixnan guard: FP16 Sub 4(48674)
E7c: held-out kappa (150..700, seeds 10-12): status, forward error, min_rel_pivot / u, and whether tau = 100 u / 300 u would trip
  mixed-F16  kappa 150  seed 10: inaccurate fwd 1.663e-03 min_rel/u =    303   trip@100u: no  trip@300u: no
  naive-F16  kappa 150  seed 10: ok         fwd 2.939e-02 min_rel/u =    306   trip@100u: no  trip@300u: no
  mixed-F16  kappa 150  seed 11: ok         fwd 1.469e-03 min_rel/u =    316   trip@100u: no  trip@300u: no
  naive-F16  kappa 150  seed 11: ok         fwd 3.342e-02 min_rel/u =    324   trip@100u: no  trip@300u: no
  mixed-F16  kappa 150  seed 12: inaccurate fwd 1.771e-03 min_rel/u =    285   trip@100u: no  trip@300u: yes
  naive-F16  kappa 150  seed 12: ok         fwd 2.964e-02 min_rel/u =    292   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 200  seed 10: inaccurate fwd 2.014e-03 min_rel/u =    251   trip@100u: no  trip@300u: yes
  naive-F16  kappa 200  seed 10: ok         fwd 4.071e-02 min_rel/u =    254   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 200  seed 11: inaccurate fwd 2.891e-03 min_rel/u =    263   trip@100u: no  trip@300u: yes
  naive-F16  kappa 200  seed 11: ok         fwd 4.152e-02 min_rel/u =    271   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 200  seed 12: inaccurate fwd 1.779e-03 min_rel/u =    234   trip@100u: no  trip@300u: yes
  naive-F16  kappa 200  seed 12: ok         fwd 3.506e-02 min_rel/u =    240   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 300  seed 10: inaccurate fwd 2.804e-03 min_rel/u =    191   trip@100u: no  trip@300u: yes
  naive-F16  kappa 300  seed 10: ok         fwd 5.331e-02 min_rel/u =    198   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 300  seed 11: inaccurate fwd 2.745e-03 min_rel/u =    201   trip@100u: no  trip@300u: yes
  naive-F16  kappa 300  seed 11: ok         fwd 5.750e-02 min_rel/u =    206   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 300  seed 12: inaccurate fwd 2.786e-03 min_rel/u =    176   trip@100u: no  trip@300u: yes
  naive-F16  kappa 300  seed 12: ok         fwd 4.518e-02 min_rel/u =    185   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 500  seed 10: inaccurate fwd 4.572e-03 min_rel/u =    134   trip@100u: no  trip@300u: yes
  naive-F16  kappa 500  seed 10: ok         fwd 8.286e-02 min_rel/u =    136   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 500  seed 11: inaccurate fwd 4.228e-03 min_rel/u =    142   trip@100u: no  trip@300u: yes
  naive-F16  kappa 500  seed 11: ok         fwd 7.098e-02 min_rel/u =    144   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 500  seed 12: inaccurate fwd 4.754e-03 min_rel/u =    122   trip@100u: no  trip@300u: yes
  naive-F16  kappa 500  seed 12: ok         fwd 7.425e-02 min_rel/u =    127   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 700  seed 10: inaccurate fwd 5.050e-03 min_rel/u =    106   trip@100u: no  trip@300u: yes
  naive-F16  kappa 700  seed 10: ok         fwd 9.300e-02 min_rel/u =    107   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 700  seed 11: inaccurate fwd 7.354e-03 min_rel/u =    112   trip@100u: no  trip@300u: yes
  naive-F16  kappa 700  seed 11: ok         fwd 9.345e-02 min_rel/u =    109   trip@100u: no  trip@300u: yes
  mixed-F16  kappa 700  seed 12: inaccurate fwd 7.127e-03 min_rel/u =     95   trip@100u: yes trip@300u: yes
  naive-F16  kappa 700  seed 12: inaccurate fwd 1.057e-01 min_rel/u =     97   trip@100u: yes trip@300u: yes
```

What it shows.
- **On the E5 grid c=100 separates the runs perfectly (105 of 105 non-ok flagged, 0 of 15 ok low-precision runs flagged). That is a coincidence of the grid.** The E5 grid has nothing between kappa=100 (min_rel about 400u) and kappa=1000 (about 90u). The held-out check (E7c, kappa 150-700, seeds 10-12) shows that at c=100 the guard misses 13 of the 14 mixed-FP16 runs that the 1000-eps rule labels inaccurate (forward error 1.7e-3 to 7e-3), while c=300 catches 13 of 14. For naive FP16, c=300 flags runs labelled ok with forward error 0.03-0.09. So there is no "right" c: it is the accuracy requirement, and my status labels (1000 eps, forward error 0.1) are not a ground truth for it.
- **The guard fires earlier than the exception.** mixed-F16 kappa=1e6: guard at pivot 83, first non-positive pivot 193, the first NaN later still; naive-F16 kappa=1e4: guard at 147, breakdown at 253; `A_delta` FP16 update: guard at pivot 1 (relative pivot 4.9e-4, which is the true small pivot of this ill-conditioned matrix), breakdown at 33.
- **On the guarded runs NixNan reports no NaN at all** (A_delta: 522 FP32 NaN and 4 NaN stores without the guard, none with it; mixed-F16 kappa=1e6: 513 FP32 NaN without, none with). What remains are the FP16 subnormals of the panel (the same counts as in E5): the guard removes the symptom NixNan sees, not those reports.
- **The guard sees what NixNan cannot:** mixed-BF16 kappa=1e2 and mixed-TF32 kappa=1e3 are inaccurate with no NixNan exception; the guard stops both (pivot 133 and 247). It does not trip on the accurate FP32 run at kappa=1e7 (min_rel 1.4e-5 > 100 u = 6e-6) and not on mixed-F16 kappa=1e2.
- **It does not repair anything**; it turns "NaN, or a quietly wrong answer" into "stop at pivot i, ill-conditioned for this precision". The remedy is a wider format or iterative refinement (`../mpir_gpu`).

Caveats. tau was picked after looking at the E5 data; E7c is the only out-of-sample check and it shows c=100 is too permissive for the 1000-eps rule. One matrix family and size. `min_rel_pivot` is taken over the whole run, including pivots computed after a failure (it is 0 for breakdown runs). The guard test uses the original diagonal of the matrix as stored (after rounding for the storage experiments); it is one common choice of reference (others: the matrix norm, as in the plan, or the running maximum diagonal) and I did not compare them. Unblocked E7 kernels check after each pivot; in the blocked kernels the check is per tile, and within a tile the stop is immediate (the tile kernel returns).
