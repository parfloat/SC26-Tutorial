# All Runs

A chronological record of every nixnan-instrumented (and directly-relevant
baseline) execution performed in this session, across all case studies.
Each entry has enough detail -- exact command, environment variables, and
working directory -- to turn directly into a shell script. Every command
assumes `LD_PRELOAD` points at the built `nixnan.so`; paths below are
relative to each case study's own directory (e.g. `../external/nixnan/nixnan.so`
from `tc_matmult/`, `lu_solve_gpu/`, etc.).

Tool locations referenced throughout:
- `external/nixnan/nixnan.so` -- the instrumentation library
- `external/nixnan/bin/autonixnan` -- upstream 3-phase wrapper (baseline / extreme-value scan / deep dive)
- `external/nixnan/bin/iter-autonixnan` -- new in this session: whitelist every kernel in turn, one per run
- `external/nixnan/bin/ksel-autonixnan` -- new in this session: `-phase1` writes a `ksel` selection file (`name : y/n ;` per kernel), `-ksel` whitelists every `y`-marked kernel together in one run

---

## tc_matmult/

Benchmark kernels from Lei Mao's Tensor Core blog post (`leimao-mm.cu`) plus
SIMT/BF16 variants written for this tutorial; `*-128x128.cu` size-reduced
copies for tractable `nixnan` observation; `*-overflow.cu` variants that
deliberately force FP16 overflow.

### Uninstrumented benchmarks (no nixnan)
```bash
cd tc_matmult
./leimao-mm                       # 1024^3, FP16 HMMA vs (commented-out) INT8
./leimao-simt-variant-fp32        # 1024^3, FP32 SIMT
./leimao-simt-variant-fp16        # 1024^3, FP16 SIMT
./leimao-tc-variant-bf16          # 1024^3, BF16 tensor core
```
Results (in `report.tex`): FP16 TC ~0.178ms vs FP16 SIMT ~0.865ms
(~4.9x); FP32 SIMT ~0.845ms vs BF16 TC ~0.177ms (~4.8x).

### 128^3 baseline (no nixnan)
```bash
cd tc_matmult
./leimao-mm-128x128 > Logs/leimao-mm-128x128-baseline-no-nixnan.log 2>&1
```
Result: `HMMA Latency: 0.008 ms` (row-major x row-major).

### nixnan overhead sweep, leimao-mm-128x128 (FP16 HMMA)
```bash
cd tc_matmult
LIB=../external/nixnan/nixnan.so

# (a) SAMPLING=128 + histogram(count:1,doublings:6) -- misleadingly cheap,
#     only launch #0 of ~88 ever instrumented
SAMPLING=128 HISTOGRAM=1 \
  BIN_SPEC_FILE=Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.json \
  LD_PRELOAD=$LIB ./leimao-mm-128x128 \
  > Logs/Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.log 2>&1
# JSON: {"count":1,"doublings":6,"f16":[[-9,-8],[7,8]]} (ranges chosen
# empirically from a plain HISTOGRAM=1 run's "Exponent range for f16" line)

# (b) unsampled, default exception checking only
LD_PRELOAD=$LIB ./leimao-mm-128x128 \
  > Logs/leimao-mm-128x128-unsampled-default-exceptions.log 2>&1

# (c) unsampled, exceptions + full count:1,doublings:6 histogram (real cost)
HISTOGRAM=1 BIN_SPEC_FILE=Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.json \
  LD_PRELOAD=$LIB ./leimao-mm-128x128 \
  > Logs/leimao-mm-128x128-unsampled-count1-doublings6-SUMMARY.log 2>&1
  # (original was 1,054,204 lines/336MB; committed log is truncated with
  # full aggregate stats preserved)
```
Results (row-major x row-major HMMA latency): baseline 0.008ms (1x);
(a) 0.015ms (~1.9x, misleading); (b) 0.706ms (~88.3x); (c) 257.009ms
(~32,126x). Zero exceptions in every run (workload never reaches NaN/Inf
at this magnitude).

### Forced FP16 overflow (main-vs-overflow variants)
```bash
cd tc_matmult
LIB=../external/nixnan/nixnan.so

# Tensor core: float accumulator (control) vs __half accumulator, omega N/A
# here -- input magnitude raised to uniform(-2000,2000) inside the .cu itself
LD_PRELOAD=$LIB ./leimao-mm-128x128-overflow \
  > Logs/leimao-mm-128x128-overflow-nixnan.log 2>&1

# SIMT, pure __half accumulate (no float promotion, no WMMA at all)
LD_PRELOAD=$LIB ./leimao-simt-variant-fp16-128x128-overflow \
  > Logs/leimao-simt-variant-fp16-128x128-overflow-nixnan.log 2>&1
```
Results: TC float accumulator never overflows (0/16384 non-finite, every
combo); TC `__half` accumulator overflows completely (100% non-finite,
pure +-Inf); SIMT pure-`__half` also 100% non-finite. nixnan catches both
at the instruction level: `HMMA.16816.F16` (TC, narrow accumulator) and
`HFMA2` (SIMT, ordinary CUDA-core instruction, zero tensor-core
involvement) -- the cleanest possible confirmation that plain SIMT FP16
adds overflow too, not just tensor cores.

---

## rd_nixnan/

Two distinct source files: `reaction_diffusion_gpu.cu` (FP64/FP32 only,
`-DRD_USE_FLOAT` toggle, no CLI args, fixed 80,000-step PDE solve) and
`rd_nixnan.cu` (genuine FP16/BF16/FP32/FP64 kernels, `-DFP64_LONGRUN=1`
build mode interleaves all four for 3800 steps then suppresses the three
lower-precision ones and runs FP64 alone).

### Build
```bash
cd rd_nixnan
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -o reaction_diffusion_gpu_fp64 reaction_diffusion_gpu.cu
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -DRD_USE_FLOAT -o reaction_diffusion_gpu_fp32 reaction_diffusion_gpu.cu
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -DFP64_LONGRUN=1 -o rd_nixnan_fp64_longrun rd_nixnan.cu
```

### iter-autonixnan sweeps (this session's new tool)
```bash
cd rd_nixnan
TOOL=../external/nixnan/bin/iter-autonixnan

# bin_spec_highest_c2d4.json: highest binade alone per format
# (bf16/f32=127, f16=15, f64=1023), count=2, doublings=4
cat > bin_spec_highest_c2d4.json <<'EOF'
{
    "count": 2,
    "doublings": 4,
    "max_reports": 0,
    "bf16": [[127, 127]],
    "f16": [[15, 15]],
    "f32": [[127, 127]],
    "f64": [[1023, 1023]]
}
EOF

# reaction_diffusion_gpu.cu: 2 kernels (rd_step_kernel, max_reduce_kernel),
# no sampling (-s 0), histogram on via -b
$TOOL -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./reaction_diffusion_gpu_fp32 \
  > iter_autonixnan/sampling0-histogram-c2d4-fp32.log 2>&1
$TOOL -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./reaction_diffusion_gpu_fp64 \
  > iter_autonixnan/sampling0-histogram-c2d4-fp64.log 2>&1

# rd_nixnan_fp64_longrun: 4 kernels (rd_step_fp16/bf16/fp32/fp64)
$TOOL -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./rd_nixnan_fp64_longrun \
  > iter_autonixnan/sampling0-histogram-c2d4-fp64longrun.log 2>&1
```
Results: `reaction_diffusion_gpu_fp32` -- 13,773 threshold reports at
exponent 127 as the solution climbs toward overflow, real exceptions (36
distinct NaN sites/553,859 repeats, 16 Infinity/92 repeats, 40
-Infinity/1,176 repeats). `reaction_diffusion_gpu_fp64` -- 0 threshold
reports (peak ~1.7e43, exponent ~144, nowhere near 1023), 0 exceptions.
**FP16/BF16 never appear in either -- `reaction_diffusion_gpu.cu` has no
half-precision code path at all** (confirmed via `grep -c "__half" reaction_diffusion_gpu.cu` = 0).
`rd_nixnan_fp64_longrun` (the *other* source file, which does have
FP16/BF16 kernels) -- all 4 formats genuinely exercised: `rd_step_fp16`
22 NaN sites (157,334 repeats), `rd_step_bf16` 15 NaN sites (95,375
repeats), `rd_step_fp32` 19 NaN sites (118,966 repeats), `rd_step_fp64` 36
NaN sites (1,699 repeats, far fewer since FP64 only goes non-finite right
at the very end, step 14,600, vs. the others overflowing early ~step 300
and propagating NaN for thousands of subsequent steps).

### Earlier work (first half of this session, `Expts-9-23/`)
Extensive prior work in `rd_nixnan/Expts-9-23/` (not re-detailed here;
see `rd_nixnan/Expts-9-23/latex/atomicCas_Stdout_Tearing_fixes.tex` and
`rd_nixnan/rd_nixnan_report.tex` for full writeups) included: fixing a
`double->float` narrowing bug in `rd_nixnan.cu`; adding the
`FP64_LONGRUN` build mode itself; porting and fixing (via `atomicCAS`) the
adaptive `doublings` threshold feature; finding and fixing a torn/
interleaved-stdout bug between nixnan's exception and histogram report
threads (`nnout_line()` mutex helper). Shell scripts from that phase
(`Exp-Output-Rate-start-1-double-8-times.sh`,
`Stop_First_Error_with_History_FP16*.sh`, etc.) live under
`rd_nixnan/Expts-9-23/shell-scripts/`.

---

## lu_solve_gpu/

`lu_solve_gpu.cu`: LU factorization with partial pivoting, 20x20,
`matrix.csv` (well-conditioned) / `bad_matrix.csv` (ill-conditioned, a
near-zero pivot snaps to exactly 0.0, division by it produces Inf/NaN).
8 kernels, 4 of them (`find_pivot_kernel`, `swap_rows_kernel`,
`compute_multipliers_kernel`, `eliminate_kernel`) launched 19 times each
(one per pivot step).

### Build
```bash
cd lu_solve_gpu
make   # nvcc -arch=sm_86 -lineinfo -o lu_solve_gpu lu_solve_gpu.cu
```

### TIME_KERNELS (nothing else set)
```bash
cd lu_solve_gpu
LD_PRELOAD=../external/nixnan/nixnan.so TIME_KERNELS=1 ./lu_solve_gpu matrix.csv
```
Result: 82 `Kernel [...] execution time:` lines. First launch of every
distinct kernel function costs ~280-330ms (one-time SASS-instrumentation-
insertion cost); every later launch of an already-seen kernel costs
~20-25us -- a ~12,000-15,000x gap. Documented with a full table in
`autonixnan-doc.tex` Sec 2.8.

### LOG_KERNELS only (confirms no first-launch cost when uninstrumented)
```bash
cd lu_solve_gpu
LD_PRELOAD=../external/nixnan/nixnan.so LOG_KERNELS=/path/to/log.txt ./lu_solve_gpu matrix.csv
```
Result: no ~300ms spike anywhere (LOG_KERNELS disables all
instrumentation). Documented in `autonixnan-doc.tex` Sec 2.7.

### Plain HISTOGRAM=1 (no BIN_SPEC_FILE)
```bash
cd lu_solve_gpu
LIB=../external/nixnan/nixnan.so
LD_PRELOAD=$LIB HISTOGRAM=1 ./lu_solve_gpu matrix.csv     > logs/histogram1-matrix.log 2>&1
LD_PRELOAD=$LIB HISTOGRAM=1 ./lu_solve_gpu bad_matrix.csv > logs/histogram1-bad_matrix.log 2>&1
```
Result: `matrix.csv` -- "Exponent range for f32/f64: [zero, 1]".
`bad_matrix.csv` -- "Exponent range for f32/f64: [zero, inf]" --
**the unconditional summary alone signals Inf was reached, with zero bin
configuration needed** -- a real signal `autonixnan`'s Phase 2 scan misses
entirely (see below).

### bin_spec_highest.json (highest binade alone, all 4 formats, record_inst)
```bash
cd lu_solve_gpu
cat > bin_spec_highest.json <<'EOF'
{
    "count": 1,
    "doublings": 6,
    "max_reports": 0,
    "bf16 (record_inst)": [[127, 127]],
    "f16 (record_inst)": [[15, 15]],
    "f32 (record_inst)": [[127, 127]],
    "f64 (record_inst)": [[1023, 1023]]
}
EOF
LIB=../external/nixnan/nixnan.so
LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_highest.json ./lu_solve_gpu matrix.csv     > logs/bin_spec_highest-matrix.log 2>&1
LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_highest.json ./lu_solve_gpu bad_matrix.csv > logs/bin_spec_highest-bad_matrix.log 2>&1
```
Result: **zero threshold reports on either input**, even `bad_matrix.csv`
-- the division-by-zero failure jumps straight from finite to Inf,
skipping the highest finite binade entirely (matches
`autonixnan-doc.tex`'s documented histogram blind spot: watched bins
structurally exclude the zero/infinity exponent codes).

### bin_spec_whole.json (every format, full range, record_inst, count=4 doublings=10)
```bash
cd lu_solve_gpu
cat > bin_spec_whole.json <<'EOF'
{
    "count": 4,
    "doublings": 10,
    "max_reports": 0,
    "bf16 (record_inst)": [[-126, 127]],
    "f16 (record_inst)": [[-14, 15]],
    "f32 (record_inst)": [[-126, 127]],
    "f64 (record_inst)": [[-1022, 1023]]
}
EOF
LIB=../external/nixnan/nixnan.so
LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_whole.json ./lu_solve_gpu matrix.csv     > logs/bin_spec_whole-matrix.log 2>&1
LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_whole.json ./lu_solve_gpu bad_matrix.csv > logs/bin_spec_whole-bad_matrix.log 2>&1

# ... and again with SAMPLING=16
SAMPLING=16 LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_whole.json ./lu_solve_gpu matrix.csv     > logs/bin_spec_whole-sampling16-matrix.log 2>&1
SAMPLING=16 LD_PRELOAD=$LIB BIN_SPEC_FILE=bin_spec_whole.json ./lu_solve_gpu bad_matrix.csv > logs/bin_spec_whole-sampling16-bad_matrix.log 2>&1
```
Result: 131 threshold reports (`matrix.csv`), 111 (`bad_matrix.csv`)
unsampled; 105 / 97 under `SAMPLING=16` -- a real but modest ~20%
reduction, much smaller than the ~85-90% a naive "2 of 19 launches
enabled" reading would predict (open question, not resolved).

### autonixnan (upstream 3-phase wrapper)
```bash
cd lu_solve_gpu
../external/nixnan/bin/autonixnan -- ./lu_solve_gpu matrix.csv     > logs/autonixnan-matrix.log 2>&1
../external/nixnan/bin/autonixnan -- ./lu_solve_gpu bad_matrix.csv > logs/autonixnan-bad_matrix.log 2>&1
```
Result: **both report "No extreme exponent values detected in any
regime"** -- `autonixnan` misses `bad_matrix.csv`'s real exceptions (316
NaN / 4 Infinity / 4 Division-by-zero, confirmed separately) entirely,
same histogram-blind-spot mechanism as `bin_spec_highest.json` above.

### iter-autonixnan
```bash
cd lu_solve_gpu
TOOL=../external/nixnan/bin/iter-autonixnan
$TOOL -s 4 -- ./lu_solve_gpu matrix.csv \
  > logs/iter-autonixnan/sampling4-matrix.log 2>&1
$TOOL -s 8 -b bin_spec_highest.json -- ./lu_solve_gpu bad_matrix.csv \
  > logs/iter-autonixnan/sampling8-binspec_highest-bad_matrix.log 2>&1
```
Result: all 8 kernels visited in succession in both runs; each run's own
banner confirms `SAMPLING`/`BIN_SPEC_FILE` correctly applied.

### ksel-autonixnan (validation only; no committed logs from lu_solve_gpu)
Used to validate the tool itself: `-phase1` against `matrix.csv` writes
an 8-kernel `ksel` file (all `n`); hand-editing two to `y`
(`find_pivot_kernel`, `eliminate_kernel`) and running `-ksel -s 4`
confirmed (via `TOOL_VERBOSE=1`, tracking `Instrumenting instruction`
lines back to their owning `Inspecting function`) that only those two
kernels actually receive instrumented instructions.

---

## mpir_gpu/

`Code2.cu` / `ir_gpu`: mixed-precision (Carson-Higham) iterative
refinement Cholesky solver. No fixed CLI args -- `--n`, `--kappa`,
`--config` (F64/F32/TF32/F16/B16T/B16S), etc. This session used a small,
fast, convergent config throughout: `--n 256 --kappa 100 --config F32`
(converges in 1 iteration, ~53ms loop time).

### iter-autonixnan
```bash
cd mpir_gpu
../external/nixnan/bin/iter-autonixnan -s 4 -t 90 -- ./ir_gpu --n 256 --kappa 100 --config F32 \
  > iter-autonixnan/sampling4-n256-kappa100-F32.log 2>&1
```
Result: 27 distinct kernels visited (cuSOLVER/cuBLAS/cutlass templates
among them), no timeouts. Real exceptions found in
`geqr2_smem_domino_fast<double, double, 8, 512>` (the QR factorization
kernel used in `make_problem()` for test-matrix generation): 22 distinct
NaN sites (154 repeats), 1 Infinity (7 repeats), 1 Division-by-zero (8
repeats) -- found directly, with no exponent-scan guesswork, exactly the
kernel `autonixnan-doc.tex`'s Empirically Observed Limitations section
already discusses as `autonixnan` Phase 2's blind spot.

### ksel-autonixnan
```bash
cd mpir_gpu
TOOL=../external/nixnan/bin/ksel-autonixnan
mkdir -p ksel-autonixnan
$TOOL -phase1 -- ./ir_gpu --n 256 --kappa 100 --config F32   # writes ./ksel (27 kernels, all 'n')
# --- hand-edit ksel, flip desired kernels to 'y' ---
$TOOL -ksel -s 4 -t 90 -- ./ir_gpu --n 256 --kappa 100 --config F32 \
  > ksel-autonixnan/sampling4-n256-kappa100-F32.log 2>&1
```
Two selections tried (both committed as separate runs):
1. 6 kernels (3 `cutlass::Kernel2` GEMM variants, `xxtrf4_set_info_ker`,
   `dtrsv_init_up`, `axpy_kernel_val<float,float>`) -- zero exceptions,
   3.5s.
2. 4 kernels (`geqr2_smem_domino_fast`, `cutlass::Kernel2<...tn_align1>`,
   `dtrsv_init_up`, `axpy_kernel_val<float,float>`) -- same real
   exceptions as the `iter-autonixnan` sweep (22 NaN/154 repeats, 1
   Infinity/7 repeats, 1 Division-by-zero/8 repeats), confirming they are
   attributable to `geqr2_smem_domino_fast` specifically -- logged as
   `mpir_gpu/ksel-autonixnan/sampling4-n256-kappa100-F32-with-geqr2.log`.

---

## hec_atan2/ and hec_fluidSim/

Local test copies of HeCBench's `atan2-cuda` and `fluidSim-cuda`
(`external/HeCBench` is a submodule with no push access, so these are
standalone copies with their own `README.md` for provenance/licensing).

### hec_atan2 (MakeShort: 8 coordinates, repeat once)
```bash
cd hec_atan2
make -f MakeShort          # builds ./main
LD_PRELOAD=../external/nixnan/nixnan.so ./main 8 1 \
  > Log/atan2-8x1-nixnan-default-exceptions.log 2>&1
```
Result: zero runtime exceptions (RMSE 0.000000, all 3 kernels). One
compile-time-immediate warning: `NaN immediate found in operand 1 of
MUFU.RSQ R9, -QNAN` -- a literal QNaN bit pattern baked into the SASS,
classified statically at instrumentation time, not a runtime event.
(`atan2(y,x)`'s only real NaN hazard, `(x,y)=(0,0)`, is unreachable by
this program's own random-input generator.)

### hec_fluidSim (MakeOverflow: omega=1e8, deliberately unstable)
```bash
cd hec_fluidSim
make -f MakeOverflow        # builds ./main_overflow (no -DVERIFY)
LD_PRELOAD=../external/nixnan/nixnan.so ./main_overflow 60 \
  > Log/fluidSim-overflow-60iter-nixnan-default-exceptions.log 2>&1
```
Result: 580,644/589,824 output cells NaN (98.4%, standalone scan).
nixnan: 1,249 distinct FP64 NaN sites (16.7M repeats) in the kernel's own
`DFMA`/`DMUL`/`DADD`. Two binary-level false positives found in the same
run, documented in `hec_fluidSim/fluidsim-report.tex`: an FP32 one
matching `lu_solve_gpu`'s already-documented `ptxas` register-move-idiom
finding exactly, and a distinct, only-partially-diagnosed FP16 anomaly (a
`DFMA` instruction -- confirmed `f64`-only in nixnan's own instruction
table -- reported with an `f16` label).

---

## external/nixnan/ (the tool itself)

Two new scripts added to `bin/` this session, both on branch
`print_histo_instrn` (`parfloat/nixnan`, separate repo, submoduled in
here as `external/nixnan/`):

- **`bin/iter-autonixnan`** (`-s SAMPLING` required, `-t TIMEOUT`,
  `-b BIN_SPEC_FILE` optional): Phase 1 collects every kernel name
  (`LOG_KERNELS`); Phase 2 whitelists each one **in turn**, one
  fully-separate run per kernel, `INSTRUMENT_EXCEPTIONS` always forced on.
- **`bin/ksel-autonixnan`** (`-phase1` / `-ksel` mutually exclusive,
  `--ksel-file` default `./ksel`): `-phase1` writes a `ksel` file (every
  kernel `name : n ;`, one per line); after hand-editing some to `y`,
  `-ksel -s SAMPLING [-b ...]` whitelists every `y`-marked kernel
  **together** in one combined run.

Also corrected in `autonixnan-doc.tex` this session (both now stale-doc
fixes, verified directly against the current source rather than trusted
from the existing text):
- `NIXNAN_TIMEOUT` is active (re-enabled by a `Re-enable timer` commit
  since the doc's original "dead" finding) -- confirmed via
  `NIXNAN_TIMEOUT=3 LD_PRELOAD=nixnan.so ./target`, which genuinely
  terminates at ~3.5s.
- `SAMPLING` is no longer checked inside `should_instrument()`; it is now
  the boolean argument to `nvbit_enable_instrumented` in the launch
  callback (`sampling > 0 ? kernel_count % sampling == 0 : true`), which
  also means the old `lu_solve_gpu_report.tex`-documented key-mismatch bug
  no longer exists in this code path.

---

## pytorch_slogdet/

Reproducer for [pytorch/pytorch#173638](https://github.com/pytorch/pytorch/issues/173638)
("`torch.linalg.slogdet` does not propagate `NaN` in CUDA"). `reproducer.py`
skips the CPU call by default (`SKIP_CPU=1`); see `README.md` in that
directory for the full issue text and this environment's own reproduction
result (torch 2.3.1+cu121 here correctly propagates the NaN -- the
silent-corruption bug is reproduced in the issue's own newer
2.10.0+cu128 environment, not this one).

### Direct (non-iter) attempts, short timeout -- both looked stalled
```bash
cd pytorch_slogdet
timeout 120 env LD_PRELOAD=../external/nixnan/nixnan.so python3 reproducer.py               # killed at 120s
timeout 120 env LD_PRELOAD=../external/nixnan/nixnan.so SAMPLING=512 python3 reproducer.py  # same wall, killed
```
Both stuck mid-SASS-rewrite on a large cuSOLVER `getrf`-family kernel
before the target program ever printed a line of output (documented in
`autonixnan-doc.tex` Sec 2.8 as a real example of the one-time
instrumentation-insertion cost). `SAMPLING` doesn't help here, since it
gates per-launch report volume, not the one-time rewrite cost, and the
reproducer only launches each kernel once or twice. Logs from these two
attempts were kept out of `git` (confirmed truncated mid-instrumentation,
never reaching the program's own output).

### iter-autonixnan -- succeeded where the short-timeout direct attempts stalled
```bash
cd pytorch_slogdet
../external/nixnan/bin/iter-autonixnan -s 8 -t 60 -- python3 reproducer.py \
  > Logs/iter-autonixnan.log 2>&1
```
Result: all 19 kernels visited, no timeouts (~32.5s each -- almost
entirely PyTorch's own CUDA/cuSOLVER startup cost, confirmed against
Phase 1's own 32.16s baseline, not nixnan overhead; ~10 minutes total
across 19 separate process launches). Directly traces the real NaN
propagation path through `slogdet`'s CUDA kernels: `getrf_pivot`
(cuSOLVER's LU factorization -- 7 distinct NaN sites/5 repeats) ->
`sign_kernel_cuda` (2, then 1 distinct NaN site) -> `log_kernel_cuda`
(2 NaN sites, 1 Infinity site) -> a final `reduce_kernel` (10 distinct
NaN sites).

### Direct (non-iter), unsampled, no timeout cap -- turns out it does finish
```bash
cd pytorch_slogdet
LD_PRELOAD=../external/nixnan/nixnan.so python3 reproducer.py \
  > Logs/slogdet-nixnan-unsampled-default-exceptions.log 2>&1
```
The exact invocation that was killed at 120s above, this time given a
long background allowance instead: **it genuinely completes** -- all 19
kernels instrumented together in one process, reaches the program's own
output and nixnan's final report (832 lines). 21 distinct FP32 NaN sites
(5 repeats), 1 Infinity site, 1 memory NaN -- consistent in magnitude
with `iter-autonixnan`'s per-kernel sweep (22 total NaN sites split
across the same four kernels). Resolves the earlier open question: the
one-time instrumentation-insertion cost for cuSOLVER's large kernels
here is *slow* (well past 120s), not an actual hang.
