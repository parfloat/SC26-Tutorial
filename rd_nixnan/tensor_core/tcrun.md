# rd_nixnan on tensor cores (FP16 / BF16): status record, 2026-10-19

## Goal

Run the rd_nixnan reaction-diffusion demo on the **tensor cores** in FP16 or
BF16, and see what nixnan reports when the values overflow inside or around
the `HMMA` instructions.

## Design: 2D problem, one time step = two GEMMs (`rd_tc.cu`)

The 1D stencil in `../rd_nixnan.cu` does not map to tensor cores, so the
problem is moved to 2D, where the FTCS step becomes a matrix product.

    du/dt = D (u_xx + u_yy) + lambda u   on the unit square, u = 0 on the edge
    u(x,y,0) = sin(pi x) sin(pi y)

* Store the N x N interior as a matrix U (N = 64, dx = 1/65).
* The second difference is T = tridiag(1, -2, 1). Because T has no entries
  outside the interior, the zero boundary is built in.
  * u_xx is approximately T U / dx^2.
  * u_yy is approximately U T / dx^2.
* One step, U' = U + r (T U + U T) + (lambda dt) U, is exactly

      U' = P U + U P,    P = ((1 + lambda dt)/2) I + r T

* Parameters:
  * D = 0.01, lambda = 50, dt = 1e-3 (the same as rd_nixnan.cu).
  * r = D dt / dx^2 = 0.04225 (stable if <= 0.25 in 2D).
  * The (1,1) mode grows by 1.049803 per step.
* P has 0.44050 on the diagonal (0.44043 in FP16, 0.44141 in BF16) and
  0.04225 off the diagonal.
* Kernel layout:
  * One warp per 16x16 output tile, so the grid is 4x4 blocks of 32 threads.
  * Each warp does `TILES` WMMA m16n16k16 `mma_sync` calls for P·U and
    `TILES` more for U·P, all into one accumulator.
  * After each step the host copies U back and counts inf and NaN explicitly,
    instead of relying on a `v > max` reduction, which silently skips NaN
    (the wrinkle documented in ../README.md).

Three modes (`./rd_tc fp16acc16|fp16acc32|bf16acc32|all`):

| mode | inputs → accumulator | how the result is stored |
|---|---|---|
| `fp16acc16` | FP16 → FP16 | `store_matrix_sync` straight from the FP16 accumulator |
| `fp16acc32` | FP16 → FP32 | FP32 to shared memory, then rounded to FP16 (`F2FP.PACK`) |
| `bf16acc32` | BF16 → FP32 | FP32 to shared memory, then rounded to BF16 (`F2FP.BF16.PACK`) |

Build:

    nvcc -std=c++17 -O2 -lineinfo -arch=sm_86 rd_tc.cu -o rd_tc

The SASS confirms that the tensor cores are used
(`cuobjdump -sass rd_tc | grep -oE 'HMMA[.A-Z0-9]*|F2FP[.A-Z0-9]*' | sort | uniq -c`):

    16 HMMA.16816.F16
    16 HMMA.16816.F32
    16 HMMA.16816.F32.BF16
     5 F2FP.PACK
     5 F2FP.BF16.PACK

nixnan's `src/instructions.json` has register-type entries for all three
HMMA variants (`HMMA.16816.F16`, `HMMA.16816.F32`, `HMMA.16816.F32.BF16`), so
they should be instrumented once NVBit works (see below).

## Native results (no nixnan), RTX 3090, CUDA 12.4

| mode | step 100 max | step 200 max | first inf | next step | all 4096 NaN |
|---|---|---|---|---|---|
| fp16acc16 | 1.2669e2 | 1.6024e4 | step 230: 120 inf, max finite 6.528e4 | step 231: 1392 NaN | step 232 |
| fp16acc32 | 1.2669e2 | 1.6032e4 | step 230: 120 inf, max finite 6.544e4 | step 231: 1392 NaN | step 232 |
| bf16acc32 | 1.5300e2 | 2.3552e4 | step 1764: 76 inf, max finite 3.3895e38 | step 1765: 4 inf + 1180 NaN | step 1766 |

These match the scalar rd_nixnan.cu timeline: FP16 overflowed there at about
step 227 and BF16/FP32 at about step 1818.

### Finding: 0 × inf = NaN spreads in a single step on the tensor core

* In the stencil code, one inf reaches only its 2 or 3 neighbours per step.
* Here P is a dense operand that is mostly zeros, and the tensor core computes
  every product, including 0 × inf = NaN.
* So any inf in U poisons its whole row (through P·U) and its whole column
  (through U·P) of U' in one step: 120 inf become 1392 NaN, and the step after
  that all 4096 entries are NaN.
* This is specific to doing the stencil as a dense GEMM. It is a good
  tensor-core lesson for the tutorial.

### Expected nixnan picture (to verify after the reboot)

* `fp16acc16`: the inf should be born inside the tensor core, in the
  destination of `HMMA.16816.F16`. The NaN should then appear in the next
  step's HMMA, from inf operands meeting the zeros of P.
* `fp16acc32`: the FP32 accumulator never overflows. The inf should be born at
  the FP32→FP16 conversion (`F2FP.PACK`) and show up as an inf operand of the
  next `HMMA.16816.F32`, followed by the NaN.
* `bf16acc32`: the same as fp16acc32, at about step 1764, with
  `HMMA.16816.F32.BF16` and `F2FP.BF16.PACK`.

## Blocker found: NVBit does not attach on this machine (2026-10-19)

nixnan printed its banner but **no reports and no summary** for any run:

* `rd_tc` in all three modes, under `SC26-Tutorial/external/nixnan/nixnan.so`
  (`LINE_INFO=1 PRINT_ILL_INSTR=1`, with and without `LOGFILE`);
* the original `../rd_nixnan` binary, whose FP16 run definitely overflows. It
  worked before: see the Sep 27–28 logs in `../iter_autonixnan/`;
* `rd_tc` under `~/repos/nixnan/nixnan.so` (branch `print_histo_instrn`). This
  is the same commit as external/nixnan, `ce35185`, with an older `.so` build
  from Sep 18.

The fault is in NVBit itself, not nixnan.

* NVBit 1.8's own sample tool, `tools/instr_count` (built from
  `external/nixnan/nvbit_release`), loads and prints its banner, but never
  sees a single kernel launch.
* None of these changed anything:
  * `NO_EAGER_LOAD=0`
  * `CUDA_MODULE_LOADING=EAGER`
  * building with `-cudart shared`
  * clearing `LD_LIBRARY_PATH`, which contains `/home/ganesh/dyninst-install/lib`
    and a missing `/usr/local/cuda-13.2/lib64`
* `TOOL_VERBOSE=1` shows no kernel inspection at all.

Cause: a driver upgrade without a reboot.

* `/proc/driver/nvidia/version`: the kernel module is **595.84**.
* `/usr/lib/x86_64-linux-gnu/libcuda.so.1` points to **595.99.02**.
* `dpkg.log` shows the 595.91.07 → 595.99.02 package upgrades on 2026-10-09.
  The machine had been up 55 days, so the old module stayed loaded.
* `nvidia-smi` fails with "Failed to initialize NVML: Driver/library version
  mismatch".
* The kernel log has: `NVRM: API mismatch: the client 'nvidia-smi' has the
  version 595.99.02, but this kernel module has the version 595.84`.
* Plain CUDA programs still run, which is why rd_tc works natively, but
  NVBit's driver hook silently never fires.

Fix: reboot, or reload the nvidia kernel module (this needs root). Afterwards,
`nvidia-smi` should work again.

## To do after the reboot

1. Check that `nvidia-smi` works.
2. Check that NVBit attaches. instr_count should print one line per kernel:

       LD_PRELOAD=<nvbit_release>/tools/instr_count/instr_count.so ./rd_tc fp16acc16

3. Check nixnan on the old binary:
   `LD_PRELOAD=~/repos/nixnan/nixnan.so ../rd_nixnan` should report FP16/FP32
   inf and NaN as before.
4. Run the tensor-core modes under `~/repos/nixnan` (`print_histo_instrn`):

       cd rd_nixnan/tensor_core
       for m in fp16acc16 fp16acc32 bf16acc32; do
         LINE_INFO=1 PRINT_ILL_INSTR=1 LOGFILE=logs/nixnan-$m.log \
           LD_PRELOAD=~/repos/nixnan/nixnan.so ./rd_tc $m > logs/stdout-$m.txt 2>&1
       done

   Then run a histogram pass (`HISTOGRAM=1 BIN_SPEC_FILE=...`, f16/bf16/f32
   bins near the top binades) to watch U climb toward 65504 and 3.4e38 in the
   HMMA operands.
5. Answer these questions:
   * Does nixnan flag the inf in the HMMA destination (fp16acc16)?
   * Does it flag the inf born at F2FP (acc32 modes)?
   * Does it flag the NaN from 0 × inf inside the HMMA?
   * Does `LINE_INFO` resolve the `mma_sync` source lines?
6. Write the results into this file, then make a README and slides, and commit.

## Results after the reboot (2026-10-19, nixnan `print_histo_instrn`, `~/repos/nixnan/nixnan.so`)

* The driver mismatch is fixed (`nvidia-smi` shows 595.99.02, same as the library).
* nixnan was still silent until `LD_LIBRARY_PATH` was unset. Run it as
  `env -u LD_LIBRARY_PATH LINE_INFO=1 PRINT_ILL_INSTR=1 LOGFILE=logs/nixnan-$m.log LD_PRELOAD=~/repos/nixnan/nixnan.so ./rd_tc $m`.
  (I did not isolate which entry of `LD_LIBRARY_PATH` is the cause.)
* instr_count (NVBit sample tool) also attaches with `LD_LIBRARY_PATH` cleared.
* Logs: `logs/nixnan-{fp16acc16,fp16acc32,bf16acc32}.log`, `logs/stdout-*.txt`.

Answers to the questions:

| Question | fp16acc16 | fp16acc32 | bf16acc32 |
|---|---|---|---|
| inf flagged in the HMMA? | yes, operand 0 of `HMMA.16816.F16` | yes, but only as an input of the next `HMMA.16816.F32` | yes, only as an input of the next `HMMA.16816.F32.BF16` |
| inf flagged at `F2FP`? | n/a | no (0 `F2FP` lines; `F2FP` has no entry in instructions.json) | no |
| NaN (0 x inf) flagged in the HMMA? | yes | yes | yes |
| steps | first inf 230, first NaN 231 | same | first inf 1764, first NaN 1765 |

`LINE_INFO=1` did not give `mma_sync` source lines in these runs. The histogram pass (to do 4) has not been run.
