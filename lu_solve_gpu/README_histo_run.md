# histo_run.sh — exponent histogram of `lu_solve_gpu` under nixnan

## What it does

Runs one `lu_solve_gpu` program on a CSV matrix under nixnan with the exponent histogram on, and
leaves the logs in one directory. For every kernel and every exponent bin, nixnan prints a line each
time that bin has been hit another 64 times. The bins cover the whole exponent range of FP16, BF16
and FP32, so low-precision values cannot fall outside them.

## Usage

    ./histo_run.sh <matrix.csv> [mode] [outdir]

| argument | default | meaning |
|---|---|---|
| `matrix.csv` | (required) | input matrix, same CSV format the programs already read |
| `mode` | `fp16acc32` | `fp16acc32`, `fp16acc16`, `bf16acc32` run `./lu_solve_gpu_tc` (tensor cores, see `TC.md`); `f64` runs `./lu_solve_gpu` (double only) |
| `outdir` | `logs/histo/<matrix>_<mode>` | where the files go. Any name works; it is only a directory |

Run it from anywhere; the script changes to its own directory first, so a relative matrix or
output path is relative to `lu_solve_gpu/`. It runs `make` for the binary it needs.

Environment variables:

| variable | default | effect |
|---|---|---|
| `RECORD_INST` | `0` | `1` makes every bin a `(record_inst)` bin, so each report also names the SASS instruction that hit the threshold |
| `COUNT` | `64` | the fixed report threshold |
| `F64_WIDTH` | `8` | width in binades of the f64 bins |
| `TMO` | `110` | timeout in seconds for the nixnan run |
| `NIXNAN_LIB` | `../external/nixnan/nixnan.so` | path of the nixnan library |

Examples:

    ./histo_run.sh bad_matrix.csv                          # fp16acc32, logs/histo/bad_matrix_fp16acc32/
    RECORD_INST=1 ./histo_run.sh bad_matrix.csv fp16acc32 fp16acc32
    ./histo_run.sh matrix_128.csv bf16acc32
    ./histo_run.sh good_matrix.csv f64

## The bins (`spec.json`)

The script writes the spec in nixnan's own `BIN_SPEC_FILE` format (the same keys as in the nixnan
README: `count`, `doublings`, `max_reports`, and `[min_exp, max_exp]` ranges per format).

| format | bins | covers |
|---|---|---|
| f16 | 32 bins of 1 binade | -15 (zero and subnormals) up to 16 (inf and NaN) |
| bf16, f32 | 256 bins of 1 binade each | -127 (zero and subnormals) up to 128 (inf and NaN) |
| f64 | 256 bins of 8 binades | -1023 (zero and subnormals) up to 1024 (inf and NaN) |

`count` is 64 and `doublings` is 0, so the threshold never changes: a (kernel, bin) pair reports
each time it is hit another 64 times. In reports the lowest bin is printed as `zero` and the
highest as `inf` (for f64, ranges such as `[zero,-1016]` and `[1017,inf]`).
The zero bin holds zeros and subnormals together; the histogram cannot tell them apart.

## How it runs

    timeout 110 env HISTOGRAM=1 BIN_SPEC_FILE=<outdir>/spec.json LD_PRELOAD=<nixnan.so> <binary> <matrix> [mode]

`env` applies the preload to the program. `LD_PRELOAD=… timeout …` would preload into `timeout`
and instrument nothing.

## Files written to `<outdir>`

| file | contents |
|---|---|
| `spec.json` | the bin spec above |
| `stdout.log` | the NVBit/nixnan banner (environment variables and their values), then the program's own output: the matrix, `x[…]` and the residual norm |
| `histo.log` | nixnan's stderr: `running kernel` lines; the threshold reports `#nixnan: <fmt> bin has reached threshold: function=<kernel> range=[lo,hi] count=64` (plus `instruction=<SASS>` with `RECORD_INST=1`); nixnan's ordinary `error [NaN\|infinity\|div0…]` reports; and the final nixnan report with per-format totals |
| `summary.txt` | made by the script: for each format and kernel, the number of threshold reports per bin |

`summary.txt` is not a full histogram. A bin hit fewer than 64 times never appears, and the exact
hit count is not recorded, only that the threshold was reached. The script prints the summary at the end.
An example directory is `fp16acc32/` (with its own `README.md`).

## Limits

- Tested only for `fp16acc32` on `bad_matrix.csv`, on one machine (RTX 3090, sm_86, CUDA 12.4).
  The other modes and matrices are untested.
- The log can be large on big matrices (11 MB for `matrix_128.csv` with the old threshold of 8). With 64 it should be smaller; not measured.
- nixnan's histogram counts the exponent of every floating-point operand, source and destination, of
  every instrumented instruction, so a bin's count is operand occurrences, not distinct values.
- `RECORD_INST=1` names only the instructions that hit a threshold. For a kernel's full SASS listing run
  `cuobjdump -sass lu_solve_gpu_tc`.
