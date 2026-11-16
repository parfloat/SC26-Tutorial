# fp16acc32/ — nixnan histogram run of `lu_solve_gpu_tc` on `bad_matrix.csv`

## The command that produced this directory

    RECORD_INST=1 ./histo_run.sh bad_matrix.csv fp16acc32 fp16acc32

Run from `lu_solve_gpu/`. The three arguments of `histo_run.sh` are:

| position | value | meaning |
|---|---|---|
| 1 | `bad_matrix.csv` | the 20 x 20 ill-conditioned input matrix |
| 2 | `fp16acc32` | mode: tensor-core LU with FP16 inputs and an FP32 accumulator, so the binary is `./lu_solve_gpu_tc` (`HMMA.16816.F32`) |
| 3 | `fp16acc32` | output directory. It is only a directory name here. Without the third argument the files would go to `logs/histo/bad_matrix_fp16acc32/` |

`RECORD_INST=1` makes every bin in the spec a `(record_inst)` bin, so each threshold report also
names the SASS instruction that pushed the bin over its threshold.

What the script does:

1. Builds `lu_solve_gpu_tc` with `make` if needed.
2. Writes `spec.json`: fixed threshold `count = 64`, `doublings = 0` (no doubling).
3. Runs `timeout 110 env HISTOGRAM=1 BIN_SPEC_FILE=fp16acc32/spec.json LD_PRELOAD=<nixnan.so> ./lu_solve_gpu_tc bad_matrix.csv fp16acc32`.
   stdout goes to `stdout.log`, stderr to `histo.log`.
4. Reads `histo.log` and writes `summary.txt`.

How the threshold works: a (kernel, bin) pair reports each time it has been hit another 64 times.
The threshold is fixed, so every report line says `count=64`. See `../README_histo_run.md` for the
full description of the script.

## Files

### `spec.json` (nixnan input, written by the script)
The `BIN_SPEC_FILE` for this run, in nixnan's own format (`count` = 64, `doublings` = 0, `max_reports`, and
one key per format holding `[min_exp, max_exp]` ranges).
- `f16 (record_inst)`: 32 bins of one binade each, from -15 (zero and subnormals) to 16 (inf and NaN).
- `bf16 (record_inst)`, `f32 (record_inst)`: 256 bins of one binade each, from -127 (zero and subnormals) to 128 (inf and NaN).
- `f64 (record_inst)`: 256 bins of 8 binades each, from -1023 (zero and subnormals) to 1024 (inf and NaN).

### `stdout.log` (the program's stdout)
- The NVBit and nixnan banner: the list of environment variables with their current values
  (this is where you can confirm `HISTOGRAM = 1` and `BIN_SPEC_FILE = fp16acc32/spec.json`).
- The program's own output: the matrix it loaded, then the solution `x[0..19]`, then
  `Residual norm ||Ax - b||`.
- In this run `x[0..19]` are all `nan`/`-nan` and the residual norm is `-nan`: the factorisation broke down.

### `histo.log` (nixnan's stderr: the main result)
Four kinds of lines:
- `#nixnan: running kernel [...]`: one line per kernel nixnan instrumented (12 kernels).
- `#nixnan: <fmt> bin has reached threshold: function=<kernel> range=[lo,hi] count=<n> instruction=<SASS> ; in function=<kernel>`:
  the histogram reports (437 here). `range=[lo,hi]` is the unbiased exponent bin (`zero` is the zero/subnormal bin,
  `inf` is the inf/NaN bin). `count=<n>` is the fixed threshold, always 64. `instruction=` appears
  because of `RECORD_INST=1`.
- `#nixnan: error [NaN|infinity|div0|...] detected in operand <k> of instruction <SASS> ; in function <kernel> of type <fmt>`:
  nixnan's ordinary exception reports (385 lines), which appear alongside the histogram reports. The first ones are the
  `MUFU.RCP64H` division by zero and the NaNs that follow in `compute_multipliers_kernel`.
- The end-of-run `nixnan Report`: totals per format (here FP16 NaN 3, FP32 NaN 8, FP64 NaN 362, Inf 4, Div by 0 4, plus the NaN
  stores to memory), and `Exponent range for f16/f32/f64: [zero, inf]`, the lowest and highest exponent bin seen per format.

### `summary.txt` (made by the script, not by nixnan)
A digest of the threshold reports in `histo.log`, grouped by format, then kernel, then bin:
`reports` is how many threshold lines that bin produced. It is not a full histogram: a bin that
was hit fewer than 64 times never appears, and the exact number of hits is not recorded.
It does not use the `instruction=` field; for that, read `histo.log`.

## Reading it

- The same few facts show up in several places. Search `histo.log` for `inf` bins to see where infinities were produced,
  and for `zero` bins to see where values collapsed to zero or a subnormal.
- To get a kernel's whole SASS listing, which `histo.log` only samples through `instruction=`, run
  `cuobjdump -sass ../lu_solve_gpu_tc`.
