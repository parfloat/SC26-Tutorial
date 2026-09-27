# hec_fluidSim

A local copy of HeCBench's `fluidSim-cuda` benchmark, kept here as this
tutorial's own scratch/test area for quick, small-scale runs -- separate
from the `external/HeCBench` submodule, which tracks upstream and isn't
something we can push changes to (see "Provenance" below).

## What this benchmark does

A D2Q9 Lattice-Boltzmann Method (LBM) fluid simulation on a fixed 256x256
2D grid. Each cell holds a particle-distribution function over the 9 D2Q9
discrete velocities (1 rest + 8 moving directions). One CUDA kernel
(`lbm`, in `kernels.cu`) fuses the two classic LBM steps every iteration:

- **Collision**: relax each cell's distributions toward the local
  equilibrium (`ced`/`computefEq`), controlled by relaxation parameter
  `omega`.
- **Streaming**: push post-collision values to the neighboring cell along
  each of the 8 directions, with the domain boundary (a ring of cells
  marked in `h_type`) handled via bounce-back.

The driver (`fluidSim` in `kernels.cu`) launches `lbm` once per requested
iteration, ping-ponging between two distribution buffers each step.
Command line: `./main <iterations>` (grid size is a compile-time constant,
not a CLI argument). Built with `-DVERIFY`, it also runs a CPU
`reference()` implementation and diffs against the GPU output.

## Files

- `main.cu`, `kernels.cu`, `reference.h`, `utils.h`, `Makefile`,
  `CMakeLists.txt`, `COPYRIGHT` -- copied verbatim from upstream (see
  Provenance), unmodified.
- `MakeShort` -- **new here, not upstream.** Identical to `Makefile`
  except its `run` target uses 1 iteration (`./main 1`) instead of
  upstream's 10000 (`./main 10000`). A quick smoke test, not a benchmark
  measurement -- use `make -f MakeShort run`.
- `run_reduced.sh` -- **new here, not upstream.** A small wrapper for
  anything between MakeShort's fixed 1 iteration and upstream's 10000:
  `./run_reduced.sh [iterations]` (default 5). Builds via `MakeShort`'s
  rules, then runs `./main <iterations>`.
- `main_overflow.cu` / `MakeOverflow` -- **new here, not upstream.** A
  deliberately *unstable* variant (`omega = 1e8` instead of `main.cu`'s
  stable `1.2`), to force real FP exceptions out of the `lbm` kernel,
  which never produces one under the default configuration (see that
  file's header comment for the mechanism: the BGK relaxation update
  amplifies any deviation from equilibrium by roughly `omega`x per
  iteration once `omega` is far outside the `0 < omega < 2` stability
  bound, overflowing double's range within ~40 iterations; the
  now-inconsistent density can then also divide-by-zero in `lbm`'s
  `u.x = (...) / rho`). No `-DVERIFY` (a GPU/CPU reference comparison is
  meaningless once both sides are diverging chaotically). Also prints its
  own independent NaN/Inf/finite cell count, not just nixnan's.
  `make -f MakeOverflow run` (60 iterations).

Neither `MakeShort` nor `run_reduced.sh` can shrink the 256x256 grid
itself -- `lbm_width`/`lbm_height` are compile-time constants in `main.cu`
(and `main_overflow.cu`), not CLI arguments. All three only vary the
iteration count (and, for `main_overflow.cu`, `omega`).

## Log/

`nixnan`-instrumented runs (`LD_PRELOAD=../external/nixnan/nixnan.so`,
unsampled, default exception checking) against `main_overflow`:

- `fluidSim-overflow-60iter-nixnan-default-exceptions.log` -- 60
  iterations at `omega=1e8`. Standalone sanity check (no nixnan) for the
  same run: 580,644 of 589,824 output cells are NaN, 0 Inf, 9,180 still
  finite. nixnan independently confirms this, attributing it to the
  kernel's own `f64` `DFMA`/`DMUL`/`DADD` instructions (1249 distinct NaN
  sites, 16.7M repeats; 831 distinct Infinity sites; 824 -Infinity; even
  26 distinct Subnormal sites, consistent with values passing through the
  full extreme-magnitude spectrum en route to NaN). It also reports a
  handful of FP16/FP32 NaN/Infinity hits despite this kernel being
  entirely `double` -- the same binary-level false-positive already
  documented in `lu_solve_gpu`'s report (`ptxas` splitting a 64-bit
  double's halves across the register file via an instruction nixnan's
  table classifies as a narrower format).

## Provenance

Copied from [zjin-lcf/HeCBench](https://github.com/zjin-lcf/HeCBench),
path `src/fluidSim-cuda`, at commit
[`2b73f4d`](https://github.com/zjin-lcf/HeCBench/commit/2b73f4da07df8aa7b031f6f6419175c6f8d86c1a)
(the version present at the `master` commit this repository's
`external/HeCBench` submodule was pinned to when this copy was made).

HeCBench is a benchmark suite by Zheming Jin (Argonne National
Laboratory / Oak Ridge National Laboratory) et al.:

> Z. Jin and J. S. Vetter, "A Benchmark Suite for Improving Performance
> Portability of the SYCL Programming Model," 2023 IEEE International
> Symposium on Performance Analysis of Systems and Software (ISPASS),
> Raleigh, NC, USA, 2023, pp. 325-327,
> doi: [10.1109/ISPASS57527.2023.00041](https://ieeexplore.ieee.org/document/10158214)

**Licensing**: the HeCBench repository as a whole is BSD 3-Clause
(Copyright Zheming Jin, Argonne National Laboratory / Oak Ridge National
Laboratory) -- see
[the repository's top-level LICENSE](https://github.com/zjin-lcf/HeCBench/blob/master/LICENSE).
`fluidSim-cuda` itself traces back to an AMD APP SDK sample and carries
its own separate copyright/license terms (copied here unmodified as
`COPYRIGHT`: "Copyright (c) 2009 Advanced Micro Devices, Inc. All rights
reserved."), which governs `main.cu`, `kernels.cu`, `reference.h`, and
`utils.h` as copied into this directory.
