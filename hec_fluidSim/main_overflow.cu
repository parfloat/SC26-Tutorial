// main_overflow.cu -- deliberately numerically-unstable variant of main.cu,
// to force real FP exceptions out of the LBM `lbm` kernel (kernels.cu),
// which otherwise never produces one (see README.md).
//
// Mechanism: the BGK collision update in `lbm` is
//   f_new = (1 - omega) * f_old + omega * f_eq
// Stability requires 0 < omega < 2 (any deviation between f_old and f_eq
// is damped each step). main.cu's own omega = 1.2 is safely inside that
// range. Setting omega far outside it (here: 1e8) makes |1 - omega| huge,
// so any nonzero deviation from equilibrium -- which appears starting at
// iteration 2, once streaming/bounce-back near the boundary makes
// neighboring cells' distributions diverge slightly from their own
// equilibrium -- gets amplified roughly 1e8x *every single iteration*.
// That reaches double's ~1.8e308 range in on the order of
// log_1e8(1e308) ~= 38 iterations; this file's default run uses 60 for
// margin. Once any distribution overflows to +-Inf, the density
// rho = f0 + sum(f1234) + sum(f5678) computed at the next iteration's
// collision step can itself pass through exactly 0 while oscillating
// (the (1-omega) factor is a large *negative* number for omega>1, so the
// update flips sign each step) -- and kernels.cu's `lbm` divides by rho
// directly (u.x = (...)/rho, u.y = (...)/rho) to get velocity, with no
// guard against rho==0. So this is expected to produce both Infinity
// (from the raw amplification) and NaN (from Inf combining with itself,
// or from the rho==0 division), not just one or the other.
//
// Everything else (grid size, boundary setup, initial velocity, weights,
// directions) is identical to main.cu. VERIFY/reference() is deliberately
// not used here: once the simulation is in this regime, GPU-vs-CPU
// floating-point non-associativity make the two diverge in different,
// equally-meaningless ways -- the point of this file is the exception,
// not a numerical match to a reference.
//
// Build: nvcc -arch=sm_60 -std=c++17 main_overflow.cu kernels.cu -o main_overflow
// Run:   ./main_overflow <iterations>   (60 recommended; see above)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "utils.h"
#include "reference.h"  // for computefEq() -- defined there unconditionally,
                         // not inside its #ifdef VERIFY blocks

int main(int argc, char * argv[])
{
  if (argc != 2) {
    printf("Usage %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iterations = atoi(argv[1]); // Simulation iterations
  const int lbm_width = 256;            // Dimension of LBM simulation area
  const int lbm_height = 256;

  int dims[2] = {lbm_width, lbm_height};
  size_t temp = dims[0] * dims[1];

   // Directions
   double e[9][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}, {1,1}, {-1,1}, {-1,-1}, {1,-1}};

   // Weights
   double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

  // Omega -- see file header. main.cu uses 1.2 (stable); this is
  // deliberately ~1e8 past the 0 < omega < 2 stability bound.
  const double omega = 1.0e8;

  double8 dirX, dirY; // Directions

  // host inputs
  double *h_if0 = (double*)malloc(sizeof(double) * temp);
  double *h_if1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_if5678 = (double*)malloc(sizeof(double4) * temp);

  // Host outputs
  double *h_of0 = (double*)malloc(sizeof(double) * temp);
  double *h_of1234 = (double*)malloc(sizeof(double4) * temp);
  double *h_of5678 = (double*)malloc(sizeof(double4) * temp);

  // Cell Type - Boundary = 1 or Fluid = 0
  bool *h_type = (bool*)malloc(sizeof(bool) * temp);

  // Density
  double *rho = (double*)malloc(sizeof(double) * temp);

  // Velocity
  double2 *u = (double2*)malloc(sizeof(double2) * temp);

  // Initial velocity, same as main.cu
  double u0[2] = {0.01, 0.01};

  srand(123);
  for (int y = 0; y < dims[1]; y++)
  {
    for (int x = 0; x < dims[0]; x++)
    {
      int pos = x + y * dims[0];

      double den = rand() % 10 + 1;

      u[pos].x = u0[0];
      u[pos].y = u0[1];

      h_if0[pos]            = computefEq(den, w[0], e[0], u0);
      h_if1234[pos * 4 + 0] = computefEq(den, w[1], e[1], u0);
      h_if1234[pos * 4 + 1] = computefEq(den, w[2], e[2], u0);
      h_if1234[pos * 4 + 2] = computefEq(den, w[3], e[3], u0);
      h_if1234[pos * 4 + 3] = computefEq(den, w[4], e[4], u0);
      h_if5678[pos * 4 + 0] = computefEq(den, w[5], e[5], u0);
      h_if5678[pos * 4 + 1] = computefEq(den, w[6], e[6], u0);
      h_if5678[pos * 4 + 2] = computefEq(den, w[7], e[7], u0);
      h_if5678[pos * 4 + 3] = computefEq(den, w[8], e[8], u0);

      if (x == 0 || x == (dims[0] - 1) || y == 0 || y == (dims[1] - 1))
        h_type[pos] = 1;
      else
        h_type[pos] = 0;
    }
  }

  dirX.s0 = e[1][0]; dirY.s0 = e[1][1];
  dirX.s1 = e[2][0]; dirY.s1 = e[2][1];
  dirX.s2 = e[3][0]; dirY.s2 = e[3][1];
  dirX.s3 = e[4][0]; dirY.s3 = e[4][1];
  dirX.s4 = e[5][0]; dirY.s4 = e[5][1];
  dirX.s5 = e[6][0]; dirY.s5 = e[6][1];
  dirX.s6 = e[7][0]; dirY.s6 = e[7][1];
  dirX.s7 = e[8][0]; dirY.s7 = e[8][1];

  printf("omega = %.3e (unstable; main.cu's default is 1.2)\n", omega);

  fluidSim (iterations, omega, dims, h_type, u, rho, dirX, dirY, w,
            h_if0, h_if1234, h_if5678,
            h_of0, h_of1234, h_of5678);

  // Independent-of-nixnan sanity check: count non-finite output cells.
  size_t n_finite = 0, n_nan = 0, n_inf = 0;
  size_t const n_scalars = temp;       // h_of0
  size_t const n_vec4 = temp * 4;      // h_of1234, h_of5678, as flat doubles
  auto scan = [&](double const* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
      double v = arr[i];
      if (std::isnan(v)) ++n_nan;
      else if (std::isinf(v)) ++n_inf;
      else ++n_finite;
    }
  };
  scan(h_of0, n_scalars);
  scan(h_of1234, n_vec4);
  scan(h_of5678, n_vec4);
  printf("Output cells: finite=%zu NaN=%zu Inf=%zu (total=%zu)\n",
         n_finite, n_nan, n_inf, n_finite + n_nan + n_inf);

  if(h_if0)    free(h_if0);
  if(h_if1234) free(h_if1234);
  if(h_if5678) free(h_if5678);
  if(h_of0)    free(h_of0);
  if(h_of1234) free(h_of1234);
  if(h_of5678) free(h_of5678);
  if(h_type)   free(h_type);
  if(rho)      free(rho);
  if(u)        free(u);

  return 0;
}
