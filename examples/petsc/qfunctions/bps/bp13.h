// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#endif

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassDiffRhs)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar       *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[3] = {0, 1., 2.};
    const CeedScalar k[3] = {1., 2., 3.};

    true_soln[i] = sin(M_PI * (c[0] + k[0] * x[i + Q * 0])) * sin(M_PI * (c[1] + k[1] * x[i + Q * 1])) * sin(M_PI * (c[2] + k[2] * x[i + Q * 2]));

    rhs[i] = w[i + Q * 0] * (M_PI * M_PI * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) + 1.0) * true_soln[i];
  }  // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the mass + diffusion operator for a scalar field.
//
// Inputs:
//   u       - Input vector at quadrature points
//   ug      - Input vector gradient at quadrature points
//   q_data  - Geometric factors
//
// Output:
//   v      - Output vector (test functions) at quadrature points
//   vg     - Output vector (test functions) gradient at quadrature points
// -----------------------------------------------------------------------------
CEED_QFUNCTION(MassDiff)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *ug = in[1], *q_data = in[2];
  CeedScalar       *v = out[0], *vg = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[3] = {ug[i + Q * 0], ug[i + Q * 1], ug[i + Q * 2]};
    // Read q_data (dXdxdXdx_T symmetric matrix)
    const CeedScalar dXdxdXdx_T[3][3] = {
        {q_data[i + 1 * Q], q_data[i + 2 * Q], q_data[i + 3 * Q]},
        {q_data[i + 2 * Q], q_data[i + 4 * Q], q_data[i + 5 * Q]},
        {q_data[i + 3 * Q], q_data[i + 5 * Q], q_data[i + 6 * Q]}
    };

    // Mass
    v[i] = q_data[i + 0 * Q] * u[i];
    // Diff
    for (int j = 0; j < 3; j++) {  // j = direction of vg
      vg[i + j * Q] = (du[0] * dXdxdXdx_T[0][j] + du[1] * dXdxdXdx_T[1][j] + du[2] * dXdxdXdx_T[2][j]);
    }
  }  // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------
