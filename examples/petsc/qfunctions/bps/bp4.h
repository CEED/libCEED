// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp4_h
#define bp4_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs3)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar       *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[3] = {0, 1., 2.};
    const CeedScalar k[3] = {1., 2., 3.};

    // Component 1
    true_soln[i + 0 * Q] =
        sin(M_PI * (c[0] + k[0] * x[i + Q * 0])) * sin(M_PI * (c[1] + k[1] * x[i + Q * 1])) * sin(M_PI * (c[2] + k[2] * x[i + Q * 2]));
    // Component 2
    true_soln[i + 1 * Q] = 2 * true_soln[i + 0 * Q];
    // Component 3
    true_soln[i + 2 * Q] = 3 * true_soln[i + 0 * Q];

    // Component 1
    rhs[i + 0 * Q] = w[i + Q * 0] * M_PI * M_PI * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) * true_soln[i + 0 * Q];
    // Component 2
    rhs[i + 1 * Q] = 2 * rhs[i + 0 * Q];
    // Component 3
    rhs[i + 2 * Q] = 3 * rhs[i + 0 * Q];
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the diffusion operator for a vector field of 3 components.
//
// Inputs:
//   ug     - Input vector Jacobian at quadrature points
//   q_data  - Geometric factors
//
// Output:
//   vJ     - Output vector (test functions) Jacobian at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff3)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *qd = in[1];
  CeedScalar       *vg = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u components
    const CeedScalar uJ[3][3] = {
        {ug[i + (0 + 0 * 3) * Q], ug[i + (0 + 1 * 3) * Q], ug[i + (0 + 2 * 3) * Q]},
        {ug[i + (1 + 0 * 3) * Q], ug[i + (1 + 1 * 3) * Q], ug[i + (1 + 2 * 3) * Q]},
        {ug[i + (2 + 0 * 3) * Q], ug[i + (2 + 1 * 3) * Q], ug[i + (2 + 2 * 3) * Q]}
    };
    // Read q_data (dXdxdXdx_T symmetric matrix)
    const CeedScalar dXdxdXdx_T[3][3] = {
        {qd[i + 1 * Q], qd[i + 2 * Q], qd[i + 3 * Q]},
        {qd[i + 2 * Q], qd[i + 4 * Q], qd[i + 5 * Q]},
        {qd[i + 3 * Q], qd[i + 5 * Q], qd[i + 6 * Q]}
    };

    for (int k = 0; k < 3; k++) {    // k = component
      for (int j = 0; j < 3; j++) {  // j = direction of vg
        vg[i + (k + j * 3) * Q] = (uJ[k][0] * dXdxdXdx_T[0][j] + uJ[k][1] * dXdxdXdx_T[1][j] + uJ[k][2] * dXdxdXdx_T[2][j]);
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp4_h
