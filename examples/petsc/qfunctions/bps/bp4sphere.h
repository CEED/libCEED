// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example for a vector field on the sphere using PETSc

#ifndef bp4sphere_h
#define bp4sphere_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs3)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *q_data = in[1];
  // Outputs
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar  R       = context[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read global Cartesian coordinates
    CeedScalar x = X[i + Q * 0], y = X[i + Q * 1], z = X[i + Q * 2];
    // Normalize quadrature point coordinates to sphere
    CeedScalar rad = sqrt(x * x + y * y + z * z);
    x *= R / rad;
    y *= R / rad;
    z *= R / rad;
    // Compute latitude and longitude
    const CeedScalar theta  = asin(z / R);  // latitude
    const CeedScalar lambda = atan2(y, x);  // longitude

    // Use absolute value of latitude for true solution
    // Component 1
    true_soln[i + 0 * Q] = sin(lambda) * cos(theta);
    // Component 2
    true_soln[i + 1 * Q] = 2 * true_soln[i + 0 * Q];
    // Component 3
    true_soln[i + 2 * Q] = 3 * true_soln[i + 0 * Q];

    // Component 1
    rhs[i + 0 * Q] = q_data[i + Q * 0] * 2 * sin(lambda) * cos(theta) / (R * R);
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
CEED_QFUNCTION(Diff3)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *q_data = in[1];
  CeedScalar       *vJ = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar uJ[3][2] = {
        {ug[i + (0 + 0 * 3) * Q], ug[i + (0 + 1 * 3) * Q]},
        {ug[i + (1 + 0 * 3) * Q], ug[i + (1 + 1 * 3) * Q]},
        {ug[i + (2 + 0 * 3) * Q], ug[i + (2 + 1 * 3) * Q]}
    };
    // Read q_data
    const CeedScalar w_det_J = q_data[i + Q * 0];
    // -- Grad-to-Grad q_data
    // ---- dXdx_j,k * dXdx_k,j
    const CeedScalar dXdxdXdx_T[2][2] = {
        {q_data[i + Q * 1], q_data[i + Q * 3]},
        {q_data[i + Q * 3], q_data[i + Q * 2]}
    };

    for (int k = 0; k < 3; k++) {    // k = component
      for (int j = 0; j < 2; j++) {  // j = direction of vg
        vJ[i + (k + j * 3) * Q] = w_det_J * (uJ[k][0] * dXdxdXdx_T[0][j] + uJ[k][1] * dXdxdXdx_T[1][j]);
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp4sphere_h
