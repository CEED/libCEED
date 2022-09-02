// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example for a vector field on the sphere using PETSc

#ifndef bp2sphere_h
#define bp2sphere_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs3)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *q_data = in[1];
  // Outputs
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar  R       = context[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Compute latitude
    const CeedScalar theta = asin(X[i + 2 * Q] / R);

    // Use absolute value of latitude for true solution
    // Component 1
    true_soln[i + 0 * Q] = fabs(theta);
    // Component 2
    true_soln[i + 1 * Q] = 2 * true_soln[i + 0 * Q];
    // Component 3
    true_soln[i + 2 * Q] = 3 * true_soln[i + 0 * Q];

    // Component 1
    rhs[i + 0 * Q] = q_data[i] * true_soln[i];
    // Component 2
    rhs[i + 1 * Q] = 2 * rhs[i + 0 * Q];
    // Component 3
    rhs[i + 2 * Q] = 3 * rhs[i + 0 * Q];
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the mass operator for a vector field of 3 components.
//
// Inputs:
//   u     - Input vector at quadrature points
//   q_data - Geometric factors
//
// Output:
//   v     - Output vector (test functions) at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass3)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar       *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Component 1
    v[i + 0 * Q] = q_data[i] * u[i + 0 * Q];
    // Component 2
    v[i + 1 * Q] = q_data[i] * u[i + 1 * Q];
    // Component 3
    v[i + 2 * Q] = q_data[i] * u[i + 2 * Q];
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp2sphere_h
