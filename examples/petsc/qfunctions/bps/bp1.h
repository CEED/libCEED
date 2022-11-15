// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example using PETSc

#ifndef bp1_h
#define bp1_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required to apply the
//   mass operator
//
// The quadrature data is stored in the array q_data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Qdata: det_J * w
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeo)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*w)                = in[2];  // Note: *X = in[0]
  // Outputs
  CeedScalar *q_data = out[0];

  const CeedInt dim = 3;
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    CeedScalar A[3][3];
    for (CeedInt j = 0; j < dim; j++) {
      for (CeedInt k = 0; k < dim; k++) {
        // Equivalent code with no mod operations:
        // A[k][j] = J[k+1][j+1]*J[k+2][j+2] - J[k+1][j+2]*J[k+2][j+1]
        A[k][j] = J[(k + 1) % dim][(j + 1) % dim][i] * J[(k + 2) % dim][(j + 2) % dim][i] -
                  J[(k + 1) % dim][(j + 2) % dim][i] * J[(k + 2) % dim][(j + 1) % dim][i];
      }
    }
    const CeedScalar detJ = J[0][0][i] * A[0][0] + J[0][1][i] * A[0][1] + J[0][2][i] * A[0][2];
    q_data[i]             = detJ * w[i];
  }  // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar       *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    true_soln[i] = sqrt(x[i] * x[i] + x[i + Q] * x[i + Q] + x[i + 2 * Q] * x[i + 2 * Q]);
    rhs[i]       = w[i] * true_soln[i];
  }  // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the mass operator for a scalar field.
//
// Inputs:
//   u     - Input vector at quadrature points
//   q_data - Geometric factors
//
// Output:
//   v     - Output vector (test functions) at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar       *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) v[i] = q_data[i] * u[i];

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp1_h
