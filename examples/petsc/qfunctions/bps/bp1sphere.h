// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example for a scalar field on the sphere using PETSc

#ifndef bp1sphere_h
#define bp1sphere_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations when reference coordinates have a different
//   dimension than the one of physical coordinates
//
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global 3D physical coordinates given by the mesh: xx \in [-R, R]^3
//   with R radius of the sphere
//
// Local 3D physical coordinates on the 2D manifold: x \in [-l, l]^3
//   with l half edge of the cube inscribed in the sphere
//
// Change of coordinates matrix computed by the library:
//   (physical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 * 2]
//
// Change of coordinates x (on the 2D manifold) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [3 * 3]
//
// Change of coordinates x (on the 2D manifold) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j [3 * 2] = dx_i/dxx_k [3 * 3] * dxx_k/dX_j [3 * 2]
//
// mod_J is given by the magnitude of the cross product of the columns of dx_i/dX_j
//
// The quadrature data is stored in the array q_data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Qdata: mod_J * w
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeo)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  // Outputs
  CeedScalar *q_data = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read global Cartesian coordinates
    const CeedScalar xx[3] = {X[i + 0 * Q], X[i + 1 * Q], X[i + 2 * Q]};

    // Read dxxdX Jacobian entries, stored as
    // 0 3
    // 1 4
    // 2 5
    const CeedScalar dxxdX[3][2] = {
        {J[i + Q * 0], J[i + Q * 3]},
        {J[i + Q * 1], J[i + Q * 4]},
        {J[i + Q * 2], J[i + Q * 5]}
    };

    // Setup
    // x = xx (xx^T xx)^{-1/2}
    // dx/dxx = I (xx^T xx)^{-1/2} - xx xx^T (xx^T xx)^{-3/2}
    const CeedScalar mod_xx_sq = xx[0] * xx[0] + xx[1] * xx[1] + xx[2] * xx[2];
    CeedScalar       xx_sq[3][3];
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) xx_sq[j][k] = xx[j] * xx[k] / (sqrt(mod_xx_sq) * mod_xx_sq);
    }

    const CeedScalar dxdxx[3][3] = {
        {1. / sqrt(mod_xx_sq) - xx_sq[0][0], -xx_sq[0][1],                       -xx_sq[0][2]                      },
        {-xx_sq[1][0],                       1. / sqrt(mod_xx_sq) - xx_sq[1][1], -xx_sq[1][2]                      },
        {-xx_sq[2][0],                       -xx_sq[2][1],                       1. / sqrt(mod_xx_sq) - xx_sq[2][2]}
    };

    CeedScalar dxdX[3][2];
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 2; k++) {
        dxdX[j][k] = 0;
        for (int l = 0; l < 3; l++) dxdX[j][k] += dxdxx[j][l] * dxxdX[l][k];
      }
    }

    // J is given by the cross product of the columns of dxdX
    const CeedScalar J[3] = {dxdX[1][0] * dxdX[2][1] - dxdX[2][0] * dxdX[1][1], dxdX[2][0] * dxdX[0][1] - dxdX[0][0] * dxdX[2][1],
                             dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1]};

    // Use the magnitude of J as our detJ (volume scaling factor)
    const CeedScalar mod_J = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);

    // Interp-to-Interp q_data
    q_data[i + Q * 0] = mod_J * w[i];
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    true_soln[i] = fabs(theta);

    rhs[i] = q_data[i] * true_soln[i];
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
  // Inputs
  const CeedScalar *u = in[0], *q_data = in[1];
  // Outputs
  CeedScalar *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) v[i] = q_data[i] * u[i];

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp1sphere_h
