// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors for solid mechanics example using PETSc

#ifndef TRACTION_BOUNDARY_H
#define TRACTION_BOUNDARY_H

#include <ceed.h>

// -----------------------------------------------------------------------------
// This QFunction computes the surface integral of the user traction vector on
//   the constrained faces.
//
// Reference (parent) 2D coordinates: X
// Physical (current) 3D coordinates: x
// Change of coordinate matrix:
//   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
//
// (J1,J2,J3) is given by the cross product of the columns of dxdX_{i,j}
//
// detJb is the magnitude of (J1,J2,J3)
//
// Computed:
//   t * (w detJb)
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupTractionBCs)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*w) = in[1];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // User stress tensor
  const CeedScalar(*traction) = (const CeedScalar(*))ctx;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // *INDENT-OFF*
    const CeedScalar dxdX[3][2] = {
        {J[0][0][i], J[1][0][i]},
        {J[0][1][i], J[1][1][i]},
        {J[0][2][i], J[1][2][i]}
    };
    // *INDENT-ON*
    // J1, J2, and J3 are given by the cross product of the columns of dxdX
    const CeedScalar J1 = dxdX[1][0] * dxdX[2][1] - dxdX[2][0] * dxdX[1][1];
    const CeedScalar J2 = dxdX[2][0] * dxdX[0][1] - dxdX[0][0] * dxdX[2][1];
    const CeedScalar J3 = dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1];

    // Qdata
    // -- Interp-to-Interp q_data
    CeedScalar wdetJb = w[i] * sqrt(J1 * J1 + J2 * J2 + J3 * J3);

    // Traction surface integral
    for (CeedInt j = 0; j < 3; j++) v[j][i] = traction[j] * wdetJb;

  }  // End of Quadrature Point Loop

  // Return
  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of TRACTION_BOUNDARY_H
