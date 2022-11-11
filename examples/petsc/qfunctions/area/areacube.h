// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example for a scalar field on the sphere using PETSc

#ifndef areacube_h
#define areacube_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates have a different dimension than the one of
//   physical coordinates
//
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global physical coordinates given by the mesh (3D): xx \in [-l, l]^3
//
// Local physical coordinates on the manifold (2D): x \in [-l, l]^2
//
// Change of coordinates matrix computed by the library:
//   (physical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 * 2]
//
// Change of coordinates x (physical 2D) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [2 * 3]
//
// Change of coordinates x (physical 2D) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j = dx_i/dxx_k * dxx_k/dX_j
//
// The quadrature data is stored in the array q_data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Qdata: w * det(dx_i/dX_j)
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeoCube)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *J = in[1], *w = in[2];
  // Outputs
  CeedScalar *q_data = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read dxxdX Jacobian entries, stored as
    // 0 3
    // 1 4
    // 2 5
    const CeedScalar dxxdX[3][2] = {
        {J[i + Q * 0], J[i + Q * 3]},
        {J[i + Q * 1], J[i + Q * 4]},
        {J[i + Q * 2], J[i + Q * 5]}
    };

    // Modulus of dxxdX column vectors
    const CeedScalar mod_g_1 = sqrt(dxxdX[0][0] * dxxdX[0][0] + dxxdX[1][0] * dxxdX[1][0] + dxxdX[2][0] * dxxdX[2][0]);
    const CeedScalar mod_g_2 = sqrt(dxxdX[0][1] * dxxdX[0][1] + dxxdX[1][1] * dxxdX[1][1] + dxxdX[2][1] * dxxdX[2][1]);

    // Use normalized column vectors of dxxdX as rows of dxdxx
    const CeedScalar dxdxx[2][3] = {
        {dxxdX[0][0] / mod_g_1, dxxdX[1][0] / mod_g_1, dxxdX[2][0] / mod_g_1},
        {dxxdX[0][1] / mod_g_2, dxxdX[1][1] / mod_g_2, dxxdX[2][1] / mod_g_2}
    };

    CeedScalar dxdX[2][2];
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        dxdX[j][k] = 0;
        for (int l = 0; l < 3; l++) dxdX[j][k] += dxdxx[j][l] * dxxdX[l][k];
      }
    }

    q_data[i + Q * 0] = (dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1]) * w[i];

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
//   v     - Output vector (test function) at quadrature points
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

#endif  // areacube_h
