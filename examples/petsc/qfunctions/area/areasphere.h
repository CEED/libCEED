// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mass operator example for a scalar field on the sphere using PETSc

#ifndef areasphere_h
#define areasphere_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates have a different dimension than the one of
//   physical coordinates
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
//   dx_i/dX_j = dx_i/dxx_k * dxx_k/dX_j [3 * 2]
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
CEED_QFUNCTION(SetupMassGeoSphere)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  // Outputs
  CeedScalar *q_data = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read global Cartesian coordinates
    const CeedScalar xx[3][1] = {{X[i + 0 * Q]}, {X[i + 1 * Q]}, {X[i + 2 * Q]}};

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
    const CeedScalar mod_xx_sq = xx[0][0] * xx[0][0] + xx[1][0] * xx[1][0] + xx[2][0] * xx[2][0];
    CeedScalar       xx_sq[3][3];
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        xx_sq[j][k] = 0;
        for (int l = 0; l < 1; l++) xx_sq[j][k] += xx[j][l] * xx[k][l] / (sqrt(mod_xx_sq) * mod_xx_sq);
      }
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
    const CeedScalar J[3][1] = {{dxdX[1][0] * dxdX[2][1] - dxdX[2][0] * dxdX[1][1]},
                                {dxdX[2][0] * dxdX[0][1] - dxdX[0][0] * dxdX[2][1]},
                                {dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1]}};
    // Use the magnitude of J as our detJ (volume scaling factor)
    const CeedScalar mod_J = sqrt(J[0][0] * J[0][0] + J[1][0] * J[1][0] + J[2][0] * J[2][0]);
    q_data[i + Q * 0]      = mod_J * w[i];
  }  // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif  // areasphere_h
