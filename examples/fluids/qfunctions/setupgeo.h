// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors (3D) for Navier-Stokes example using PETSc

#ifndef setup_geo_h
#define setup_geo_h

#include <ceed.h>
#include <math.h>

#include "setupgeo_helpers.h"

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and coordinate transformations
//
// Reference (parent) coordinates: X
// Physical (current) coordinates: x
// Change of coordinate matrix: dxdX_{i,j} = x_{i,j} (indicial notation)
// Inverse of change of coordinate matrix: dXdx_{i,j} = (detJ^-1) * X_{i,j}
//
// All quadrature data is stored in 10 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of the form: int( v u )
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjugate ij
//
// Stored: w detJ
//   in q_data[0]
//
// We require the transpose of the inverse of the Jacobian to properly compute integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   dXdx_i,j = Aij / detJ
//
// Stored: Aij / detJ
//   in q_data[1:9] as
//   (detJ^-1) * [A11 A12 A13]
//               [A21 A22 A23]
//               [A31 A32 A33]
// *****************************************************************************
CEED_QFUNCTION(Setup)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*w)                = in[1];
  CeedScalar(*q_data)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar detJ, dXdx[3][3];
    InvertMappingJacobian_3D(Q, i, J, dXdx, &detJ);
    q_data[0][i] = w[i] * detJ;
    q_data[1][i] = dXdx[0][0];
    q_data[2][i] = dXdx[0][1];
    q_data[3][i] = dXdx[0][2];
    q_data[4][i] = dXdx[1][0];
    q_data[5][i] = dXdx[1][1];
    q_data[6][i] = dXdx[1][2];
    q_data[7][i] = dXdx[2][0];
    q_data[8][i] = dXdx[2][1];
    q_data[9][i] = dXdx[2][2];
  }
  return 0;
}

// *****************************************************************************
// This QFunction sets up the geometric factor required for integration when reference coordinates are in 2D and the physical coordinates are in 3D
//
// Reference (parent) 2D coordinates: X
// Physical (current) 3D coordinates: x
// Change of coordinate matrix:
//   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
// Inverse change of coordinate matrix:
//   dXdx_{i,j} = dX_i/dx_j (indicial notation) [2 * 3]
//
// (J1,J2,J3) is given by the cross product of the columns of dxdX_{i,j}
//
// detJb is the magnitude of (J1,J2,J3)
//
// dXdx is calculated via Moore–Penrose inverse:
//
//   dX_i/dx_j = (dxdX^T dxdX)^(-1) dxdX
//             = (dx_l/dX_i * dx_l/dX_k)^(-1) dx_j/dX_k
//
// All quadrature data is stored in 10 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Stored: w detJb
//   in q_data_sur[0]
//
// Normal vector = (J1,J2,J3) / detJb
//
//   - TODO Could possibly remove normal vector, as it could be calculated in the Qfunction from dXdx
//    See https://github.com/CEED/libCEED/pull/868#discussion_r871979484
// Stored: (J1,J2,J3) / detJb
//   in q_data_sur[1:3] as
//   (detJb^-1) * [ J1 ]
//                [ J2 ]
//                [ J3 ]
//
// Stored: dXdx_{i,j}
//   in q_data_sur[4:9] as
//    [dXdx_11 dXdx_12 dXdx_13]
//    [dXdx_21 dXdx_22 dXdx_23]
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*w)                = in[1];
  CeedScalar(*q_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar detJb, normal[3], dXdx[2][3];

    NormalVectorFromdxdX_3D(Q, i, J, normal, &detJb);
    q_data_sur[0][i] = w[i] * detJb;
    q_data_sur[1][i] = normal[0];
    q_data_sur[2][i] = normal[1];
    q_data_sur[3][i] = normal[2];

    InvertBoundaryMappingJacobian_3D(Q, i, J, dXdx);
    q_data_sur[4][i] = dXdx[0][0];
    q_data_sur[5][i] = dXdx[0][1];
    q_data_sur[6][i] = dXdx[0][2];
    q_data_sur[7][i] = dXdx[1][0];
    q_data_sur[8][i] = dXdx[1][1];
    q_data_sur[9][i] = dXdx[1][2];
  }
  return 0;
}

// *****************************************************************************

#endif  // setup_geo_h
