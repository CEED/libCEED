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
#include "utils.h"

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
  const CeedScalar(*x)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*q_data)                 = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar detJ, dXdx[3][3];
    InvertMappingJacobian_3D(Q, i, J, dXdx, &detJ);
    const CeedScalar wdetJ = w[i] * detJ;

    StoredValuesPack(Q, i, 0, 1, &wdetJ, q_data);
    StoredValuesPack(Q, i, 1, 9, (const CeedScalar *)dXdx, q_data);
//    q_data[10][i]=LinearRampCoefficient(context->idl_amplitude, context->idl_length, context->idl_start    , x[0][i]);
//  idl_decay_time: 3.6e-4
//  coeff needs 1/3.6e-4=2777.78
//  idl_start: -3.1
//  idl_length: 0.2
    CeedScalar xo=x[0][i];
    q_data[10][i]=LinearRampCoefficient(2777.78,0.2,-3.1, xo);
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
  CeedScalar(*q_data_sur)             = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar detJb, normal[3], dXdx[2][3];

    NormalVectorFromdxdX_3D(Q, i, J, normal, &detJb);
    InvertBoundaryMappingJacobian_3D(Q, i, J, dXdx);
    const CeedScalar wdetJ = w[i] * detJb;

    StoredValuesPack(Q, i, 0, 1, &wdetJ, q_data_sur);
    StoredValuesPack(Q, i, 1, 3, normal, q_data_sur);
    StoredValuesPack(Q, i, 4, 6, (const CeedScalar *)dXdx, q_data_sur);
  }
  return 0;
}

// *****************************************************************************

#endif  // setup_geo_h
