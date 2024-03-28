// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors (2D) for Navier-Stokes example using PETSc
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
//   detJ = J11*J22 - J21*J12
//     Jij = Jacobian entry ij
//
// Stored: w detJ
//   in q_data[0]
//
// We require the transpose of the inverse of the Jacobian to properly compute integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   dXdx_i,j = Aij / detJ
//   Aij = Adjugate ij
//
// Stored: Aij / detJ
//   in q_data[1:4] as
//   (detJ^-1) * [A11 A12]
//               [A21 A22]
// *****************************************************************************
CEED_QFUNCTION(Setup2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];
  const CeedScalar(*w)                = in[1];
  CeedScalar(*q_data)                 = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar dXdx[2][2], detJ;
    InvertMappingJacobian_2D(Q, i, J, dXdx, &detJ);
    const CeedScalar wdetJ = w[i] * detJ;

    StoredValuesPack(Q, i, 0, 1, &wdetJ, q_data);
    StoredValuesPack(Q, i, 1, 4, (const CeedScalar *)dXdx, q_data);
  }
  return 0;
}

// *****************************************************************************
// This QFunction sets up the geometric factor required for integration when reference coordinates are in 1D and the physical coordinates are in 2D
//
// Reference (parent) 1D coordinates: X
// Physical (current) 2D coordinates: x
// Change of coordinate vector:
//           J1 = dx_1/dX
//           J2 = dx_2/dX
//
// detJb is the magnitude of (J1,J2)
//
// All quadrature data is stored in 3 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of the form: int( u v )
//
// Stored: w detJb
//   in q_data_sur[0]
//
// Normal vector is given by the cross product of (J1,J2)/detJ and ẑ
//
// Stored: (J1,J2,0) x (0,0,1) / detJb
//   in q_data_sur[1:2] as
//   (detJb^-1) * [ J2 ]
//                [-J1 ]
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*w)             = in[1];
  CeedScalar(*q_data_sur)          = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar normal[2], detJb;
    NormalVectorFromdxdX_2D(Q, i, J, normal, &detJb);
    const CeedScalar wdetJ = w[i] * detJb;

    StoredValuesPack(Q, i, 0, 1, &wdetJ, q_data_sur);
    StoredValuesPack(Q, i, 1, 2, normal, q_data_sur);
  }
  return 0;
}
