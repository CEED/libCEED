// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors (3D) for Navier-Stokes example using PETSc

#ifndef setupgeo_helpers_h
#define setupgeo_helpers_h

#include <ceed.h>
#include <math.h>

#include "utils.h"

/**
 * @brief Calculate dXdx from dxdX for 3D elements
 *
 * Reference (parent) coordinates: X
 * Physical (current) coordinates: x
 * Change of coordinate matrix: dxdX_{i,j} = x_{i,j} (indicial notation)
 * Inverse of change of coordinate matrix: dXdx_{i,j} = (detJ^-1) * X_{i,j}
 *
 * Determinant of Jacobian:
 *   detJ = J11*A11 + J21*A12 + J31*A13
 *     Jij = Jacobian entry ij
 *     Aij = Adjugate ij
 *
 * Inverse of Jacobian:
 *   dXdx_i,j = Aij / detJ
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] dXdx     Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void InvertMappingJacobian_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar dXdx[3][3],
                                                    CeedScalar *detJ_ptr) {
  const CeedScalar dxdX_11 = dxdX_q[0][0][i];
  const CeedScalar dxdX_21 = dxdX_q[0][1][i];
  const CeedScalar dxdX_31 = dxdX_q[0][2][i];
  const CeedScalar dxdX_12 = dxdX_q[1][0][i];
  const CeedScalar dxdX_22 = dxdX_q[1][1][i];
  const CeedScalar dxdX_32 = dxdX_q[1][2][i];
  const CeedScalar dxdX_13 = dxdX_q[2][0][i];
  const CeedScalar dxdX_23 = dxdX_q[2][1][i];
  const CeedScalar dxdX_33 = dxdX_q[2][2][i];
  const CeedScalar A11     = dxdX_22 * dxdX_33 - dxdX_23 * dxdX_32;
  const CeedScalar A12     = dxdX_13 * dxdX_32 - dxdX_12 * dxdX_33;
  const CeedScalar A13     = dxdX_12 * dxdX_23 - dxdX_13 * dxdX_22;
  const CeedScalar A21     = dxdX_23 * dxdX_31 - dxdX_21 * dxdX_33;
  const CeedScalar A22     = dxdX_11 * dxdX_33 - dxdX_13 * dxdX_31;
  const CeedScalar A23     = dxdX_13 * dxdX_21 - dxdX_11 * dxdX_23;
  const CeedScalar A31     = dxdX_21 * dxdX_32 - dxdX_22 * dxdX_31;
  const CeedScalar A32     = dxdX_12 * dxdX_31 - dxdX_11 * dxdX_32;
  const CeedScalar A33     = dxdX_11 * dxdX_22 - dxdX_12 * dxdX_21;
  const CeedScalar detJ    = dxdX_11 * A11 + dxdX_21 * A12 + dxdX_31 * A13;

  dXdx[0][0] = A11 / detJ;
  dXdx[0][1] = A12 / detJ;
  dXdx[0][2] = A13 / detJ;
  dXdx[1][0] = A21 / detJ;
  dXdx[1][1] = A22 / detJ;
  dXdx[1][2] = A23 / detJ;
  dXdx[2][0] = A31 / detJ;
  dXdx[2][1] = A32 / detJ;
  dXdx[2][2] = A33 / detJ;
  if (detJ_ptr) *detJ_ptr = detJ;
}

/**
 * @brief Calculate face element's normal vector from dxdX
 *
 * Reference (parent) 2D coordinates: X
 * Physical (current) 3D coordinates: x
 * Change of coordinate matrix:
 *   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
 * Inverse change of coordinate matrix:
 *   dXdx_{i,j} = dX_i/dx_j (indicial notation) [2 * 3]
 *
 * (J1,J2,J3) is given by the cross product of the columns of dxdX_{i,j}
 *
 * detJb is the magnitude of (J1,J2,J3)
 *
 * Normal vector = (J1,J2,J3) / detJb
 *
 * Stored: (J1,J2,J3) / detJb
 *   in q_data_sur[1:3] as
 *   (detJb^-1) * [ J1 ]
 *                [ J2 ]
 *                [ J3 ]
 *
 * @param[in]  Q        Number of quadrature points
 * @param[in]  i        Current quadrature point
 * @param[in]  dxdX_q   Mapping Jacobian (gradient of the coordinate space)
 * @param[out] normal   Inverse of mapping Jacobian at quadrature point i
 * @param[out] detJ_ptr Determinate of the Jacobian, may be NULL is not desired
 */
CEED_QFUNCTION_HELPER void NormalVectorFromdxdX_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar normal[3],
                                                   CeedScalar *detJ_ptr) {
  const CeedScalar dxdX[3][2] = {
      {dxdX_q[0][0][i], dxdX_q[1][0][i]},
      {dxdX_q[0][1][i], dxdX_q[1][1][i]},
      {dxdX_q[0][2][i], dxdX_q[1][2][i]}
  };
  // J1, J2, and J3 are given by the cross product of the columns of dxdX
  const CeedScalar J1 = dxdX[1][0] * dxdX[2][1] - dxdX[2][0] * dxdX[1][1];
  const CeedScalar J2 = dxdX[2][0] * dxdX[0][1] - dxdX[0][0] * dxdX[2][1];
  const CeedScalar J3 = dxdX[0][0] * dxdX[1][1] - dxdX[1][0] * dxdX[0][1];

  const CeedScalar detJ = sqrt(J1 * J1 + J2 * J2 + J3 * J3);

  normal[0] = J1 / detJ;
  normal[1] = J2 / detJ;
  normal[2] = J3 / detJ;
  if (detJ_ptr) *detJ_ptr = detJ;
}

/**
 * @brief Calculate inverse of mapping Jacobian, (dxdX)^-1
 *
 * Reference (parent) 2D coordinates: X
 * Physical (current) 3D coordinates: x
 * Change of coordinate matrix:
 *   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
 * Inverse change of coordinate matrix:
 *   dXdx_{i,j} = dX_i/dx_j (indicial notation) [2 * 3]
 *
 * dXdx is calculated via Moore–Penrose inverse:
 *
 *   dX_i/dx_j = (dxdX^T dxdX)^(-1) dxdX
 *             = (dx_l/dX_i * dx_l/dX_k)^(-1) dx_j/dX_k
 *
 * @param[in]  Q      Number of quadrature points
 * @param[in]  i      Current quadrature point
 * @param[in]  dxdX_q Mapping Jacobian (gradient of the coordinate space)
 * @param[out] dXdx   Inverse of mapping Jacobian at quadrature point i
 */
CEED_QFUNCTION_HELPER void InvertBoundaryMappingJacobian_3D(CeedInt Q, CeedInt i, const CeedScalar (*dxdX_q)[3][CEED_Q_VLA], CeedScalar dXdx[2][3]) {
  const CeedScalar dxdX[3][2] = {
      {dxdX_q[0][0][i], dxdX_q[1][0][i]},
      {dxdX_q[0][1][i], dxdX_q[1][1][i]},
      {dxdX_q[0][2][i], dxdX_q[1][2][i]}
  };

  // dxdX_k,j * dxdX_j,k
  CeedScalar dxdXTdxdX[2][2] = {{0.}};
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      for (CeedInt l = 0; l < 3; l++) dxdXTdxdX[j][k] += dxdX[l][j] * dxdX[l][k];
    }
  }

  const CeedScalar detdxdXTdxdX = dxdXTdxdX[0][0] * dxdXTdxdX[1][1] - dxdXTdxdX[1][0] * dxdXTdxdX[0][1];

  // Compute inverse of dxdXTdxdX
  CeedScalar dxdXTdxdX_inv[2][2];
  dxdXTdxdX_inv[0][0] = dxdXTdxdX[1][1] / detdxdXTdxdX;
  dxdXTdxdX_inv[0][1] = -dxdXTdxdX[0][1] / detdxdXTdxdX;
  dxdXTdxdX_inv[1][0] = -dxdXTdxdX[1][0] / detdxdXTdxdX;
  dxdXTdxdX_inv[1][1] = dxdXTdxdX[0][0] / detdxdXTdxdX;

  // Compute dXdx from dxdXTdxdX^-1 and dxdX
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      dXdx[j][k] = 0;
      for (CeedInt l = 0; l < 2; l++) dXdx[j][k] += dxdXTdxdX_inv[l][j] * dxdX[k][l];
    }
  }
}

#endif  // setupgeo_helpers_h
