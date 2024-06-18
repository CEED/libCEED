// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Geometric factors (3D) for Navier-Stokes example using PETSc
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
  CeedScalar(*q_data)                 = out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar detJ, dXdx[3][3];
    InvertMappingJacobian_3D(Q, i, J, dXdx, &detJ);
    const CeedScalar wdetJ = w[i] * detJ;

    StoredValuesPack(Q, i, 0, 1, &wdetJ, q_data);
    StoredValuesPack(Q, i, 1, 9, (const CeedScalar *)dXdx, q_data);
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


/**
  @brief Compute geometric factors for integration, gradient transformations, and coordinate transformations on element faces.

  Reference (parent) 2D coordinates are given by `X` and physical (current) 3D coordinates are given by `x`.
  The change of coordinate matrix is given by`dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]`.

  `(N_1, N_2, N_3)` is given by the cross product of the columns of `dxdX_{i,j}`.

  `detNb` is the magnitude of `(N_1, N_2, N_3)`.

  @param[in]   ctx  QFunction context, unused
  @param[in]   Q    Number of quadrature points
  @param[in]   in   Input arrays
                      - 0 - Jacobian of cell coordinates
                      - 1 - Jacobian of face coordinates
                      - 2 - quadrature weights
  @param[out]  out  Output array
                      - 0 - qdata, `w detNb`, `dXdx`, and `N`

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION(SetupBoundaryGradient)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*J_cell)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*J_face)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*w)                     = in[2];

  // Outputs
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const CeedInt dim = 3;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // N_1, N_2, and N_3 are given by the cross product of the columns of dxdX
    CeedScalar normal[3];

    for (CeedInt j = 0; j < dim; j++) {
      // Equivalent code with no mod operations:
      // normal[j] = J_face[0][j+1]*J_face[1][j+2] - J_face[0][j+2]*J_face[1][j+1]
      normal[j] = J_face[0][(j + 1) % dim][i] * J_face[1][(j + 2) % dim][i] - J_face[0][(j + 2) % dim][i] * J_face[1][(j + 1) % dim][i];
    }
    const CeedScalar detJ_face = Norm3(normal);
    // Gradient
    CeedScalar dxdX[3][3];
    RatelGradUnpack(Q, i, J_cell, dxdX);

    const CeedScalar detJ_cell = RatelMatDetA(dxdX);
    CeedScalar       dXdx[3][3];

    RatelMatInverse(dxdX, detJ_cell, dXdx);

    // Qdata
    // -- Interp-to-Interp q_data
    q_data[0][i] = w[i] * detJ_face;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        q_data[j * 3 + k + 1][i] = dXdx[j][k];
      }
    }
    // -- Normal vector
    for (CeedInt j = 0; j < 3; j++) q_data[j + 10][i] = normal[j] / detJ_face;
  }  // End of Quadrature Point Loop
  return CEED_ERROR_SUCCESS;
}

/// @}
