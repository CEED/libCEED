// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

// Compute det(A)
CEED_QFUNCTION_HELPER CeedScalar MatDet2x2(const CeedScalar A[2][2]) { return A[0][0] * A[1][1] - A[1][0] * A[0][1]; }

// Compute alpha * A^T * B = C
CEED_QFUNCTION_HELPER int AlphaMatTransposeMatMult2x2(const CeedScalar alpha, const CeedScalar A[2][2], const CeedScalar B[2][2],
                                                      CeedScalar C[2][2]) {
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 2; m++) {
        C[j][k] += alpha * A[m][j] * B[m][k];
      }
    }
  }

  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*w) = in[0], (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
        (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {
        {dxdX[0][0][i], dxdX[1][0][i]},
        {dxdX[0][1][i], dxdX[1][1][i]}
    };
    const CeedScalar det_J = MatDet2x2(J);

    // Piola map: J^T*J*u*w/detJ
    // 1) Compute J^T * J
    CeedScalar JT_J[2][2];
    AlphaMatTransposeMatMult2x2(1., J, J, JT_J);

    // 2) Compute J^T*J*u * w /detJ
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = 0;
      for (CeedInt m = 0; m < 2; m++) v[k][i] += JT_J[k][m] * u[m][i] * w[i] / det_J;
    }
  }  // End of Quadrature Point Loop

  return 0;
}
