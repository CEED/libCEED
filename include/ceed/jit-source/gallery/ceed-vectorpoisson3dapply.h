// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for applying the geometric data for the 3D Poisson
           on a vector system with three components
           operator
**/

#ifndef vectorpoisson3dapply_h
#define vectorpoisson3dapply_h

#include <ceed.h>

CEED_QFUNCTION(Vector3Poisson3DApply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // in[0] is gradient u, shape [3, nc=3, Q]
  // in[1] is quadrature data, size (6*Q)
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // out[0] is output to multiply against gradient v, shape [3, nc=3, Q]
  CeedScalar(*vg)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedInt dim = 3, num_comp = 3;

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 5 4
    // 5 1 3
    // 4 3 2
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[3][3] = {
        {q_data[0][i], q_data[5][i], q_data[4][i]},
        {q_data[5][i], q_data[1][i], q_data[3][i]},
        {q_data[4][i], q_data[3][i], q_data[2][i]}
    };
    // *INDENT-ON*

    // Apply Poisson Operator
    // j = direction of vg
    for (CeedInt j = 0; j < dim; j++)
      for (CeedInt c = 0; c < num_comp; c++)
        vg[j][c][i] = (ug[0][c][i] * dXdxdXdxT[0][j] + ug[1][c][i] * dXdxdXdxT[1][j] + ug[2][c][i] * dXdxdXdxT[2][j]);
  }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // vectorpoisson3dapply_h
