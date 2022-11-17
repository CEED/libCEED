// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for applying the 2D Poisson operator
**/

#ifndef poisson2dapply_h
#define poisson2dapply_h

#include <ceed.h>

CEED_QFUNCTION(Poisson2DApply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar(*ug)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar(*vg)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedInt dim = 2;

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[2][2] = {
        {q_data[0][i], q_data[2][i]},
        {q_data[2][i], q_data[1][i]}
    };
    // *INDENT-ON*

    // Apply Poisson operator
    // j = direction of vg
    for (CeedInt j = 0; j < dim; j++) vg[j][i] = (ug[0][i] * dXdxdXdxT[0][j] + ug[1][i] * dXdxdXdxT[1][j]);
  }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // poisson2dapply_h
