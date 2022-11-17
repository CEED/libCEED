// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  const CeedScalar *w = in[0], (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1];
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Qdata stored in Voigt convention
    // J: 0 2   q_data: 0 2   adj(J):  J11 -J01
    //    1 3           2 1           -J10  J00
    const CeedScalar J00 = J[0][0][i];
    const CeedScalar J10 = J[0][1][i];
    const CeedScalar J01 = J[1][0][i];
    const CeedScalar J11 = J[1][1][i];
    const CeedScalar qw  = w[i] / (J00 * J11 - J10 * J01);
    q_data[0][i]         = qw * (J01 * J01 + J11 * J11);
    q_data[1][i]         = qw * (J00 * J00 + J10 * J10);
    q_data[2][i]         = -qw * (J00 * J01 + J10 * J11);
  }  // End of Quadrature Point Loop

  return 0;
}

CEED_QFUNCTION(diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*ug)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1];
  CeedScalar(*vg)[2][CEED_Q_VLA] = (CeedScalar(*)[2][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedInt    dim         = 2;
  const CeedScalar num_comp    = 2;
  const CeedScalar scale[2][2] = {
      {1.0, 2.0},
      {3.0, 4.0},
  };

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
    for (CeedInt j = 0; j < dim; j++) {
      for (CeedInt k = 0; k < num_comp; k++) {
        vg[j][k][i] = (ug[0][k][i] * dXdxdXdxT[0][j] * scale[0][j] + ug[1][k][i] * dXdxdXdxT[1][j] * scale[1][j]);
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
