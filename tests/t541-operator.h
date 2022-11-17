// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup_diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *w = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *q_data = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Qdata stored in Voigt convention
    // J: 0 2   q_data: 0 2   adj(J):  J22 -J12
    //    1 3           2 1           -J21  J11
    const CeedScalar J11 = J[i + Q * 0];
    const CeedScalar J21 = J[i + Q * 1];
    const CeedScalar J12 = J[i + Q * 2];
    const CeedScalar J22 = J[i + Q * 3];
    const CeedScalar qw  = w[i] / (J11 * J22 - J21 * J12);
    q_data[i + Q * 0]    = qw * (J12 * J12 + J22 * J22);
    q_data[i + Q * 1]    = qw * (J11 * J11 + J21 * J21);
    q_data[i + Q * 2]    = -qw * (J11 * J12 + J21 * J22);
  }  // End of Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar *ug = in[0], *q_data = in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *vg = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[2] = {ug[i + Q * 0], ug[i + Q * 1]};

    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[2][2] = {
        {q_data[i + 0 * Q], q_data[i + 2 * Q]},
        {q_data[i + 2 * Q], q_data[i + 1 * Q]}
    };
    // *INDENT-ON*

    // Apply Poisson operator
    // j = direction of vg
    for (int j = 0; j < 2; j++) vg[i + j * Q] = (du[0] * dXdxdXdxT[0][j] + du[1] * dXdxdXdxT[1][j]);
  }  // End of Quadrature Point Loop
  return 0;
}
