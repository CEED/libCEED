// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

typedef struct {
  CeedInt num_comp_u;
  CeedInt num_comp_p;
} Context_t;

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is quadrature weights, size (Q)
  // in[1] is Jacobians with shape [2, nc=2, Q]
  const CeedScalar *weight = in[0];
  const CeedScalar *J      = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature point loop
  for (CeedInt i = 0; i < Q; i++) {
    // J: 0 2   qd: 0 2   adj(J):  J22 -J12
    //    1 3       2 1           -J21  J11
    const CeedScalar J11 = J[i + Q * 0];
    const CeedScalar J21 = J[i + Q * 1];
    const CeedScalar J12 = J[i + Q * 2];
    const CeedScalar J22 = J[i + Q * 3];
    const CeedScalar w   = weight[i] / (J11 * J22 - J21 * J12);
    q_data[0][i]         = w * (J12 * J12 + J22 * J22);
    q_data[1][i]         = w * (J11 * J11 + J21 * J21);
    q_data[2][i]         = -w * (J11 * J12 + J21 * J22);
  }

  return 0;
}

CEED_QFUNCTION(multi_basis_diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is quadrature data, size (3*Q)
  // in[1] is gradient u, shape [2, nc_u, Q]
  // in[2] is gradient p, shape [2, nc_p, Q]
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar *du                  = in[1];
  const CeedScalar *dp                  = in[2];

  // out[0] is output to multiply against gradient q, shape [2, nc_p, Q]
  // out[1] is output to multiply against gradient v, shape [2, nc_u, Q]
  CeedScalar *dq = out[0];
  CeedScalar *dv = out[1];

  const Context_t *context = (Context_t *)ctx;

  // Quadrature point loop
  for (CeedInt i = 0; i < Q; i++) {
    // Component loop
    for (CeedInt c = 0; c < context->num_comp_u; c++) {
      const CeedScalar du0                        = du[i + c * Q + context->num_comp_u * Q * 0];
      const CeedScalar du1                        = du[i + c * Q + context->num_comp_u * Q * 1];
      dv[i + c * Q + context->num_comp_u * Q * 0] = q_data[0][i] * du0 + q_data[2][i] * du1;
      dv[i + c * Q + context->num_comp_u * Q * 1] = q_data[2][i] * du0 + q_data[1][i] * du1;
    }

    for (CeedInt c = 0; c < context->num_comp_p; c++) {
      const CeedScalar dp0                        = dp[i + c * Q + context->num_comp_p * Q * 0];
      const CeedScalar dp1                        = dp[i + c * Q + context->num_comp_p * Q * 1];
      dq[i + c * Q + context->num_comp_p * Q * 0] = q_data[0][i] * dp0 + q_data[2][i] * dp1;
      dq[i + c * Q + context->num_comp_p * Q * 1] = q_data[2][i] * dp0 + q_data[1][i] * dp1;
    }
    // Add artificial, asymmetric coupling terms to test block assembly
    dv[i + 0 * Q + context->num_comp_u * Q * 1] += dq[i + (context->num_comp_p - 1) * Q + context->num_comp_p * Q * 0];
    dq[i] += dv[i + (context->num_comp_u - 1) * Q + context->num_comp_u * Q * 1];
  }
  return 0;
}
