// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // J: 0 2   qd: 0 2   adj(J):  J22 -J12
    //    1 3       2 1           -J21  J11
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J12 = J[i+Q*2];
    const CeedScalar J22 = J[i+Q*3];
    const CeedScalar w = qw[i] / (J11*J22 - J21*J12);
    qd[i+Q*0] =   w * (J12*J12 + J22*J22);
    qd[i+Q*2] =   w * (J11*J11 + J21*J21);
    qd[i+Q*1] = - w * (J11*J12 + J21*J22);
  }

  return 0;
}

CEED_QFUNCTION(diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar *du = in[0], *qd = in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *dv = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar du0 = du[i+Q*0];
    const CeedScalar du1 = du[i+Q*1];
    dv[i+Q*0] = qd[i+Q*0]*du0 + qd[i+Q*2]*du1;
    dv[i+Q*1] = qd[i+Q*2]*du0 + qd[i+Q*1]*du1;
  }

  return 0;
}

CEED_QFUNCTION(diff_lin)(void *ctx, const CeedInt Q,
                         const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (4*Q)
  const CeedScalar *du = in[0], *qd = in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *dv = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar du0 = du[i+Q*0];
    const CeedScalar du1 = du[i+Q*1];
    // Linearized Qdata is provided column-major
    //  0 2
    //  1 3
    dv[i+Q*0] = qd[i+Q*0]*du0 + qd[i+Q*2]*du1;
    dv[i+Q*1] = qd[i+Q*1]*du0 + qd[i+Q*3]*du1;
  }

  return 0;
}
