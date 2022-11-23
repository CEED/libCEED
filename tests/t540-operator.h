// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *J = in[0], *weight = in[1];
  CeedScalar       *rho = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]);
  }
  return 0;
}

CEED_QFUNCTION(apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is u, size (Q)
  // in[1] is mass quadrature data, size (Q)
  const CeedScalar *u = in[0], *qd_mass = in[1];

  // out[0] is output to multiply against v, size (Q)
  CeedScalar *v = out[0];

  // Quadrature point loop
  for (CeedInt i = 0; i < Q; i++) {
    // Mass
    v[i] = qd_mass[i] * u[i];
  }

  return 0;
}
