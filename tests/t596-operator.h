// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar       *rho = out[0];

  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]);
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  CeedInt           num_comp = *(CeedInt *)ctx;
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar       *v = out[0];

  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c = 0; c < num_comp; c++) v[i + c * Q] = rho[i] * c * u[i + c * Q];
  }
  return 0;
}
