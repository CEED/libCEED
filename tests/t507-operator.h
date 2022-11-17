// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar       *rho = out[0];

  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * dxdX[i];
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  const CeedScalar *rho = in[0], (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  for (CeedInt i = 0; i < Q; i++) {
    v[0][i] = rho[i] * u[0][i];
    v[1][i] = rho[i] * u[1][i];
  }
  return 0;
}
