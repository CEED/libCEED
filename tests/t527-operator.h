// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *dxdX = in[1];
  CeedScalar       *rho = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * dxdX[i];
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u           = in[0];
  const CeedScalar *rho         = in[1];
  const CeedScalar *passive_in  = in[2];
  CeedScalar       *v           = out[0];
  CeedScalar       *passive_out = out[1];

  for (CeedInt i = 0; i < Q; i++) {
    v[i]           = rho[i] * u[i];
    passive_out[i] = passive_in[i];
  }
  return CEED_ERROR_SUCCESS;
}
