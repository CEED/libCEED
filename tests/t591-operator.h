// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[2][2] = (const CeedScalar(*)[2][2])in[0];
  CeedScalar *q_data         = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { q_data[i] = J[i][0][0] * J[i][1][0] - J[i][0][1] * J[i][1][1]; }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *rho = in[1];
  CeedScalar       *v = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = rho[i] * u[i]; }
  return 0;
}
