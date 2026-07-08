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
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar       *rho = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]);
  }
  return 0;
}

CEED_QFUNCTION(multi_basis)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)        = (const CeedScalar(*))in[0];
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*p)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*q)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[1];

  const Context_t *context = (Context_t *)ctx;

  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar mass_u = 0;
    for (CeedInt j = 0; j < context->num_comp_u; j++) {
      v[j][i] = q_data[i] * u[j][i];
      mass_u += v[j][i];
    }
    for (CeedInt j = 0; j < context->num_comp_p; j++) {
      q[j][i] = q_data[i] * p[j][i] + (j + 1) * mass_u;
    }
  }
  return 0;
}
