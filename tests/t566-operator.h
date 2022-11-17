// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar       *rho = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]);
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  const CeedScalar(*q_data) = (const CeedScalar(*))in[0], (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedScalar num_comp    = 2;
  const CeedScalar scale[2][2] = {
      {1.0, 2.0},
      {3.0, 4.0},
  };

  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c_out = 0; c_out < num_comp; c_out++) {
      v[c_out][i] = 0.0;
      for (CeedInt c_in = 0; c_in < num_comp; c_in++) {
        v[c_out][i] += q_data[i] * u[c_in][i] * scale[c_in][c_out];
      }
    }
  }
  return 0;
}
