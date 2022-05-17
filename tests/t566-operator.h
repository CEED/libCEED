// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }
  return 0;
}

CEED_QFUNCTION(mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar *v = out[0];

  const CeedScalar scale[2][2] = {
    {1.0, 2.0},
    {3.0, 4.0},
  };

  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c_out = 0; c_out < 2; c_out++) {
      v[i+Q*c_out] = 0.0;
      for (CeedInt c_in = 0; c_in < 2; c_in++) {
        v[i+Q*c_out] += rho[i] * u[i+Q*c_in] * scale[c_in][c_out];
      }
    }
  }
  return 0;
}
