// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

CEED_QFUNCTION(apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u_0, shape [2, num_comp=2, Q]
  // in[1] is mass quadrature data, size (Q)
  // in[2] is Poisson quadrature data, size (Q)
  // in[3] is u_0, size (Q)
  // in[4] is u_1, size (Q)
  const CeedScalar *du_0 = in[0], *qd_mass = in[1], *qd_diff = in[2], *u_0 = in[3], *u_1 = in[4];

  // out[0] is output to multiply against v_0, size (Q)
  // out[1] is output to multiply against v_1, size (Q)
  // out[2] is output to multiply against gradient v_0, shape [2, num_comp=2, Q]
  CeedScalar *v_0 = out[0], *v_1 = out[1], *dv_0 = out[2];

  const CeedInt num_comp_0 = 2;

  // Quadrature point loop
  for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c = 0; c < num_comp_0; c++) {
      // Mass
      v_0[i + Q * c] = qd_mass[i] * (c + 1) * u_0[i + Q * c];
      // Diff
      dv_0[i + Q * (0 * num_comp_0 + c)] =
          qd_diff[i + Q * 0] * (c + 1) * du_0[i + Q * (0 * num_comp_0 + c)] + qd_diff[i + Q * 2] * du_0[i + Q * (1 * num_comp_0 + c)];
      dv_0[i + Q * (1 * num_comp_0 + c)] =
          qd_diff[i + Q * 2] * (c + 1) * du_0[i + Q * (0 * num_comp_0 + c)] + qd_diff[i + Q * 1] * du_0[i + Q * (1 * num_comp_0 + c)];
    }
    // Mass
    v_1[i] = qd_mass[i] * u_1[i];
  }

  return 0;
}
