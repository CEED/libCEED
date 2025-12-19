// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed/types.h>
#include "ex-common.h"

/// libCEED Q-function for building quadrature data for a mass operator
CEED_QFUNCTION(build_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, dim, Q]
  // in[1] is quadrature weights with shape [1, Q]
  const CeedScalar    *w          = in[1];
  CeedScalar          *q_data     = out[0];
  struct BuildContext *build_data = (struct BuildContext *)ctx;

  switch (build_data->dim + 10 * build_data->space_dim) {
    case 11: {
      const CeedScalar(*J)[1][CEED_Q_VLA] = (const CeedScalar(*)[1][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { q_data[i] = J[0][0][i] * w[i]; }  // End of Quadrature Point Loop
    } break;
    case 22: {
      const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        q_data[i] = (J[0][0][i] * J[1][1][i] - J[0][1][i] * J[1][0][i]) * w[i];
      }  // End of Quadrature Point Loop
    } break;
    case 33: {
      const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];

      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        q_data[i] =
            (J[0][0][i] * (J[1][1][i] * J[2][2][i] - J[1][2][i] * J[2][1][i]) - J[0][1][i] * (J[1][0][i] * J[2][2][i] - J[1][2][i] * J[2][0][i]) +
             J[0][2][i] * (J[1][0][i] * J[2][1][i] - J[1][1][i] * J[2][0][i])) *
            w[i];
      }  // End of Quadrature Point Loop
    } break;
  }
  return CEED_ERROR_SUCCESS;
}

/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(apply_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0], out[0] are solution variables with shape [1, Q]
  // in[1] is quadrature data with shape [1, Q]
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar       *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = q_data[i] * u[i]; }  // End of Quadrature Point Loop
  return CEED_ERROR_SUCCESS;
}
