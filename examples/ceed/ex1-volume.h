// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef ex1_volume_h
#define ex1_volume_h

#include <ceed.h>

/// A structure used to pass additional data to f_build_mass
struct BuildContext {
  CeedInt dim, space_dim;
};

/// libCEED Q-function for building quadrature data for a mass operator
CEED_QFUNCTION(f_build_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  struct BuildContext *bc = (struct BuildContext *)ctx;
  const CeedScalar    *J = in[0], *w = in[1];
  CeedScalar          *q_data = out[0];

  switch (bc->dim + 10 * bc->space_dim) {
    case 11:
      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { q_data[i] = J[i] * w[i]; }  // End of Quadrature Point Loop
      break;
    case 22:
      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // 0 2
        // 1 3
        q_data[i] = (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]) * w[i];
      }  // End of Quadrature Point Loop
      break;
    case 33:
      // Quadrature Point Loop
      CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
        // 0 3 6
        // 1 4 7
        // 2 5 8
        q_data[i] = (J[i + Q * 0] * (J[i + Q * 4] * J[i + Q * 8] - J[i + Q * 5] * J[i + Q * 7]) -
                     J[i + Q * 1] * (J[i + Q * 3] * J[i + Q * 8] - J[i + Q * 5] * J[i + Q * 6]) +
                     J[i + Q * 2] * (J[i + Q * 3] * J[i + Q * 7] - J[i + Q * 4] * J[i + Q * 6])) *
                    w[i];
      }  // End of Quadrature Point Loop
      break;
  }
  return 0;
}

/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(f_apply_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar       *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { v[i] = q_data[i] * u[i]; }  // End of Quadrature Point Loop
  return 0;
}

#endif  // ex1_volume_h
