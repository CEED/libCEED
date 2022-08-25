// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for building the geometric data for the 2D Poisson operator
**/

#ifndef poisson2dbuild_h
#define poisson2dbuild_h

#include <ceed.h>

CEED_QFUNCTION(Poisson2DBuild)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  // *INDENT-OFF*
  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0], *w = in[1];
  // out[0] is qdata, size (3*Q)
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Qdata stored in Voigt convention
    // J: 0 2   q_data: 0 2   adj(J):  J11 -J01
    //    1 3           2 1           -J10  J00
    const CeedScalar J00 = J[0][0][i];
    const CeedScalar J10 = J[0][1][i];
    const CeedScalar J01 = J[1][0][i];
    const CeedScalar J11 = J[1][1][i];
    const CeedScalar qw  = w[i] / (J00 * J11 - J10 * J01);
    q_data[0][i]         = qw * (J01 * J01 + J11 * J11);
    q_data[1][i]         = qw * (J00 * J00 + J10 * J10);
    q_data[2][i]         = -qw * (J00 * J01 + J10 * J11);
  }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // poisson2dbuild_h
