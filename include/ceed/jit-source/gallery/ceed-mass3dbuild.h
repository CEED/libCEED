// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for building the geometric data for the 3D mass matrix
**/

#ifndef mass3dbuild_h
#define mass3dbuild_h

#include <ceed.h>

CEED_QFUNCTION(Mass3DBuild)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // in[0] is Jacobians with shape [2, nc=3, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], *w = in[1];
  // out[0] is quadrature data, size (Q)
  CeedScalar *q_data = out[0];
  // *INDENT-ON*

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    q_data[i] = (J[0][0][i] * (J[1][1][i] * J[2][2][i] - J[1][2][i] * J[2][1][i]) - J[0][1][i] * (J[1][0][i] * J[2][2][i] - J[1][2][i] * J[2][0][i]) +
                 J[0][2][i] * (J[1][0][i] * J[2][1][i] - J[1][1][i] * J[2][0][i])) *
                w[i];
  }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // mass3dbuild_h
